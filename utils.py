import cv2
import numpy as np


def convert_coco(annotation, n_kpt):
    # Create a dictionary that maps image ids to image names
    id_to_name = {image["id"]: image["file_name"] for image in annotation["images"]}

    # Initialize the final dictionary
    final_dict = {
        name: {"keypoints": [[0, 0, 0] for _ in range(n_kpt)]}
        for name in id_to_name.values()
    }

    # Populate the final dictionary
    for ann in annotation["annotations"]:
        image_name = id_to_name[ann["image_id"]]
        category_id = ann["category_id"] - 1  # Subtract 1 to make it 0-indexed
        final_dict[image_name]["keypoints"][category_id] = ann["keypoints"]

    return final_dict


def affine_transform_and_crop(image, keypoints, output_size):
    # Compute mean of keypoints
    mean_keypoint = np.mean(keypoints, axis=0)

    # Translate image so that mean of keypoints is at center of image
    rows, cols, _ = image.shape
    translate_to_center = np.float32(
        [
            [1, 0, cols / 2 - mean_keypoint[0]],
            [0, 1, rows / 2 - mean_keypoint[1]],
            [0, 0, 1],
        ]
    )

    # Randomly rotate image between -45 and +45 degrees
    angle = np.random.uniform(-45, 45)
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotation_matrix = np.vstack([rotation_matrix, [0, 0, 1]])

    # Combine transformations into a single matrix by multiplying them
    combined_matrix = np.dot(rotation_matrix, translate_to_center)

    # Convert back to a 2x3 matrix
    combined_matrix = combined_matrix[:2, :]

    # Apply combined transformations
    image = cv2.warpAffine(image, combined_matrix, (cols, rows))

    # Transform the keypoints
    transformed_keypoints = cv2.transform(np.float32([keypoints]), combined_matrix)[0]

    # Calculate distance y between first and fourth keypoints
    kpt_min = np.array(
        [transformed_keypoints[:, 0].min(), transformed_keypoints[:, 1].min()]
    )
    kpt_max = np.array(
        [transformed_keypoints[:, 0].max(), transformed_keypoints[:, 1].max()]
    )

    # Choose random side length between 1.2*y and 3.0*y
    side_length = np.random.uniform(1.5, 4.0) * np.linalg.norm(kpt_max - kpt_min)

    # Randomize the location of the keypoints in the crop while
    # ensuring all keypoints are within the crop
    min_u = max(0, max(transformed_keypoints[:, 0]) - side_length * 0.8)
    min_v = max(0, max(transformed_keypoints[:, 1]) - side_length * 0.8)
    max_u = min(cols, min(transformed_keypoints[:, 0]) + side_length * 0.8)
    max_v = min(rows, min(transformed_keypoints[:, 1]) + side_length * 0.8)

    u1 = np.random.uniform(min_u, max_u - side_length)
    v1 = np.random.uniform(min_v, max_v - side_length)

    # Crop the image
    image = image[int(v1) : int(v1 + side_length), int(u1) : int(u1 + side_length)]

    # Adjust the keypoints for the crop
    transformed_keypoints -= np.array([u1, v1])

    # Resize image to the output size and adjust keypoints accordingly
    height, width, _ = image.shape
    image = cv2.resize(image, output_size)
    transformed_keypoints[:, 0] *= output_size[0] / width
    transformed_keypoints[:, 1] *= output_size[1] / height

    return image, transformed_keypoints


def generate_keypoint_maps(image, keypoints, conf, stride, sigma):
    n_keypoints = keypoints.shape[0]
    n_rows, n_cols, _ = image.shape
    keypoint_maps = np.zeros(
        shape=(n_keypoints, n_rows // stride, n_cols // stride), dtype=np.float32
    )

    for keypoint_idx in range(n_keypoints):
        keypoint = keypoints[keypoint_idx]
        if conf[keypoint_idx] > 0:
            add_gaussian(
                keypoint_maps[keypoint_idx], keypoint[0], keypoint[1], stride, sigma
            )

    return keypoint_maps


def add_gaussian(keypoint_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = keypoint_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + (
                map_y * stride + shift - y
            ) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            keypoint_map[map_y, map_x] += np.exp(-exponent)
            if keypoint_map[map_y, map_x] > 1:
                keypoint_map[map_y, map_x] = 1


def draw_keypoints(image, heatmap_np, stride=4):
    # Assuming that the heatmap is a torch tensor.
    # Convert heatmap to numpy for easier handling
    keypoints = []

    # Iterate over all channels to find argmax.
    for c in range(heatmap_np.shape[0]):
        hmap = heatmap_np[c]
        max_loc = np.unravel_index(np.argmax(hmap), hmap.shape)
        # the heatmap is 1/4 size of original image, scale back
        max_loc_scaled = (max_loc[1] * stride, max_loc[0] * stride)  # (x, y) format
        keypoints.append(max_loc_scaled)

    # Draw keypoints on the image
    for i, point in enumerate(keypoints):
        cv2.circle(image, point, 3, (0, 255, 0), -1)
        cv2.putText(
            image,
            str(i),
            (point[0] - 10, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (36, 255, 12),
            2,
        )

    return image


def crop_image(image, coords):
    """
    coords: [xmin, ymin, xmax, ymax]
    """

    # Find the size of the image
    img_height, img_width = image.shape[:2]

    # Determine how much padding is needed
    pad_x_min = max(0, -coords[0])
    pad_y_min = max(0, -coords[1])
    pad_x_max = max(0, coords[2] - img_width)
    pad_y_max = max(0, coords[3] - img_height)

    # Create the padding for each side of the image
    padding = [(pad_y_min, pad_y_max), (pad_x_min, pad_x_max), (0, 0)]

    # Pad the image
    padded_image = np.pad(image, padding, mode="constant")

    # Adjust the coordinates for the padded image
    coords_padded = [
        coords[0] + pad_x_min,
        coords[1] + pad_y_min,
        coords[2] + pad_x_min,
        coords[3] + pad_y_min,
    ]

    # Crop the image
    cropped_image = padded_image[
        coords_padded[1] : coords_padded[3], coords_padded[0] : coords_padded[2]
    ]

    return cropped_image


def get_weighted_average_max(hmap, window_size=5):
    # Compute the coarse location of the maximum
    max_loc_coarse = np.unravel_index(np.argmax(hmap), hmap.shape)

    # Define the window for the weighted average
    start_y = max(0, max_loc_coarse[0] - window_size // 2)
    end_y = min(hmap.shape[0], max_loc_coarse[0] + window_size // 2 + 1)
    start_x = max(0, max_loc_coarse[1] - window_size // 2)
    end_x = min(hmap.shape[1], max_loc_coarse[1] + window_size // 2 + 1)

    # Extract the window from the heatmap
    window = hmap[start_y:end_y, start_x:end_x]

    # Compute the coordinates grid for the window
    x = np.arange(start_x, end_x)
    y = np.arange(start_y, end_y)
    xv, yv = np.meshgrid(x, y)

    # Compute the weighted average of the coordinates
    total_weight = window.sum()
    x_avg = (window * xv).sum() / total_weight
    y_avg = (window * yv).sum() / total_weight

    return x_avg, y_avg


def triangulate_by_dlt(kpts_pix, w2p_mats, confs=None):
    """
    Triangulate keypoints in pixel coordinate from multiple views
    using direct linear transformation (DLT)

    Args:
        kpts_pix : Numpy array (n_view, n_keypoint, 2)
            Keypoints in pixel coordinate from images.
        w2p_mats : (n_view, 3, 4)
        confs : Optional[Numpy array] (n_view, n_keypoint)

    Returns:
        kpts_wld : Numpy array (n_keypoint, 3)
    """
    n_view, n_kpt, _ = kpts_pix.shape

    if confs is None:
        confs = np.ones((n_view, n_kpt))
    w2p_mats_3 = np.concatenate([w2p_mats[:, 2, :].reshape(n_view, 1, 4)] * 2, 1)

    kpts_wld = []
    for kpt_id in range(n_kpt):
        A = (
            np.multiply(kpts_pix[:, kpt_id, :].reshape([n_view, 2, 1]), w2p_mats_3)
            - w2p_mats[:, :2, :]
        ) * confs[:, kpt_id].reshape([n_view, 1, 1])

        A = A.reshape([2 * n_view, 4])

        u, s, vh = np.linalg.svd(A, full_matrices=False)
        kpt_wld_homo = vh[3, :]

        kpt_wld = kpt_wld_homo[0:3] / (kpt_wld_homo[3] + 1e-9)

        kpts_wld.append(kpt_wld)

    return np.array(kpts_wld)
