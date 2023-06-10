import enum
import os
import numpy as np
import cv2
import argparse
import torch
from model import TeethKptNet
from utils import crop_image, get_weighted_average_max
import json

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main(args):
    base_options = python.BaseOptions(
        model_asset_path="data/face_landmarker_v2_with_blendshapes.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    net = TeethKptNet(n_kpt=4)
    net.eval()
    net.cuda()
    state_dict = torch.load(args.ckp, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)

    result = {}
    for cam_idx in [0, 1]:
        result[f"cam{cam_idx}"] = []
        img_dir = os.path.join(args.data, f"cam{cam_idx}")
        for i in sorted(os.listdir(img_dir)):
            image_np = np.array(cv2.imread(os.path.join(img_dir, i)))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            detection_result = detector.detect(image)
            face_landmarks = []
            for j, lm in enumerate(detection_result.face_landmarks[0]):
                x = int(lm.x * image_np.shape[1])
                y = int(lm.y * image_np.shape[0])
                face_landmarks.append((x, y))
            face_landmarks = np.array(face_landmarks)

            p1 = face_landmarks[178]
            p2 = face_landmarks[317]
            center = (p1 + p2) * 0.5
            side_len = np.linalg.norm(p2 - p1) * 2.0
            x_min = int(center[0] - side_len * 0.5)
            y_min = int(center[1] - side_len * 0.5)
            x_max = int(center[0] + side_len * 0.5)
            y_max = int(center[1] + side_len * 0.5)

            image_np_cropped = crop_image(image_np, [x_min, y_min, x_max, y_max])
            scale_x = 256.0 / image_np_cropped.shape[1]
            scale_y = 256.0 / image_np_cropped.shape[0]
            image_np_cropped = cv2.resize(image_np_cropped, (256, 256))

            image_tensor = (
                torch.tensor(image_np_cropped / 256.0)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .cuda()
            )
            heatmap_np = net(image_tensor)[0].detach().cpu().numpy()

            keypoints = []
            # Iterate over all channels to find argmax.
            for c in range(heatmap_np.shape[0]):
                hmap = heatmap_np[c]
                max_loc = get_weighted_average_max(hmap)
                # the heatmap is 1/4 size of original image, scale back
                max_loc_scaled = (max_loc[0] * 4, max_loc[1] * 4)  # (x, y) format
                keypoints.append(max_loc_scaled)
            keypoints = np.array(keypoints)
            keypoints[:, 0] = keypoints[:, 0] / scale_x + x_min
            keypoints[:, 1] = keypoints[:, 1] / scale_y + y_min

            result[f"cam{cam_idx}"].append(
                {
                    "image_name": i,
                    "face_landmarks": face_landmarks.tolist(),
                    "keypoints": keypoints.tolist(),
                }
            )

            # for j, point in enumerate(face_landmarks[[178, 317]]):
            #     cv2.circle(image_np, point, 3, (0, 255, 0), -1)
            #     cv2.putText(
            #         image_np,
            #         str(j),
            #         (point[0] - 10, point[1] - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.9,
            #         (36, 255, 12),
            #         2,
            #     )

            for j, point in enumerate(keypoints):
                point = (int(point[0]), int(point[1]))
                cv2.circle(image_np, point, 3, (36, 255, 12), -1)
                cv2.putText(
                    image_np,
                    str(j),
                    (point[0] - 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )
            cv2.rectangle(image_np, (x_min, y_min), (x_max, y_max), (36, 255, 12), thickness=2)
            cv2.imshow('img', image_np)
            cv2.waitKey(30)

    json.dump(result, open("result.json", "w"), indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/demo/images", help="dataset root")
    parser.add_argument(
        "--ckp",
        default="./outputs/checkpoint/checkpoint_099.pth",
        help="path to checkpoint",
    )

    args = parser.parse_args()

    main(args)
