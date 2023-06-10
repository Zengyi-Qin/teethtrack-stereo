import json
import numpy as np
from utils import triangulate_by_dlt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import pdb


def main():
    inference_res = json.load(open("./data/demo/result.json"))
    camera_param = json.load(open("./data/demo/camera_param.json"))
    intr = np.array(camera_param["intrinsics"])
    extr = np.array(camera_param["extrinsics"])
    w2p_mats = np.matmul(intr, extr)

    face_landmarks_seq = []
    keypoints_seq = []

    n_frames = len(inference_res['cam0'])
    for i in range(n_frames):
        lm = np.array(
            [
                inference_res["cam0"][i]["face_landmarks"],
                inference_res["cam1"][i]["face_landmarks"],
            ]
        )
        face_landmarks = triangulate_by_dlt(lm, w2p_mats)
        kp = np.array(
            [
                inference_res["cam0"][i]["keypoints"],
                inference_res["cam1"][i]["keypoints"],
            ]
        )
        keypoints = triangulate_by_dlt(kp, w2p_mats)
        face_landmarks_seq.append(face_landmarks)
        keypoints_seq.append(keypoints)

    center = np.mean(face_landmarks_seq, axis=(0, 1))
    rot_mat = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    face_landmarks_seq = np.matmul(face_landmarks_seq - center, rot_mat)
    keypoints_seq = np.matmul(keypoints_seq - center, rot_mat)

    np.save(open('face_landmarks_seq.npy', 'wb'), face_landmarks_seq)
    np.save(open('keypoints_seq.npy', 'wb'), keypoints_seq)

    num_frames = len(face_landmarks_seq)
    # Create the 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Function to update the scatter plot for each frame
    def update(frame):
        ax.cla()  # Clear the plot
        ax.scatter(
            face_landmarks_seq[frame][:, 0],
            face_landmarks_seq[frame][:, 1],
            face_landmarks_seq[frame][:, 2] * 1.5,
            s=10,
            color='grey'
        )
        ax.scatter(
            keypoints_seq[frame][:, 0],
            keypoints_seq[frame][:, 1],
            keypoints_seq[frame][:, 2] * 1.5,
            s=10,
            color='red'
        )
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-100, 100])

        ax.view_init(elev=0, azim=frame)
        

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20)

    plt.show()


main()
