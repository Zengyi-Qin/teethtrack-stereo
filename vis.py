import json
import numpy as np
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise

from scipy.spatial.transform import Rotation


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps 
    corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def get_face_template(landmarks):
    # Initialize a matrix to hold the registered landmarks
    registered_landmarks = np.zeros_like(landmarks)

    # Use the first frame as the target for registration
    target = landmarks[0]

    # Register all frames to the target
    for i in range(landmarks.shape[0]):
        _, R, t = best_fit_transform(landmarks[i], target)
        registered_landmarks[i] = landmarks[i].dot(R.T) + t

    # Compute the mean landmark positions to serve as a template
    template = registered_landmarks.mean(axis=0)

    return template


def register_template(landmarks, template):
    _, R, t = best_fit_transform(template, landmarks[0])
    template_aligned = template.dot(R.T) + t
    R_seq = []
    t_seq = []
    for i in range(landmarks.shape[0]):
        _, R, t = best_fit_transform(template_aligned, landmarks[i])
        R_seq.append(R)
        t_seq.append(t)

    return np.array(R_seq), np.array(t_seq)


def transform_coords(points, R_seq, t_seq):
    # Check that the dimensions are compatible
    assert points.shape[0] == R_seq.shape[0] == t_seq.shape[0]
    assert points.shape[2] == R_seq.shape[2] == R_seq.shape[1] == t_seq.shape[1] == 3

    T = points.shape[0]
    N = points.shape[1]

    # Initialize transformed points array
    transformed_points = np.empty_like(points)

    for t in range(T):
        # Reshape points for matrix multiplication
        points_reshape = points[t].T.reshape(3, N)

        # Apply rotation and translation
        transformed_points[t] = np.dot(
            np.linalg.inv(R_seq[t]), points_reshape - t_seq[t].reshape(3, 1)
        )

        # Reshape back to original shape
        transformed_points[t] = transformed_points[t].T.reshape(N, 3)

    return transformed_points


def f_state(state, dt):
    # Check that the state vector has the correct size
    assert state.shape == (12,)

    # Split the state vector into position, velocity, orientation and angular velocity
    pos = state[0:3]
    vel = state[3:6]
    orientation = state[6:9]
    angular_vel = state[9:]

    # Predict next state by assuming constant velocity and angular velocity
    next_pos = pos + vel * dt
    next_orientation = orientation + angular_vel * dt

    # The velocities remain unchanged under the constant velocity assumption
    next_vel = vel
    next_angular_vel = angular_vel

    # Concatenate to form the next state vector
    next_state = np.concatenate(
        (next_pos, next_vel, next_orientation, next_angular_vel)
    )

    return next_state


def h_observation(state):
    # Check that the state vector has the correct size
    assert state.shape == (12,)

    # Split the state vector into position, velocity, orientation and angular velocity
    pos = state[0:3]
    orientation = state[6:9]

    z = np.concatenate([pos, orientation])
    return z


def rotation_translation_to_vector(R_seq, t_seq):
    T = R_seq.shape[0]
    vec_seq = np.empty((T, 6))

    for t in range(T):
        rotation = Rotation.from_matrix(R_seq[t])
        euler_angles = rotation.as_euler("xyz", degrees=False)

        vec_seq[t, :3] = t_seq[t]
        vec_seq[t, 3:] = euler_angles

    return vec_seq


def vector_to_rotation_translation(vec_seq):
    T = vec_seq.shape[0]
    R_seq = np.empty((T, 3, 3))
    t_seq = np.empty((T, 3))

    for t in range(T):
        # Convert the Euler angles back into a rotation matrix
        euler_angles = vec_seq[t, 3:]
        rotation = Rotation.from_euler("xyz", euler_angles, degrees=False)
        R_seq[t] = rotation.as_matrix()

        # Extract the translation vector
        t_seq[t] = vec_seq[t, :3]

    return R_seq, t_seq


def estimate_variance(state):
    m = (state[0:-2] + state[1:-1] + state[2:]) / 3.0
    v = ((state[0:-2] - m) ** 2 + (state[1:-1] - m) ** 2 + (state[2:] - m) ** 2) / 3.0
    v = np.mean(v, axis=0)

    return v


def ukf_filter(R_seq, t_seq, dt=0.04):
    obs = rotation_translation_to_vector(R_seq, t_seq)

    sigmas = MerweScaledSigmaPoints(12, alpha=0.1, beta=2.0, kappa=0)
    ukf = UKF(dim_x=12, dim_z=6, fx=f_state, hx=h_observation, dt=dt, points=sigmas)

    ukf.x = np.concatenate([obs[0, 0:3], np.zeros(3), obs[0, 3:6], np.zeros(3)])
    ukf.P = np.diag(
        [
            0.05,
            0.05,
            0.05,
            0.01,
            0.01,
            0.01,
            0.0005,
            0.0005,
            0.0005,
            0.0005,
            0.0005,
            0.0005,
        ]
    )
    ukf.R = np.diag([0.05, 0.05, 0.05, 0.0005, 0.0005, 0.0005])

    Q = np.zeros((12, 12))
    Q[0:6, 0:6] = Q_discrete_white_noise(
        dim=2, dt=dt, var=0.1, block_size=3, order_by_dim=False
    )
    Q[6:12, 6:12] = Q_discrete_white_noise(
        dim=2, dt=dt, var=0.01, block_size=3, order_by_dim=False
    )
    ukf.Q = Q

    mean, cov = ukf.batch_filter(obs)

    mean = np.concatenate([mean[:, 0:3], mean[:, 6:9]], axis=1)
    R_pred, t_pred = vector_to_rotation_translation(mean)

    return R_pred, t_pred


def transform_coords(points, R_seq, t_seq):
    # Check that the dimensions are compatible
    assert points.shape[0] == R_seq.shape[0] == t_seq.shape[0]
    assert points.shape[2] == R_seq.shape[2] == R_seq.shape[1] == t_seq.shape[1] == 3

    T = points.shape[0]
    N = points.shape[1]

    # Initialize transformed points array
    transformed_points = np.empty_like(points)

    for t in range(T):
        # Reshape points for matrix multiplication
        points_reshape = points[t].T.reshape(3, N)
        # Apply rotation and translation
        points_reshape = np.dot(
            np.linalg.inv(R_seq[t]), points_reshape - t_seq[t].reshape(3, 1)
        )

        # Reshape back to original shape
        transformed_points[t] = points_reshape.T.reshape(N, 3)

    return transformed_points


def smooth_points(points, k):
    # Create a uniform kernel for convolution

    # Apply convolution along the time axis
    smoothed_points = np.zeros_like(points)
    for i in range(points.shape[0]):
        start = max(0, i - k // 2)
        end = min(points.shape[0], i + k // 2)
        smoothed_points[i] = np.mean(points[start:end], axis=0)

    return smoothed_points


def main():
    face_landmarks_seq = np.load(open("face_landmarks_seq.npy", "rb"))
    keypoints_seq = np.load(open("keypoints_seq.npy", "rb"))

    template = get_face_template(face_landmarks_seq)

    R_seq, t_seq = register_template(face_landmarks_seq, template)

    R_pred, t_pred = ukf_filter(R_seq, t_seq)

    keypoints_transformed = transform_coords(keypoints_seq, R_pred, t_pred)
    face_landmarks_transformed = transform_coords(face_landmarks_seq, R_pred, t_pred)

    num_frames = len(face_landmarks_seq)
    # Create the 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    def update_axes(frame):
        ax.cla()
        R, t = R_seq[frame], t_seq[frame]
        ax.scatter(*t, color="g")

        # Plot each axis of this transform
        for i, color in enumerate(["g", "g", "g"]):
            # Calculate the end point of this axis
            end = t + R[:, i] * 3
            ax.quiver(*t, *(end - t), color=color)

        R, t = R_pred[frame], t_pred[frame]
        ax.scatter(*t, color="r")

        # Plot each axis of this transform
        for i, color in enumerate(["r", "r", "r"]):
            # Calculate the end point of this axis
            end = t + R[:, i] * 3
            ax.quiver(*t, *(end - t), color=color)

        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])
        ax.set_zlim([-20, 20])

    face_landmarks_vis = smooth_points(face_landmarks_transformed, 10)
    keypoints_vis = smooth_points(keypoints_transformed, 10)

    # Function to update the scatter plot for each frame
    def update(frame):
        ax.cla()  # Clear the plot
        ax.scatter(
            face_landmarks_vis[frame][:, 0],
            face_landmarks_vis[frame][:, 1],
            face_landmarks_vis[frame][:, 2] * 1.5,
            s=5,
            color="grey",
        )

        ax.scatter(
            keypoints_vis[frame][:, 0],
            keypoints_vis[frame][:, 1],
            keypoints_vis[frame][:, 2] * 1.5,
            s=30,
            color="darkorange",
        )
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-100, 100])

        ax.view_init(elev=0, azim=frame)
        ax.set_axis_off()

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=20)

    # ani = animation.FuncAnimation(fig, update_axes, frames=num_frames, interval=20)

    plt.show()


main()
