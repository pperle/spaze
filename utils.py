from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data

from dataset.dataset_utils import CameraArcane


def fix_qt() -> None:
    """
    https://forum.qt.io/post/655935
    delete `QT_* ` env vars set by cv2

    :return: None
    """
    import os
    for k, v in os.environ.items():
        if k.startswith("QT_") and "cv2" in v:
            del os.environ[k]


def compute_basis(cameras_aligned: List[CameraArcane], inverse_rotation: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    from Erik Lindén

    :param cameras_aligned:
    :param inverse_rotation: (Tensor[batch_size, 3, 3])
    :param point: 2D gaze origin in the normalized image, predicted by the network (o_2D) (Tensor[batch_size, 2])
    :return: (Tensor[batch_size, 3, 3])
    """
    device = inverse_rotation.device
    batch_size = inverse_rotation.shape[0]

    f = torch.stack([camera_aligned.focal_length for camera_aligned in cameras_aligned]).unsqueeze(2).float()
    fx, fy = f[:, 0], f[:, 1]

    uv = (point - torch.stack([camera_aligned.principal_point for camera_aligned in cameras_aligned])).unsqueeze(2).float()
    u, v = uv[:, 0], uv[:, 1]

    one = torch.ones(batch_size).unsqueeze(1).to(device)
    unit_y = torch.Tensor([0, 1, 0] * batch_size).reshape(-1, 3).to(device)

    # `basis_z` is the back-projection and `basis_x` is perpendicular to the y-axis of the aligned camera.
    basis_z = F.normalize(torch.cat([u / fx, v / fy, one], dim=1), dim=1)
    basis_x = F.normalize(torch.cross(unit_y, basis_z, dim=1), dim=1)
    basis_y = F.normalize(torch.cross(basis_z, basis_x, dim=1), dim=1)

    # The rotation matrix is supposed to be applied to the left,
    # and we store (x, y, z) per column since we stack along the column dimension.
    basis = torch.stack([basis_x, basis_y, basis_z], dim=2)

    # Rotated the basis from the aligned space to the original space.
    basis_rotated = torch.bmm(inverse_rotation.to(device), basis.to(device))

    return basis_rotated


def calc_intersection_with_screen(c: torch.Tensor, d_2d_left: torch.Tensor, d_2d_right: torch.Tensor, o_2d_left: torch.Tensor, o_2d_right: torch.Tensor, data_batch: dict) -> Tuple[List[torch.Tensor], Tuple]:
    """
    Calculate the intersection of the gaze ray with the display.

    :param c: rough distance correction value (Tensor[batch_size, 1])
    :param d_2d_left: 2D gaze origin in the normalized left image (Tensor[batch_size, 2])
    :param d_2d_right: 2D gaze origin in the normalized right image (Tensor[batch_size, 2])
    :param o_2d_left: 2D gaze direction in the normalized left image (Tensor[batch_size, 2])
    :param o_2d_right: 2D gaze direction in the normalized right image (Tensor[batch_size, 2])
    :param data_batch: current batch
    :return: gaze target on the screen in the camera coordinate system (Tensor[batch_size, 3, 1]) for each eye and further params for debugging
    """
    device = c.device
    batch_size = c.shape[0]

    distance_rough = data_batch['distance_rough'].float().to(device)
    distance = distance_rough.unsqueeze(1) * c  # correct rough distance (63 mm interocular distance) by the network

    screen_translation_vector = data_batch['screen_translation_vector'].float()
    screen_rotation_matrix = data_batch['screen_rotation_matrix'].float()

    # define plane (screen)
    plane_normal = screen_rotation_matrix[:, :, 2]
    plane_origin = screen_translation_vector
    plane_d = plane_normal.unsqueeze(1) @ plane_origin.unsqueeze(2)

    left_eye_camera_aligned = [CameraArcane(focal_length, principal_point) for focal_length, principal_point in zip(data_batch['left_eye_camera_aligned_focal_length'], data_batch['left_eye_camera_aligned_principal_point'])]
    right_eye_camera_aligned = [CameraArcane(focal_length, principal_point) for focal_length, principal_point in zip(data_batch['right_eye_camera_aligned_focal_length'], data_batch['right_eye_camera_aligned_principal_point'])]

    predicted_gaze_target_3d = []
    o_s = []
    d_s = []
    for camera_aligned, inverse_rotation, o_2d, d_2d in [
        (left_eye_camera_aligned, data_batch['left_eye_inverse_rotation'].float(), o_2d_left, d_2d_left),
        (right_eye_camera_aligned, data_batch['right_eye_inverse_rotation'].float(), o_2d_right, d_2d_right),
    ]:
        # 3D origin of the gaze ray, o
        basis_rotated = compute_basis(camera_aligned, inverse_rotation, o_2d)

        intrinsic_matrix = torch.stack([camera_aligned.intrinsic_matrix() for camera_aligned in camera_aligned]).float()
        o = torch.linalg.inv(intrinsic_matrix) @ torch.cat((o_2d, torch.ones(batch_size).unsqueeze(-1).to(device)), dim=1).unsqueeze(-1)  # torch.linalg.inv(intrinsic_matrix) @ (o_2D[0], o_2D[1], 1)
        o = o / torch.linalg.norm(o)
        o = basis_rotated @ o  # backproject the 2D gaze origin o_2D through the normalized camera
        o = o * (distance / o[:, 2]).unsqueeze(2)  # back-project to distance c

        # 3D direction of the gaze ray, d
        d = torch.transpose(basis_rotated[:, :2], 1, 2) @ d_2d.unsqueeze(2) + basis_rotated[:, -1].unsqueeze(2)  # d = [xy] * d_2D + z.
        d = d.squeeze(2)
        d = d / torch.linalg.norm(d, dim=1).reshape(-1, 1)

        # intersection of gaze ray with screen
        a11 = d[:, 1]
        a12 = -d[:, 0]
        b1 = d[:, 1].unsqueeze(1) * o[:, 0] - d[:, 0].unsqueeze(1) * o[:, 1]

        a22 = d[:, 2]
        a23 = -d[:, 1]
        b2 = d[:, 2].unsqueeze(1) * o[:, 1] - d[:, 1].unsqueeze(1) * o[:, 2]

        line_w = torch.cat([
            torch.cat([torch.cat([a11.unsqueeze(1), a12.unsqueeze(1)], dim=1), torch.zeros(batch_size).unsqueeze(1).to(device)], dim=1).unsqueeze(1),
            torch.cat([torch.zeros(batch_size).unsqueeze(1).to(device), torch.cat([a22.unsqueeze(1), a23.unsqueeze(1)], dim=1)], dim=1).unsqueeze(1),
        ], dim=1)

        line_b = torch.cat([b1.unsqueeze(1), b2.unsqueeze(1)], dim=1)

        matrix = torch.cat([line_w.to(device), plane_normal.unsqueeze(1).to(device)], dim=1)
        bias = torch.cat([line_b.to(device), plane_d.to(device)], dim=1)

        try:
            solution = torch.linalg.solve(matrix, bias)
        except RuntimeError as e:
            print(e)
            solution = torch.zeros([batch_size, 3, 1]).to(device)  # return zeros instead of crashing for singular matrix
        predicted_gaze_target_3d.append(solution)

        o_s.append(o)
        d_s.append(d)

    return predicted_gaze_target_3d, (d_s, o_s, plane_d, plane_normal, screen_translation_vector)


def normalize_ray(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.norm(vector, axis=1).reshape(-1, 1)


def calc_angular_between_rays(outputs_norm: torch.Tensor, labels_norm: torch.Tensor) -> torch.Tensor:
    """
    Calculates the angle (in degrees) between two normalized vectors.

    :param outputs_norm: first normalized vectors
    :param labels_norm: second normalized vectors
    :return: angle between the two normalized vectors in degrees
    """

    angles = F.cosine_similarity(outputs_norm, labels_norm, dim=1)
    angles = torch.clip(angles, -1.0, 1.0)  # fix NaN values for 1.0 < angles && angles < -1.0
    angles_rad = torch.arccos(angles)
    return torch.rad2deg(angles_rad)


def get_data_from_batch(train_batch):
    person_idx = train_batch['person_idx'].long()
    left_eye_image = train_batch['left_eye_image'].float()
    right_eye_image = train_batch['right_eye_image'].float()
    face_image = train_batch['face_image'].float()
    eye_3d_left = train_batch['eye_3d_left'].float()
    eye_3d_right = train_batch['eye_3d_right'].float()
    return face_image, left_eye_image, person_idx, right_eye_image, eye_3d_left, eye_3d_right


def visualize_gaze_ray(d_s: List[torch.Tensor], gaze_target_3d: torch.Tensor, o_s: List[torch.Tensor], plane_d: torch.Tensor, plane_normal, predicted_gaze_target_3d, screen_translation_vector, idx: int = 0) -> None:
    """
    Visualize the results by drawing the screen, the gaze origins (eyes) and the gaze directions in 3D.

    :param d_s: 3D gaze direction for left and right eye (Tensor[batch_size, 3] each)
    :param gaze_target_3d: ground truth 3D gaze target on the screen (Tensor[batch_size, 3])
    :param o_s: 3D gaze origin (eye center) for left and right eye (Tensor[batch_size, 3] each)
    :param plane_d: (Tensor[batch_size, 1, 1])
    :param plane_normal: (Tensor[batch_size, 3])
    :param predicted_gaze_target_3d: predicted 3D gaze targets on the screen for left and right eye (Tensor[batch_size, 3] each)
    :param screen_translation_vector: translation vector of the screen/plane (Tensor[batch_size, 3])
    :param idx: which of the `batch_size` results should be drawn
    :return:
    """

    from matplotlib import pyplot as plt

    def plt_legend():
        # https://stackoverflow.com/a/13589144/9309705
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    screen_translation_vector = screen_translation_vector.cpu().detach().numpy()[idx]
    plane_normal = plane_normal[idx].cpu().detach().numpy()
    plane_d = plane_d[idx].cpu().detach().numpy().item()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')
    length = 600
    ax.set_xlim(-length / 2, length / 2)
    ax.set_ylim(-100, length - 100)
    ax.set_zlim(-10, length - 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot(0, 0, 0, linestyle="", marker="o", color='b', label='webcam')  # webcam

    temp_gaze_target_3d = gaze_target_3d[0].cpu().detach().numpy().reshape(-1)
    print('temp_gaze_target_3d', temp_gaze_target_3d)

    ax.plot(temp_gaze_target_3d[0], temp_gaze_target_3d[1], temp_gaze_target_3d[2], linestyle="", marker="X", color='g', label='temp_gaze_target_3d')

    for temp_predicted_gaze_target_3d, o_temp, d_temp in zip(predicted_gaze_target_3d, o_s, d_s):
        o_temp = o_temp[0].cpu().detach().numpy().reshape(-1)
        d_temp = d_temp[0].cpu().detach().numpy().reshape(-1)
        ax.plot(o_temp[0], o_temp[1], o_temp[2], linestyle="", marker="X", color='r', label='3d eye through camera_aligned')

        temp_predicted_gaze_target_3d = temp_predicted_gaze_target_3d[0].cpu().detach().numpy().reshape(-1)
        ax.plot(temp_predicted_gaze_target_3d[0], temp_predicted_gaze_target_3d[1], temp_predicted_gaze_target_3d[2], linestyle="", marker="X", color='y', label='predicted_gaze_target_3d')
        ax.plot([o_temp[0], temp_predicted_gaze_target_3d[0]], [o_temp[1], temp_predicted_gaze_target_3d[1]], [o_temp[2], temp_predicted_gaze_target_3d[2]], color='r', label='gaze')

        o_d = o_temp - 1000 * d_temp  # has to be negative ?
        ax.plot([o_temp[0], o_d[0]], [o_temp[1], o_d[1]], [o_temp[2], o_d[2]], color='y', label='???')

    screen_width_mm = 500  # my local values
    screen_height_mm = 281  # my local values
    screen_offset_y = screen_translation_vector[1]

    xx, yy = np.meshgrid(range(int(-screen_width_mm / 2), int(screen_width_mm / 2)), range(int(screen_offset_y), int(screen_height_mm + screen_offset_y)))
    z = (plane_normal[0] * xx - plane_normal[1] * yy + plane_d) * 1. / plane_normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color='green')

    plt_legend()
    plt.show()
