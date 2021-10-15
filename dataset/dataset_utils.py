from typing import Tuple

import cv2
import numpy as np
import skimage.transform
import torch


def back_project(intrinsic_matrix: np.ndarray, px: np.ndarray) -> np.ndarray:
    return np.linalg.inv(intrinsic_matrix) @ np.append(px, 1)


class CameraArcane:
    def __init__(self, focal_length, principal_point):
        self.focal_length = focal_length
        self.principal_point = principal_point

    def intrinsic_matrix(self):
        if torch.is_tensor(self.focal_length):
            return torch.Tensor([
                [self.focal_length[0], 0, self.principal_point[0]],
                [0, self.focal_length[1], self.principal_point[1]],
                [0, 0, 1.0],
            ]).to(self.focal_length.device)
        else:
            return np.array([
                [self.focal_length[0], 0, self.principal_point[0]],
                [0, self.focal_length[1], self.principal_point[1]],
                [0, 0, 1.0],
            ])


def normalized(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def center_from_size(size):
    return np.asarray([size[0] / 2, size[1] / 2])


def compute_aligned_image_camera_and_rotation(intrinsic_matrix: np.ndarray, eyes_positions: np.ndarray, eyes_difference, image, center, image_output_size: Tuple[int, int], interocular_distance: int, mirror=False):
    """
    from Erik Lindén

    image transformation matrix = C_n * R C^−1
    C_r = projection matrix of the real camera
    C_n = projection matrix of the normalized camera

    :param intrinsic_matrix:
    :param eyes_positions:
    :param eyes_difference:
    :param image:
    :param center: either an eye detection point or the midpoint between the eyes
    :param image_output_size: output image size (224×112 pixels for the eye images and 224×56 pixels for the face image)
    :param interocular_distance: in pixel for normalized camera (C_n)
    :param mirror:
    :return:
    """

    # Find a rotation matrix such that `center` lies on the z-axis and
    # such that the vector between the eyes lies in the x-z-plane.
    basis_z = normalized(back_project(intrinsic_matrix, center))
    basis_x_z = normalized(back_project(intrinsic_matrix, center + eyes_difference))
    basis_x = normalized(basis_x_z - np.dot(basis_x_z, basis_z) * basis_z)
    basis_y = normalized(np.cross(basis_z, basis_x))
    rotation = np.array([basis_x, basis_y, basis_z])

    if mirror:
        rotation[0, :] *= -1

    # Undo the real camera and rotate.
    intrinsic_inverse = np.linalg.inv(intrinsic_matrix)
    transform_rotate = skimage.transform.ProjectiveTransform(np.dot(rotation, intrinsic_inverse))

    # Create our canonical camera.
    eyes_transformed = transform_rotate(eyes_positions)
    focal_length = interocular_distance / np.linalg.norm(eyes_transformed[1] - eyes_transformed[0])
    focal_length = np.tile(focal_length, (2,))
    crop_center = center_from_size(image_output_size)

    camera_aligned = CameraArcane(  # arcane.dataset.Camera(
        focal_length=focal_length,
        principal_point=crop_center,
    )

    transform_canonical = skimage.transform.ProjectiveTransform(camera_aligned.intrinsic_matrix())
    transform = transform_rotate + transform_canonical

    # Warp image into the aligned camera.
    image_aligned = skimage.transform.warp(
        image=image,
        inverse_map=transform.inverse,
        output_shape=image_output_size[::-1],
        mode="edge",
    )

    # Use the center of the crop as the reference point.
    camera_aligned.principal_point -= crop_center

    # Apply this to points in the normalized camera to get back to the original space.
    inverse_rotation = np.linalg.inv(rotation)

    return image_aligned, camera_aligned, inverse_rotation


def compute_rough_distance(intrinsic_matrix, eye_right, eye_left):
    """
    A practical way to compute a rough distance is to back project the eye detections through the camera.
    That gives you a unit vector for each eye, n_l and n_r. Then you find a distance t such that ||t n_l - t n_r|| = interocular_distance.

    Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision. Cambridge University Press. https://doi.org/10.1017/cbo9780511811685
    6.2.2  Action of a projective camera on points (Back-projection of points to rays.)

    :param intrinsic_matrix:
    :param eye_right:
    :param eye_left:
    :return: distance to eyes
    """
    avg_human_interocular_distance_mm = 63

    direction_right = back_project(intrinsic_matrix, eye_right)
    direction_right = direction_right / np.linalg.norm(direction_right)

    direction_left = back_project(intrinsic_matrix, eye_left)
    direction_left = direction_left / np.linalg.norm(direction_left)

    # TODO use binary search e.g. in rang eof 0-100 cm
    scaling_factor = 0
    while np.linalg.norm((direction_right - direction_left) * scaling_factor) < avg_human_interocular_distance_mm:
        scaling_factor += 0.1

    return np.mean([(direction_right * scaling_factor)[-1], (direction_left * scaling_factor)[-1]])


def get_cropped_images(intrinsic_matrix: np.ndarray, eye_left_2d, eye_right_2d, image):
    """
    Image normalization [3.1]

    :param intrinsic_matrix:
    :param eye_left_2d: left eye in image
    :param eye_right_2d: right eye in image
    :param image: image of a person’s face
    :return: "three images: two high-resolution eye images centered at the eye detection points and one low-resolution face image centered at the midpoint between the eyes"
    """
    eyes_positions = np.asarray([eye_left_2d, eye_right_2d])
    eyes_difference = eye_right_2d - eye_left_2d

    return (
        compute_aligned_image_camera_and_rotation(intrinsic_matrix, eyes_positions, eyes_difference, image, eye_left_2d, (224, 112), 320, mirror=True),
        compute_aligned_image_camera_and_rotation(intrinsic_matrix, eyes_positions, eyes_difference, image, eye_right_2d, (224, 112), 320),
        compute_aligned_image_camera_and_rotation(intrinsic_matrix, eyes_positions, eyes_difference, image, np.mean(np.asarray([eye_left_2d, eye_right_2d]), axis=0), (224, 56), 84)
    )


def get_face_landmarks(head_rotation, head_translation, org_face_model):
    head_rotation_matrix, _ = cv2.Rodrigues(head_rotation)
    face_landmarks = np.dot(head_rotation_matrix, org_face_model) + head_translation.reshape((3, 1))  # 3D positions of facial landmarks
    face_landmarks = face_landmarks.T
    return face_landmarks


def get_annotations(annotation):
    face_landmarks_2d = annotation[2:14].to_numpy().reshape(-1, 2).astype(int)
    head_rotation = annotation[14:17].to_numpy().reshape(-1).astype(float)
    head_translation = annotation[17:20].to_numpy().reshape(-1).astype(float)
    gaze_target_3d = annotation[23:26].to_numpy().reshape(-1).astype(float)
    return face_landmarks_2d, gaze_target_3d, head_rotation, head_translation
