import glob

import cv2
import pandas as pd
import scipy.io
import skimage.io
import skimage.transform
import torch
from torch.utils.data import Dataset

from dataset.dataset_utils import get_annotations, get_face_landmarks, get_cropped_images, compute_rough_distance


class MPIIFaceGaze(Dataset):
    """
    MPIIFaceGaze dataset with online preprocessing (very slow)

    """

    def __init__(self, data_path: str, transform=None):
        self.intrinsic_matrix_list = []
        self.screen_translation_vector_list = []
        self.screen_rotation_matrix_list = []
        self.df_annotations = pd.DataFrame()
        self.image_path_list = []

        self.org_face_model = scipy.io.loadmat(f'{data_path}/6 points-based face model.mat')['model']

        for person_idx, person_path in enumerate(sorted(glob.glob(f'{data_path}/p*'))):
            camera = scipy.io.loadmat(f'{person_path}/Calibration/Camera.mat')
            self.intrinsic_matrix_list.append(camera['cameraMatrix'])

            monitor_pose = scipy.io.loadmat(f'{person_path}/Calibration/monitorPose.mat')
            screen_translation_vector = monitor_pose["tvecs"].reshape(-1)
            screen_rotation_matrix, _ = cv2.Rodrigues(monitor_pose["rvects"])
            self.screen_translation_vector_list.append(screen_translation_vector)
            self.screen_rotation_matrix_list.append(screen_rotation_matrix)

            self.df_annotations = self.df_annotations.append(pd.read_csv(f'{person_path}/p{person_idx:02d}.txt', sep=' ', header=None, index_col=0))

            for day_path in sorted(glob.glob(f'{person_path}/day*')):
                for image_path in sorted(glob.glob(f'{day_path}/*.jpg')):
                    self.image_path_list.append(image_path)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.image_path_list[idx]
        image = skimage.io.imread(image_path)

        person_idx = int(self.image_path_list[idx].split('/')[-3][1:])
        intrinsic_matrix = self.intrinsic_matrix_list[person_idx]

        face_landmarks_2d, gaze_target_3d, head_rotation, head_translation = get_annotations(self.df_annotations.iloc[idx])
        face_landmarks = get_face_landmarks(head_rotation, head_translation, self.org_face_model)

        eye_left_2d = face_landmarks_2d[:2].mean(axis=0).round().astype(int)
        eye_right_2d = face_landmarks_2d[2:4].mean(axis=0).round().astype(int)

        output = get_cropped_images(intrinsic_matrix, eye_left_2d, eye_right_2d, image)
        face_image, face_image_camera_aligned, face_image_inverse_rotation = output[0]
        left_eye_image, left_eye_camera_aligned, left_eye_inverse_rotation = output[1]
        right_eye_image, right_eye_camera_aligned, right_eye_inverse_rotation = output[2]

        distance_rough = compute_rough_distance(intrinsic_matrix, eye_right_2d, eye_left_2d)

        if self.transform:
            left_eye_image = self.transform(image=left_eye_image)["image"]
            right_eye_image = self.transform(image=right_eye_image)["image"]
            face_image = self.transform(image=face_image)["image"]

        return {
            'person_idx': person_idx,

            'left_eye_image': left_eye_image,
            'right_eye_image': right_eye_image,
            'face_image': face_image,

            'distance_rough': distance_rough,
            'eye_3d_left': face_landmarks[:2].mean(axis=0),
            'eye_3d_right': face_landmarks[2:4].mean(axis=0),
            'target_gaze': gaze_target_3d,  # 3D label gaze position on screen

            'intrinsic_matrix': intrinsic_matrix,
            'screen_translation_vector': self.screen_translation_vector_list[person_idx],
            'screen_rotation_matrix': self.screen_rotation_matrix_list[person_idx],

            'left_eye_camera_aligned': left_eye_camera_aligned,
            'left_eye_inverse_rotation': left_eye_inverse_rotation,
            'right_eye_camera_aligned': right_eye_camera_aligned,
            'right_eye_inverse_rotation': right_eye_inverse_rotation,
            'face_image_camera_aligned': face_image_camera_aligned,
            'face_image_inverse_rotation': face_image_inverse_rotation,
        }


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    dataset = MPIIFaceGaze('./data/MPIIFaceGaze')
    print('len(dataset)', len(dataset))
    for element in dataset:
        print(element)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.imshow(element['left_eye_image'])
        ax2.imshow(element['right_eye_image'])
        ax3.imshow(element['face_image'])
        plt.show()
