import glob
import pathlib
from collections import defaultdict
from typing import List

import cv2
import h5py
import numpy as np
import pandas as pd
import scipy.io
import skimage.io
import skimage.transform
import torch

from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.dataset_utils import compute_rough_distance, get_cropped_images, get_face_landmarks, get_annotations
from utils import fix_qt


def get_each_of_one_grid_idx(k, gaze_locations_filtered, screen_sizes_filtered):
    np.random.seed(42)
    grids = int(np.sqrt(k))  # get grid size from k

    grid_width = screen_sizes_filtered[0][0] / grids
    height_width = screen_sizes_filtered[0][1] / grids

    gaze_locations_filtered = np.asarray(gaze_locations_filtered)

    valid_random_idx = []

    for width_range in range(grids):
        filter_width = (grid_width * width_range < gaze_locations_filtered[:, :1]) & (gaze_locations_filtered[:, :1] < grid_width * (width_range + 1))
        for height_range in range(grids):
            filter_height = (height_width * height_range < gaze_locations_filtered[:, 1:]) & (gaze_locations_filtered[:, 1:] < height_width * (height_range + 1))
            complete_filter = filter_width & filter_height
            complete_filter = complete_filter.reshape(-1)
            if sum(complete_filter) > 0:
                true_idxs = np.argwhere(complete_filter == True)
                random_idx = (np.random.rand(1) * len(true_idxs)).astype(int).item()
                valid_random_idx.append(true_idxs[random_idx].item())
            else:
                pass

    if len(valid_random_idx) != k:
        print('missing', k - len(valid_random_idx))
        missing_k = k - len(valid_random_idx)
        missing_idxs = (np.random.rand(missing_k) * len(gaze_locations_filtered)).astype(int)
        for missing_idx in missing_idxs:
            valid_random_idx.append(missing_idx.item())

    return valid_random_idx


class MPIIFaceGazePreprocessed(Dataset):
    """
    MPIIFaceGaze dataset with offline preprocessing (= already preprocessed)
    """

    def __init__(self, data_path: str, file_name: str, keep_person_idxs: List[int], transform=None, validation_set=False, subset=None):
        if keep_person_idxs is not None:
            assert len(keep_person_idxs) > 0
            assert max(keep_person_idxs) <= 14  # last person id = 14
            assert min(keep_person_idxs) >= 0  # first person id = 0

        self.data_path = data_path
        self.transform = transform

        self.hdf5_file_name = f'{data_path}/{file_name}'
        self.h5_file = None

        with h5py.File(self.hdf5_file_name, 'r') as f:
            self.file_names = [file_name.decode('utf-8') for file_name in f['file_name']]

        self.idx2ValidIdx = self.filter_persons_by_idx(keep_person_idxs, validation_set)

        if subset is not None:
            # subset images of the first ~2,500 eye images for training
            np.random.seed(42)
            self.idx2ValidIdx = np.random.choice(self.idx2ValidIdx[:-500], subset, replace=False)

            # position calibration points in grid
            # hdfs = h5py.File('./screen.h5', 'r', swmr=True)
            # gaze_locations = np.asarray(hdfs['gaze_location'][:])
            # screen_sizes = np.asarray(hdfs['screen_size'][:])
            # gaze_locations_filtered = [gaze_locations[idx] for idx in self.idx2ValidIdx[:-500]]
            # screen_sizes_filtered = [screen_sizes[idx] for idx in self.idx2ValidIdx[:-500]]
            # self.idx2ValidIdx = get_each_of_one_grid_idx(subset, gaze_locations_filtered, screen_sizes_filtered)

    def filter_persons_by_idx(self, keep_person_idxs: List[int], validation_set: bool) -> List[int]:
        idx_per_person = [[] for _ in range(15)]
        if keep_person_idxs is not None:
            keep_person_idxs = [f'p{person_idx:02d}/' for person_idx in set(keep_person_idxs)]
            for idx, file_name in enumerate(self.file_names):
                if any(keep_person_idx in file_name for keep_person_idx in keep_person_idxs):
                    person_idx = int(file_name.split('/')[-3][1:])
                    idx_per_person[person_idx].append(idx)
        else:
            for idx, file_name in enumerate(self.file_names):
                person_idx = int(file_name.split('/')[-3][1:])
                idx_per_person[person_idx].append(idx)

        idx_2_valid_idx = []
        for indexes in idx_per_person:
            if validation_set:  # 500 eye images for testing
                idx_2_valid_idx.extend(indexes[-500:])
            else:
                idx_2_valid_idx.extend(indexes)

        return idx_2_valid_idx

    def __len__(self) -> int:
        return len(self.idx2ValidIdx)

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

    def open_hdf5(self):
        self.h5_file = h5py.File(self.hdf5_file_name, 'r')

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.h5_file is None:
            self.open_hdf5()

        idx = self.idx2ValidIdx[idx]

        file_name = self.h5_file['file_name'][idx].decode('utf-8')
        person_idx = int(file_name.split('/')[-3][1:])

        left_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-left_eye.png")
        right_eye_image = skimage.io.imread(f"{self.data_path}/{file_name}-right_eye.png")
        face_image = skimage.io.imread(f"{self.data_path}/{file_name}-face_image.png")

        distance_rough = self.h5_file['distance_rough'][idx]
        screen_translation_vector = self.h5_file['screen_translation_vector_list'][person_idx]
        screen_rotation_matrix = self.h5_file['screen_rotation_matrix_list'][person_idx]
        target_gaze = self.h5_file['target_gaze'][idx]

        left_eye_camera_aligned_focal_length = self.h5_file['left_eye_camera_aligned_focal_length'][idx]
        left_eye_camera_aligned_principal_point = self.h5_file['left_eye_camera_aligned_principal_point'][idx]
        left_eye_inverse_rotation = self.h5_file['left_eye_inverse_rotation'][idx]
        right_eye_camera_aligned_focal_length = self.h5_file['right_eye_camera_aligned_focal_length'][idx]
        right_eye_camera_aligned_principal_point = self.h5_file['right_eye_camera_aligned_principal_point'][idx]
        right_eye_inverse_rotation = self.h5_file['right_eye_inverse_rotation'][idx]
        face_image_camera_aligned_focal_length = self.h5_file['face_image_camera_aligned_focal_length'][idx]
        face_image_camera_aligned_principal_point = self.h5_file['face_image_camera_aligned_principal_point'][idx]
        face_image_inverse_rotation = self.h5_file['face_image_inverse_rotation'][idx]

        eye_3d_left = self.h5_file['eye_3d_left'][idx]
        eye_3d_right = self.h5_file['eye_3d_right'][idx]

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
            'screen_translation_vector': screen_translation_vector,
            'screen_rotation_matrix': screen_rotation_matrix,
            'target_gaze': target_gaze,  # 3D label gaze position on screen

            'left_eye_camera_aligned_focal_length': left_eye_camera_aligned_focal_length,
            'left_eye_camera_aligned_principal_point': left_eye_camera_aligned_principal_point,
            'left_eye_inverse_rotation': left_eye_inverse_rotation,
            'right_eye_camera_aligned_focal_length': right_eye_camera_aligned_focal_length,
            'right_eye_camera_aligned_principal_point': right_eye_camera_aligned_principal_point,
            'right_eye_inverse_rotation': right_eye_inverse_rotation,
            'face_image_camera_aligned_focal_length': face_image_camera_aligned_focal_length,
            'face_image_camera_aligned_principal_point': face_image_camera_aligned_principal_point,
            'face_image_inverse_rotation': face_image_inverse_rotation,

            'eye_3d_left': eye_3d_left,
            'eye_3d_right': eye_3d_right,
        }


def preprocess_mpii_face_gaze(data_path: str, output_path: str):
    intrinsic_matrix_list = []
    screen_translation_vector_list = []
    screen_rotation_matrix_list = []

    data = defaultdict(list)

    org_face_model = scipy.io.loadmat(f'{data_path}/6 points-based face model.mat')['model']

    for person_idx, person_path in enumerate(tqdm(sorted(glob.glob(f'{data_path}/p*')), desc='person')):
        intrinsic_matrix = scipy.io.loadmat(f'{person_path}/Calibration/Camera.mat')['cameraMatrix']
        intrinsic_matrix_list.append(intrinsic_matrix)

        monitor_pose = scipy.io.loadmat(f'{person_path}/Calibration/monitorPose.mat')
        screen_translation_vector = monitor_pose["tvecs"].reshape(-1)
        screen_rotation_matrix, _ = cv2.Rodrigues(monitor_pose["rvects"])
        screen_translation_vector_list.append(screen_translation_vector)
        screen_rotation_matrix_list.append(screen_rotation_matrix)

        df_annotations = pd.read_csv(f'{person_path}/p{person_idx:02d}.txt', sep=' ', header=None, index_col=0)

        for day_path in tqdm(sorted(glob.glob(f'{person_path}/day01/')), desc='day'):
            for image_path in sorted(glob.glob(f'{day_path}/*.jpg')):
                image = skimage.io.imread(image_path)

                person_idx = int(image_path.split('/')[-3][1:])
                intrinsic_matrix = intrinsic_matrix_list[person_idx]

                face_landmarks_2d, gaze_target_3d, head_rotation, head_translation = get_annotations(df_annotations.loc['/'.join(image_path.split('/')[-2:])])
                face_landmarks = get_face_landmarks(head_rotation, head_translation, org_face_model)

                eye_left_2d = face_landmarks_2d[:2].mean(axis=0).round().astype(int)
                eye_right_2d = face_landmarks_2d[2:4].mean(axis=0).round().astype(int)

                output = get_cropped_images(intrinsic_matrix, eye_left_2d, eye_right_2d, image)
                left_eye_image, left_eye_camera_aligned, left_eye_inverse_rotation = output[0]
                right_eye_image, right_eye_camera_aligned, right_eye_inverse_rotation = output[1]
                face_image, face_image_camera_aligned, face_image_inverse_rotation = output[2]

                distance_rough = compute_rough_distance(intrinsic_matrix, eye_right_2d, eye_left_2d)

                pathlib.Path(f"{output_path}/{'/'.join(image_path.split('/')[-3:-1])}").mkdir(parents=True, exist_ok=True)
                file_name = '/'.join(image_path.split('/')[-3:])[:-4]
                base_file_name = f"{output_path}/{file_name}"
                skimage.io.imsave(f"{base_file_name}-left_eye.png", (left_eye_image * 255).astype(np.uint8), check_contrast=False)
                skimage.io.imsave(f"{base_file_name}-right_eye.png", (right_eye_image * 255).astype(np.uint8), check_contrast=False)
                skimage.io.imsave(f"{base_file_name}-face_image.png", (face_image * 255).astype(np.uint8), check_contrast=False)

                data['file_name'].append(file_name)
                data['distance_rough'].append(distance_rough)
                data['target_gaze'].append(gaze_target_3d)

                data['left_eye_camera_aligned_focal_length'].append(left_eye_camera_aligned.focal_length)
                data['left_eye_camera_aligned_principal_point'].append(left_eye_camera_aligned.principal_point)
                data['left_eye_inverse_rotation'].append(left_eye_inverse_rotation)

                data['right_eye_camera_aligned_focal_length'].append(right_eye_camera_aligned.focal_length)
                data['right_eye_camera_aligned_principal_point'].append(right_eye_camera_aligned.principal_point)
                data['right_eye_inverse_rotation'].append(right_eye_inverse_rotation)

                data['face_image_camera_aligned_focal_length'].append(face_image_camera_aligned.focal_length)
                data['face_image_camera_aligned_principal_point'].append(face_image_camera_aligned.principal_point)
                data['face_image_inverse_rotation'].append(face_image_inverse_rotation)

                data['eye_left'].append(eye_left_2d)
                data['eye_right'].append(eye_right_2d)

                data['eye_3d_left'].append(face_landmarks[:2].mean(axis=0))
                data['eye_3d_right'].append(face_landmarks[2:4].mean(axis=0))

        data['intrinsic_matrix_list'] = intrinsic_matrix_list
        data['screen_translation_vector_list'] = screen_translation_vector_list
        data['screen_rotation_matrix_list'] = screen_rotation_matrix_list

        with h5py.File(f'{output_path}/data.h5', 'w') as file:
            for key, value in data.items():
                if key == 'file_name':  # only str
                    file.create_dataset(key, data=value, compression='gzip', chunks=True)
                else:
                    value = np.asarray(value)
                    file.create_dataset(key, data=value, shape=value.shape, compression='gzip', chunks=True)


if __name__ == '__main__':
    fix_qt()

    input_path = './data/MPIIFaceGaze'
    output_path = './data/SPAZE_MPIIFaceGaze'

    preprocess_mpii_face_gaze(input_path, output_path)

    from matplotlib import pyplot as plt

    for element in MPIIFaceGazePreprocessed('./data/SPAZE_MPIIFaceGaze', 'data.h5', keep_person_idxs=[0], subset=9):
        print(element)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.imshow(element['left_eye_image'])
        ax2.imshow(element['right_eye_image'])
        ax3.imshow(element['face_image'])
        plt.show()
