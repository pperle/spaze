from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataset.mpii_face_gaze_preprocessed import MPIIFaceGazePreprocessed


class SpazeDataModule(LightningDataModule):
    def __init__(self, data_dir: str, num_workers: int = 16, batch_size: int = 64, fine_tune_while_training: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.fine_tune_while_training = fine_tune_while_training

        # "randomly offset the detections in a disk with a radius equal to 4% of the interocular distance"
        self.transform = {
            'train': A.Compose([
                A.ShiftScaleRotate(p=1.0, shift_limit=0.1, scale_limit=0.05, rotate_limit=2),  # "eye detections are jittered for data augmentation in training"
                A.Normalize(),
                ToTensorV2()
            ]),
            'valid': A.Compose([
                A.Normalize(),
                ToTensorV2()
            ])
        }

    def setup(self, stage: Optional[str] = None):
        # self.spaze_train = MPIIFaceGaze(self.data_dir, transform=self.transform['train'])
        # self.spaze_val = MPIIFaceGaze(self.data_dir, transform=self.transform['valid'])

        # "For within-dataset evaluation, we perform leave-one-out training on MPIIGaze."
        # Training on person 1-14 and (later ?) evaluation on person 0
        self.spaze_train = MPIIFaceGazePreprocessed(self.data_dir, 'data.h5', keep_person_idxs=list(range(1, 15)), transform=self.transform['train'])

        if self.fine_tune_while_training:
            self.spaze_fine_tune = MPIIFaceGazePreprocessed(self.data_dir, 'data.h5', keep_person_idxs=[0], transform=self.transform['valid'], subset=9)
            self.spaze_val = MPIIFaceGazePreprocessed(self.data_dir, 'data.h5', keep_person_idxs=[0], transform=self.transform['valid'], validation_set=True)
        else:
            self.spaze_val = MPIIFaceGazePreprocessed(self.data_dir, 'data.h5', keep_person_idxs=[0], transform=self.transform['valid'], validation_set=True)

    def train_dataloader(self):
        print('train_dataloader', len(self.spaze_train))
        return DataLoader(self.spaze_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def fine_tune_dataloader(self):
        print('fine_tune_dataloader', len(self.spaze_fine_tune))
        return DataLoader(self.spaze_fine_tune, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        print('val_dataloader', len(self.spaze_val))
        return DataLoader(self.spaze_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
