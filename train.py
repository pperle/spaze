import os
from argparse import ArgumentParser

import numpy as np
import torch
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from scipy import optimize

from dataset.spaze_data_module import SpazeDataModule
from model import Spaze
from utils import calc_intersection_with_screen, normalize_ray, calc_angular_between_rays, fix_qt, visualize_gaze_ray, get_data_from_batch

DEBUG = False
num_workers = 0 if DEBUG else os.cpu_count()

FINE_TUNE_WHILE_TRAINING = True


class Model(Spaze):
    def __init__(self, learning_rate: float, weight_decay: float, batch_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        self.save_hyperparameters()  # log hyperparameters

    def configure_optimizers(self):
        # "We train using Adam with a learning rate of 10^−3 and a weight decay of 10^−5 ." [3.4]
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)

    def training_step(self, train_batch: dict, batch_idx: int, is_valid: bool = False) -> STEP_OUTPUT:
        face_image, left_eye_image, person_idx, right_eye_image, eye_3d_left, eye_3d_right = get_data_from_batch(train_batch)
        gaze_target_3d = train_batch['target_gaze'].float()

        # same image just further away, miss distance is larger
        outputs = self(person_idx, face_image, right_eye_image, left_eye_image)
        c, o_2d_right, d_2d_right, o_2d_left, d_2d_left = outputs

        predicted_gaze_target_3d, debug_value = calc_intersection_with_screen(c, d_2d_left, d_2d_right, o_2d_left, o_2d_right, train_batch)

        if DEBUG:
            d_s, o_s, plane_d, plane_normal, screen_translation_vector = debug_value
            visualize_gaze_ray(d_s, gaze_target_3d, o_s, plane_d, plane_normal, predicted_gaze_target_3d, screen_translation_vector)

        # angle error as loss???
        # L1 loss??? L2 might be better.
        # loss = F.mse_loss(  # "miss distance between the gaze ray and the 3D gaze target" [3.4]
        #     torch.cat(predicted_gaze_target_3d, dim=2),
        #     torch.cat([gaze_target_3d.unsqueeze(2), gaze_target_3d.unsqueeze(2)], dim=2)
        # )

        # TODO hinge loss
        # To prevent unphysical solutions when training on current datasets, we introduced two regularizing terms: 1) a hinge
        # loss on the 2D gaze origin o 2D , penalizing if it moves outside the eye image and 2) a hinge loss on the distance correction
        # c, penalizing changes in distance by more than 40 % in either direction.

        diff_mm = (torch.linalg.norm(gaze_target_3d - predicted_gaze_target_3d[0].squeeze(2), dim=1).mean() + torch.linalg.norm(gaze_target_3d - predicted_gaze_target_3d[1].squeeze(2), dim=1).mean()) / 2
        loss = diff_mm

        real_ray_left = normalize_ray(gaze_target_3d - eye_3d_left)
        real_ray_right = normalize_ray(gaze_target_3d - eye_3d_right)
        predicted_ray_left = normalize_ray(predicted_gaze_target_3d[0].squeeze(2) - eye_3d_left)
        predicted_ray_right = normalize_ray(predicted_gaze_target_3d[1].squeeze(2) - eye_3d_right)

        angles_left_deg = calc_angular_between_rays(predicted_ray_left, real_ray_left).mean()
        angles_right_deg = calc_angular_between_rays(predicted_ray_right, real_ray_right).mean()

        angular_difference = (angles_left_deg + angles_right_deg) / 2

        if not is_valid:
            self.log('train/loss', loss)
            self.log('train/diff_mm', diff_mm)
            self.log('train/angular_difference', angular_difference)
        else:
            self.log('valid/diff_mm', diff_mm)
            self.log('valid/angular_difference', angular_difference)
        return loss

    def training_epoch_end(self, _) -> None:
        if FINE_TUNE_WHILE_TRAINING:
            from fine_tune import optimize_me

            self.x_s = []
            self.y_s = []

            # few-shot learning, find personal calibration parameters
            fine_tune_dataloader = self.trainer.datamodule.fine_tune_dataloader()
            x0 = self.all_calibration_params[1:].mean(axis=0).cpu().detach().numpy()  # initial guess is mean of all_calibration_params
            _ = optimize.minimize(lambda x: optimize_me(self, fine_tune_dataloader, 0, x), x0, method='BFGS', jac=True)

            min_idx = np.asarray(self.y_s).argmin()
            x = self.x_s[min_idx]
            print(self.y_s[min_idx], self.x_s[min_idx])

            model.freeze()
            model.all_calibration_params[0] = torch.Tensor(x).to(self.all_calibration_params.device)
            model.unfreeze()

    def validation_step(self, val_batch: dict, batch_idx: int) -> STEP_OUTPUT:
        loss = self.training_step(val_batch, batch_idx, is_valid=True)

        self.log('valid/loss_epoch', loss)
        return loss


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/SPAZE_MPIIFaceGaze')
    parser.add_argument("--visualize", type=bool, default=False)
    args = parser.parse_args()

    DEBUG = args.visualize

    fix_qt()
    seed_everything(42)

    batch_size = 64

    data_module = SpazeDataModule(args.data_path, num_workers, batch_size, fine_tune_while_training=FINE_TUNE_WHILE_TRAINING)
    data_module.setup()

    model = Model(learning_rate=1e-4, weight_decay=1e-5, batch_size=batch_size)

    run_name = 'SPAZE'
    trainer = Trainer(
        gpus=1,
        max_epochs=30,  # "train for 30 epochs" [3.4]
        default_root_dir='./saved_models/',
        logger=[
            TensorBoardLogger("tb_logs", name=run_name),
        ],
        callbacks=[
            ModelCheckpoint(),
        ],
        benchmark=True
    )

    trainer.fit(model, datamodule=data_module)
