import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchvision import models


class Backbone(LightningModule):
    def __init__(self, backbone: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.backbone.fc = nn.Sequential()  # empty sequential layer does nothing (pass-through)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)


def fc_block(input_size: int, out_size: int):
    return nn.Sequential(
        nn.Linear(input_size, out_size),
        nn.BatchNorm1d(out_size),
        nn.ReLU(),
        nn.Dropout()
    )


class Spaze(LightningModule):
    """
    The network has five outputs.
    For each eye, it predicts a 2D gaze origin o_2D and a 2D gaze direction d_2D.
    It also generates a distance correction term, c, which is common to both eyes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.N = 3  # "find that 3 parameters per eye are enough to provide an efficient person-specific gaze tracking" [3.4]
        self.all_calibration_params = nn.Parameter(torch.rand(15, 2 * self.N))  # 2N personal calibration parameters (learned during training)

        self.cnn_face = Backbone(models.resnet18(pretrained=True))
        self.cnn_eye = Backbone(models.resnet18(pretrained=True))

        # self.cnn_face.freeze()
        # self.cnn_eye.freeze()

        # "FC(3072)→BN→ReLU→DO→FC(3072)→BN→ReLU→DO→FC(1 or 4)" [3.3]
        self.fc1 = nn.Sequential(
            fc_block(512 * 3, 3072),
            fc_block(3072, 3072),
            nn.Linear(3072, 1)
        )
        self.fc2 = nn.Sequential(
            fc_block(512 + 1 + self.N, 3072),
            fc_block(3072, 3072),
            nn.Linear(3072, 4)
        )

    def forward(self, person_idx: torch.Tensor, face: torch.Tensor, right_eye: torch.Tensor, left_eye: torch.Tensor):
        """

        :param person_idx: index of the person
        :param face: face image
        :param right_eye: left eye image
        :param left_eye: right eye image
        :return: a distance correction term (c) +  2D gaze origin (o_2D) and a 2D gaze direction (d_2D) for each eye
        """

        out_face = self.cnn_face(face)
        out_right_eye = self.cnn_eye(right_eye)
        out_left_eye = self.cnn_eye(left_eye)

        input_fc1 = torch.cat((out_face, out_right_eye, out_left_eye), dim=1)
        c = self.fc1(input_fc1)  # distance correction (common to both eyes)

        calibration_params = self.all_calibration_params[person_idx].squeeze(1)

        # The convolutional network output for each eye is concatenated with a set of N personal calibration parameters and the distance correction.
        input_fc2_right_eye = torch.cat((out_right_eye, calibration_params[:, : self.N], c), dim=1)
        out_fc2_right_eye = self.fc2(input_fc2_right_eye)

        input_fc2_left_eye = torch.cat((out_left_eye, calibration_params[:, self.N:], c), dim=1)
        out_fc2_left_eye = self.fc2(input_fc2_left_eye)

        return c, out_fc2_right_eye[:, :2], out_fc2_right_eye[:, 2:], out_fc2_left_eye[:, :2], out_fc2_left_eye[:, 2:]


if __name__ == '__main__':
    model = Spaze()
    model.summarize(mode="top")

    from torchinfo import summary

    batch_size = 16
    summary(model, [
        (batch_size, 1),
        (batch_size, 3, 224, 56),
        (batch_size, 3, 224, 112), (batch_size, 3, 224, 112)
    ], dtypes=[torch.long, torch.float, torch.float, torch.float])
