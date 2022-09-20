from pathlib import Path
from typing import Union, Type, Optional, Callable

import numpy as np
import random as rand
import librosa
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1


def transform_clamp(minimum, maximum):
    def clamp(tensor):
        return torch.clamp(tensor, minimum, maximum)
    return clamp


def transform_log():
    def log(tensor):
        return torch.log10(tensor)
    return log


def transform_remove_inf():
    def remove_inf(tensor):
        tensor[tensor.isinf()] = 0
        return tensor

    return remove_inf


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def load_sound_file(path: Path, sr):
    try:
        data, rate = librosa.load(str(path),
                                  sr=sr)

    except Exception as e:
        print(f"Reading of sample {path.name} failed")
        raise e

    return data, rate


class SongIngestion(torch.utils.data.Dataset):
    def __init__(self, metadata, sample_length: int, transformations,
                 sr: int, n_mels: int, sound_files: dict, seed=1994):
        super(SongIngestion).__init__()
        self.metadata = metadata
        self.n_mels = n_mels

        self.n = 0
        rand.seed(seed)
        np.random.seed(seed)

        self.start = 0
        self.end = self.metadata.shape[0]
        self.sound_files = sound_files
        self.sample_length = sr * sample_length
        self.transformations = transformations
        self.sr = sr
        self.indexes = self.metadata.index.to_list()

    def load_sound_file(self, track_id):
        if track_id not in self.sound_files:
            self.sound_files[track_id] = load_sound_file(self.metadata.loc[[track_id]].iloc[0, -1], self.sr)
        return self.sound_files[track_id]

    def load_sample(self, index):
        sample, rate = self.load_sound_file(index)

        if len(sample) > self.sample_length:
            sample_start = rand.randint(0, len(sample) - self.sample_length)
            sample = sample[sample_start: (sample_start + self.sample_length)]
        else:
            new_sample = np.zeros(self.sample_length, sample.dtype)
            sample_start = rand.randint(0, len(new_sample) - self.sample_length)
            new_sample[sample_start:(sample_start + len(sample))] = sample
            sample = new_sample

        sample = librosa.feature.melspectrogram(y=sample)
        sample = sample.reshape((1, sample.shape[0], sample.shape[1]))
        sample = torch.from_numpy(sample)
        transformed_sample = self.transformations(sample)

        return transformed_sample, sample

    def __getitem__(self, index):
        transformed_sample, sample = self.load_sample(self.indexes[index])
        return transformed_sample, sample

    def __len__(self):
        return self.end


def conv_transpose3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv_transpose1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockTranspose(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_transpose3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_transpose3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleneckTranspose(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_transpose1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv_transpose3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv_transpose1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.Tanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out





class SoundGenerator(nn.Module):
    def __init__(self) -> None:
        super(SoundGenerator, self).__init__()
        self.inplanes = 4
        self._norm_layer = nn.BatchNorm2d
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.inplanes, kernel_size=5, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),

            self._make_layer_encode(Bottleneck, 32, 2, stride=2),
            self._make_layer_encode(Bottleneck, 64, 2, stride=2),
            self._make_layer_encode(BasicBlock, 256, 2, stride=2),
            self._make_layer_encode(Bottleneck, 256, 2, stride=2),
            self._make_layer_encode(BasicBlock, 512, 2, stride=2),
            self._make_layer_encode(BasicBlock, 512, 2, stride=2),
        )

        self.encode_flat = nn.Sequential(
            nn.Linear(512*2*7, 2048),
            nn.ReLU()
        )

        self.decode_flat = nn.Sequential(
            nn.Linear(2048, 512*2*7),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            self._make_layer_decode(BasicBlockTranspose, 512, 2, stride=2),
            self._make_layer_decode(BottleneckTranspose, 256, 2, stride=2),
            self._make_layer_decode(BasicBlockTranspose, 256, 2, stride=2),
            self._make_layer_decode(BottleneckTranspose, 64, 2, stride=2),
            self._make_layer_decode(BasicBlockTranspose, 32, 2, stride=2),
            self._make_layer_decode(BasicBlockTranspose, 16, 2, stride=2),
            self._make_layer_decode(BasicBlockTranspose, 8, 2, stride=2),
            self._make_layer_decode(BasicBlockTranspose, 4, 2, stride=2),
            nn.Conv2d(4, 1, kernel_size=(3, 26), stride=(2, 1), dilation=(1, 27))
        )

    def _make_layer_encode(self, block: Type[Union[BasicBlock, Bottleneck]],
                           planes: int, blocks: int, stride: int = 1, dilate: bool = False,) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _make_layer_decode(self, block: Type[Union[BasicBlock, Bottleneck]],
                           planes: int, blocks: int, stride: int = 1, dilate: bool = False,) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv_transpose1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer)]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def encode(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.encoder(sample)
        x = torch.flatten(x, 1)
        x = self.encode_flat(x)
        return x

    def decode(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.decode_flat(sample)
        x = x.reshape((x.shape[0], 512, 2, 7))
        x = self.decoder(x)
        return x

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.encode(sample)
        x = self.decode(x)
        return x
