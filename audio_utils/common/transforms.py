import os
import torch
import random

import torch_audiomentations
import torchaudio
import numpy as np
from torchvision.transforms import Compose, transforms
from audio_utils.common.utilities import _check_transform_input


class UseWithProb:
    def __init__(self, transform, prob=.5):
        self.transform = transform
        self.prob = prob

    def __call__(self, image, trg=None):
        if trg is None:
            if random.random() < self.prob:
                image = self.transform(image)
            return image
        else:
            if random.random() < self.prob:
                image, trg = self.transform(image, trg)
            return image, trg


class OneOf:
    def __init__(self, transforms, p=None):
        self.transforms = transforms
        self.p = p

    def __call__(self, image, trg=None):
        transform = np.random.choice(self.transforms, p=self.p)
        if trg is None:
            image = transform(image)
            return image
        else:
            image, trg = transform(image, trg)
            return image, trg


class WaveformTransforms:
    def __init__(self):
        super(WaveformTransforms, self).__init__()

    def check_input(self, batch):
        return _check_transform_input(batch, desired_dims=2)

    def __call__(self, batch):
        raise NotImplementedError()


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        dim = signal.dim()
        if dim == 2:
            if signal.shape[1] <= self.size:
                return signal
            start = random.randint(0, signal.shape[1] - self.size - 1)
            output = signal[:, start: start + self.size]
        if dim == 3:
            if signal.shape[2] <= self.size:
                return signal
            start = random.randint(0, signal.shape[2] - self.size - 1)
            output = signal[:, :, start: start + self.size]
        return output


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, signal):
        dim = signal.dim()
        if dim == 2:
            if signal.shape[1] > self.size:
                start = (signal.shape[1] - self.size) // 2
                return signal[:, start: start + self.size]
            else:
                return signal
        if dim == 3:
            if signal.shape[2] > self.size:
                start = (signal.shape[2] - self.size) // 2
                return signal[:, :, start: start + self.size]
            else:
                return signal


class PadToSize:
    def __init__(self, size, mode='constant'):
        assert mode in ['constant', 'wrap']
        self.size = size
        self.mode = mode

    def __call__(self, signal):
        dim = signal.dim()
        if dim == 2:
            if signal.shape[1] < self.size:
                padding = self.size - signal.shape[1]
                offset = padding // 2
                pad_width = ((0, 0), (offset, padding - offset))
                if self.mode == 'constant':
                    signal = torch.nn.functional.pad(signal, pad_width[1], "constant", value=signal.min())
                else:
                    try:
                        signal = torch.nn.functional.pad(signal, pad_width[1], "replicate")
                    except NotImplementedError as ex:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! signal.shape", signal.shape, self.size)
        elif dim == 3:
            if signal.shape[2] < self.size:
                padding = self.size - signal.shape[1]
                offset = padding // 2
                pad_width = ((0, 0), (offset, padding - offset))
                if self.mode == 'constant':
                    signal = torch.nn.functional.pad(signal, pad_width[1], "constant", value=signal.min())
                else:
                    try:
                        signal = torch.nn.functional.pad(signal, pad_width[1], "replicate")
                    except NotImplementedError as ex:
                        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! signal.shape", signal.shape, self.size)
        return signal


class ClipValue:
    def __init__(self, max_clip_val=0.1):
        self.clamp_factor = max_clip_val

    def __call__(self, x):
        factor = random.uniform(0.0, self.clamp_factor)
        x_min, x_max = x.min(), x.max()
        x.clamp_(min=x_min * factor, max=x_max * factor)
        return x


class TorchAudiomentationTransform:
    def __init__(self, transform):
        super(TorchAudiomentationTransform, self).__init__()
        self.transform = transform

    def _check_torch_audiomentation_inputs(self, x):
        input_dim = x.dim()
        if input_dim == 2:
            spec = x.unsqueeze(1)
        elif input_dim == 1:
            spec = x.unsqueeze(0).unsqueeze(0)
        return spec

    def _post_torch_audiomentation(self, x, orig_input_dim):
        if orig_input_dim == 2:
            x = x.squeeze(1)
        elif orig_input_dim == 1:
            x = x.squeeze()
        return x

    def __call__(self, x):
        input_dim = x.dim()
        spec = self._check_torch_audiomentation_inputs(x)
        spec = self.transform(spec)
        spec = self._post_torch_audiomentation(spec, input_dim)
        return spec


class RandomGain(TorchAudiomentationTransform):
    def __init__(self, min_gain_in_db=-18.0,
                 max_gain_in_db=6.0, prob=0.5, sr=16000):
        gain = torch_audiomentations.Gain(min_gain_in_db=min_gain_in_db,
                                          max_gain_in_db=max_gain_in_db,
                                          p=prob, sample_rate=sr)
        super(RandomGain, self).__init__(gain)


class AddGaussianNoise:
    """
    Adds Random Gaussian Noise
    this changes the amplitude of the audio and can yield values outside [-1., 1.]
    so normalize after this
    """

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015):
        super().__init__()
        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def __call__(self, x):
        noise = torch.randn(x.shape).float()
        random_amplitude = random.uniform(self.min_amplitude, self.max_amplitude)
        return x + random_amplitude * noise


class PeakNormalization(TorchAudiomentationTransform):
    def __init__(self, sr=16000):
        peak_norm = torch_audiomentations.PeakNormalization(apply_to="only_too_loud_sounds",
                                                                 p=1., sample_rate=sr)
        super(PeakNormalization, self).__init__(peak_norm)


class TimeStretch:
    def __init__(self, hop_length, n_freq, time_stretch=0.2):
        super(TimeStretch, self).__init__()
        self.time_stretch_range = (1-time_stretch, 1+time_stretch)
        self.tfs = torchaudio.transforms.TimeStretch(hop_length, n_freq)

    def __call__(self, batch):
        time_stretch_factor = random.uniform(self.time_stretch_range[0], self.time_stretch_range[1])
        batch = self.tfs(batch, time_stretch_factor)
        return batch


class FreqTimeMasking:
    """
    Frequency and Time Masking as part of the SpecAugment paradigm
    TimeStretching -> Real conversion should precede these transform
    """
    def __init__(self,
                 freq_mask_param,
                 time_mask_param,
                 num_freq_masks=2,
                 num_time_masks=2,
                 mask_mode="zero",
                 freq_axis=1,
                 time_axis=2
                 ):
        super(FreqTimeMasking, self).__init__()
        assert num_freq_masks > 0
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        assert mask_mode in ["zero"]
        self.mask_mode = mask_mode
        self.freq_axis = freq_axis
        self.time_axis = time_axis

    def get_mask_value(self, batch):
        if self.mask_mode == "zero":
            return 0.
        elif self.mask_mode == 'min':
            return batch.min()
        else:
            return batch.max()

    def __call__(self, batch):
        """
        :param batch: torch tensor of shape (N,F,T)
        :return: masked batch
        """
        num_freq_mask = random.randint(1, self.num_freq_masks)
        num_time_masks = random.randint(1, self.num_time_masks)
        mask_value = self.get_mask_value(batch)
        for ix in range(num_freq_mask):
            batch = torchaudio.functional.mask_along_axis(batch, self.freq_mask_param, mask_value, self.freq_axis)

        for ix in range(num_time_masks):
            batch = torchaudio.functional.mask_along_axis(batch, self.time_mask_param, mask_value, self.time_axis)
        return batch


BATCHED_TRANSFORMS = {
    "random_gain": RandomGain,
    "peak_norm": PeakNormalization,
    "gaussian_noise": AddGaussianNoise,
}

RAW_WAVEFORM_TRANSFORMS = {
    "pad_to_size": PadToSize
}