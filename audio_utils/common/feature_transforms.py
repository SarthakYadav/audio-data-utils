import torch
import torchaudio
import numpy as np
from torch.nn.functional import normalize
from audio_utils.common.utilities import _check_transform_input


class BaseAudioParser(object):
    def __init__(self):
        super().__init__()

    def check_sample(self, audio_sample):
        return _check_transform_input(audio_sample)

    def __call__(self, audio):
        raise NotImplementedError("Abstract method called")


class SpectrogramParser(BaseAudioParser):
    def __init__(self,
                 window_length=400,
                 hop_length=160,
                 n_fft=400,
                 center=True,
                 window_fn=torch.hann_window,
                 pad=0,
                 pad_mode="reflect"):
        super(SpectrogramParser, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_length = window_length
        self.center = center
        self.return_complex = True
        # self.window_fn = window_fn
        self.window = window_fn(window_length)
        self.pad = pad
        self.pad_mode = pad_mode

        # always returns complex
        # therefore, output will be complex tensor, which is desired for SpecAugment
        # Hence, add a transform on top to convert to absolute value and desired power

    def __call__(self, batch):
        """

        :param batch: float array/tensor of shape (N, T) or (T,) for a single input
        :return: tensor of dtype complex
        """
        batch = self.check_sample(batch)
        batch = torchaudio.functional.spectrogram(
            batch,
            self.pad,
            self.window,
            self.n_fft,
            self.hop_length,
            self.window_length,
            power=None,             # power other than None enforces real valued spec
            normalized=False,       # normalization enforces real valued spec
            center=self.center,
            pad_mode=self.pad_mode,
            onesided=True,
            return_complex=True
        )
        return batch


class SpectrogramPostProcess:
    def __init__(self,
                 window_length=400,
                 window_fn=torch.hann_window,
                 power=2,
                 normalize=True,
                 log_compress=True):
        super(SpectrogramPostProcess, self).__init__()
        self.power = power
        self.normalize = normalize
        self.window = window_fn(window_length)
        self.log_compress = log_compress
        if log_compress:
            print("log_compression is set to True in SpectrogramPostProcess. If using MelScale down the line, disable it")

    def __call__(self, batch):
        """

        :param batch: float tensor of shape (N, F, T)
        :return:
        """
        if self.normalize:
            batch /= self.window.pow(2.).sum().sqrt()
        if self.power:
            if self.power == 1.0:
                batch = batch.abs()
            else:
                batch = batch.abs().pow(self.power)
        if self.log_compress:
            batch = torch.clamp(batch, min=1e-8, max=1e8)
            batch = torch.log(batch)
        return batch


class ToMelScale(BaseAudioParser):
    def __init__(self,
                 sample_rate=16000,
                 window_length=400,
                 hop_length=160,
                 n_fft=1024,
                 n_mels=64,
                 fmin=60.0,
                 fmax=7800.0,
                 norm=None,
                 center=True,
                 mel_scale="htk"):
        super(ToMelScale, self).__init__()
        self.sample_rate = sample_rate
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.mel_scale = torchaudio.transforms.MelScale(
            self.n_mels,
            self.sample_rate,
            self.fmin,
            self.fmax,
            self.n_fft // 2 + 1,
            norm,
            mel_scale
        )

    def __call__(self, batch):
        """
        Accepts output of SpectrogramParser -> ... -> SpectrogramPostProcess and converts it to MelScale
        This pipeline allows us to use torchaudio.transforms.TimeStretching
        :param batch:
        :return:
        """
        batch = self.mel_scale(batch)
        batch = torch.clamp(batch, min=1e-8, max=1e8)
        batch = torch.log(batch)
        return batch


class RawAudioParser(BaseAudioParser):
    """
    :param normalize_waveform
        whether to N(0,1) normalize audio waveform
    """
    def __init__(self, normalize_waveform=False):
        super().__init__()
        self.normalize_waveform = normalize_waveform
        if self.normalize_waveform:
            print("ATTENTION!!! Normalizing waveform")

    def normalize_sample(self, audio):
        return normalize(audio, 2, dim=-1)

    def __call__(self, audio):
        output = torch.from_numpy(audio.astype("float32")).float()
        if self.normalize_waveform:
            output = self.normalize_sample(output)
        output = output.unsqueeze(0)
        return output, None
