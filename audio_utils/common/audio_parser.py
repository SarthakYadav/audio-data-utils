import torch
import torchaudio
import numpy as np
from torch.nn.functional import normalize


class BaseAudioParser(object):
    def __init__(self):
        super().__init__()

    def normalize_sample(self, audio):
        raise NotImplementedError("Abstract method called")

    def __call__(self, audio):
        raise NotImplementedError("Abstract method called")


class LogMelSpecParser(BaseAudioParser):
    def __init__(self, sample_rate=16000,
                 frame_length=400,
                 frame_step=160,
                 n_fft=1024,
                 n_mels=64,
                 fmin=60.0,
                 fmax=7800.0,
                 center=True,
                 normalize=True):
        super(LogMelSpecParser, self).__init__()
        self.spec_gram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=frame_length,
            hop_length=frame_step,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            center=center,
            normalized=normalize
        )

    def normalize_sample(self, audio):
        """dont need to do anything here I guess"""
        return audio

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype("float32")).float()
        single_instance = x.dim() == 1
        if single_instance:
            x = x.reshape(1, -1)
        x = self.spec_gram(x)
        x = torch.clamp(x, min=1e-5, max=1e8)
        x = torch.log(x)
        if single_instance:
            x = x.squeeze()
        return x, None


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
