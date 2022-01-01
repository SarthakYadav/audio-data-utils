import os
from audio_utils.common.utilities import Features
from collections import OrderedDict


class AudioConfig(object):
    def __init__(self, **kwargs):
        super(AudioConfig, self).__init__()
        self.kwargs = kwargs
        if self.kwargs:
            self.parse_from_config(self.kwargs)

    def parse_from_config(self, audio_config):
        self.sr = int(audio_config.get("sample_rate", 16000))
        self.normalize = bool(audio_config.get("normalize", False))
        self.min_duration = float(audio_config.get("min_duration", 2.5))
        self.background_noise_path = audio_config.get("background_noise_path", None)
        self.features = Features(audio_config.get("features", "raw"))
        self.view_size = int(audio_config.get("random_clip_size", 2.5) * self.sr)
        self.cropped_read = bool(audio_config.get("cropped_read", False))

        for key in audio_config:
            if hasattr(self, key):
                continue
            setattr(self, key, audio_config[key])

        # now validate attributes
        if self.features == Features.LOGMEL:
            for k in ["n_fft", "win_len", "hop_len", "n_mels", "fmin", "fmax"]:
                assert hasattr(self, k), "{} not found".format(k)
        # other necessary attributes
        if self.cropped_read:
            self.num_frames = int(audio_config.get("random_clip_size", 2.5) * self.sr)
        else:
            self.num_frames = -1
