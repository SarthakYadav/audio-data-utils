import os
import json
import torch
import pandas as pd
from typing import Union
from torch.utils.data import Dataset
from audio_utils.common.utilities import Features, TrainingMode
from audio_utils.common.audio_config import AudioConfig
from audio_utils.common.feature_transforms import RawAudioParser


class BaseAudioDataset(Dataset):
    def __init__(self,
                 manifest_path: str,
                 labels_map: str,
                 audio_config: AudioConfig,
                 mode: Union[str, TrainingMode] = TrainingMode.MULTICLASS,
                 mixer=None, pre_feature_transforms=None,
                 post_feature_transforms=None,
                 is_val=False,
                 labels_delimiter=","):
        super(BaseAudioDataset, self).__init__()
        assert os.path.isfile(labels_map)
        assert os.path.splitext(labels_map)[-1] == ".json"
        assert audio_config is not None
        self.is_val = is_val
        self.pre_feature_transforms = pre_feature_transforms
        self.post_feature_transforms = post_feature_transforms
        self.mixer = mixer
        if type(mode == "str"):
            self.mode = TrainingMode(mode)
        else:
            self.mode = mode
        self.manifest_path = manifest_path
        with open(labels_map, 'r') as fd:
            self.labels_map = json.load(fd)
        self.audio_config = audio_config
        self.labels_delimiter = labels_delimiter

        df = pd.read_csv(manifest_path)
        files = df['files'].values.tolist()
        self.files = files
        self.length = len(self.files)

        self.raw_waveform_parser = RawAudioParser(normalize_waveform=self.audio_config.normalize)
        # if self.audio_config.features == Features.RAW:
        #     self.audio_parser =
        # elif self.audio_config.features == Features.LOGMEL:
        #     self.audio_parser = LogMelSpecParser(
        #         sample_rate=self.audio_config.sr,
        #         frame_length=self.audio_config.win_len,
        #         frame_step=self.audio_config.hop_len,
        #         n_fft=self.audio_config.n_fft,
        #         n_mels=self.audio_config.n_mels,
        #         normalize=self.audio_config.normalize,
        #         fmin=self.audio_config.fmin,
        #         fmax=self.audio_config.fmax
        #     )
        # else:
        #     raise ValueError("In valid feature type {} requested. Supported are ['raw', 'log_mel']")

    def __get_feature__(self, audio):
        return self.raw_waveform_parser(audio)

    def __get_item_helper__(self, record):
        raise NotImplementedError("Abstract method called")

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        raise NotImplementedError("Abstract method called")

    def __parse_labels__(self, lbls):
        if self.mode == TrainingMode.CONTRASTIVE:
            return None
        elif self.mode == TrainingMode.MULTILABEL:
            label_tensor = torch.zeros(len(self.labels_map)).float()
            for lbl in lbls.split(self.labels_delimiter):
                label_tensor[self.labels_map[lbl]] = 1

            return label_tensor
        elif self.mode == TrainingMode.MULTICLASS:
            # print("multiclassssss")
            return self.labels_map[lbls]
        else:
            return ValueError("Unsupporting training mode {}".format(self.mode))