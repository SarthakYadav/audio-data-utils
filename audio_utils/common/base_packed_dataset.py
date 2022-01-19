from typing import Union
from google.cloud import storage
from audio_utils.common.base_audio_dataset import BaseAudioDataset
from audio_utils.common.utilities import readfile, unpack_batch, TrainingMode, RecordFormat
from audio_utils.common.utilities import load_numpy_buffer, load_audio
from audio_utils.common.audio_config import AudioConfig


class BasePackedDataset(BaseAudioDataset):
    def __init__(self,
                 manifest_path: str,
                 audio_config: AudioConfig,
                 labels_map: str = None,
                 mode: Union[str, TrainingMode] = TrainingMode.MULTICLASS,
                 mixer=None, is_val=False,
                 pre_feature_transforms=None,
                 post_feature_transforms=None,
                 gcs_bucket_path=None,
                 labels_delimiter=",",
                 record_format=RecordFormat.ENCODED
                 ):
        super(BasePackedDataset, self).__init__(manifest_path=manifest_path, labels_map=labels_map,
                                                audio_config=audio_config, mode=mode,
                                                mixer=mixer, pre_feature_transforms=pre_feature_transforms,
                                                post_feature_transforms=post_feature_transforms, is_val=is_val,
                                                labels_delimiter=labels_delimiter)
        self.gcs_bucket_path = gcs_bucket_path
        if type(record_format) == str:
            self.record_format = RecordFormat(record_format)
        else:
            self.record_format = record_format
        if self.gcs_bucket_path:
            self.client = None
            self.bucket = None

    def init_gcs(self):
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.gcs_bucket_path)

    def read_packed_file(self, path):
        if self.gcs_bucket_path:
            if self.client is None:
                self.init_gcs()
            blob = self.bucket.blob(path)
            with blob.open("rb") as fp:
                buffer = fp.read()
        else:
            buffer = readfile(path)
        block = unpack_batch(buffer)
        return block

    def audio_loader_helper(self, audio, sr, min_duration, read_cropped,
                            frames_to_read, audio_size, dtype="float32"):
        if self.record_format == RecordFormat.ENCODED:
            audio = load_audio(audio, sr, min_duration, read_cropped=read_cropped,
                               frames_to_read=frames_to_read, audio_size=audio_size)
        else:
            audio = load_numpy_buffer(audio, sr, min_duration, read_cropped=read_cropped,
                                      frames_to_read=frames_to_read, audio_size=audio_size, dtype=dtype)
        return audio
