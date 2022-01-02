from typing import Union
from google.cloud import storage
from audio_utils.common.base_audio_dataset import BaseAudioDataset
from audio_utils.common.utilities import readfile, unpack_batch, TrainingMode
from audio_utils.common.audio_config import AudioConfig


class BasePackedDataset(BaseAudioDataset):
    def __init__(self,
                 manifest_path: str,
                 labels_map: str,
                 audio_config: AudioConfig,
                 mode: Union[str, TrainingMode] = TrainingMode.MULTICLASS,
                 mixer=None, transform=None, is_val=False,
                 pre_feature_transforms=None,
                 post_feature_transforms=None,
                 gcs_bucket_path=None,
                 labels_delimiter=","
                 ):
        super(BasePackedDataset, self).__init__(manifest_path=manifest_path, labels_map=labels_map,
                                                audio_config=audio_config, mode=mode,
                                                mixer=mixer, pre_feature_transforms=pre_feature_transforms,
                                                post_feature_transforms=post_feature_transforms, is_val=is_val,
                                                labels_delimiter=labels_delimiter)
        self.gcs_bucket_path = gcs_bucket_path
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
