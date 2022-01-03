import numpy as np
from audio_utils.common.base_packed_dataset import BasePackedDataset
from audio_utils.common.utilities import load_audio, _collate_fn_multilabel, _collate_fn_multiclass


class PackedDataset(BasePackedDataset):
    def __init__(self, **kwargs):
        super(PackedDataset, self).__init__(**kwargs)

    def __get_item_helper__(self, record):
        lbls = record['label']
        if self.audio_config.cropped_read:
            dur = record['duration']
        else:
            dur = None
        preprocessed_audio = load_audio(record['audio'],
                                        self.audio_config.sr, self.audio_config.min_duration,
                                        read_cropped=self.audio_config.cropped_read,
                                        frames_to_read=self.audio_config.num_frames,
                                        audio_size=dur)
        real, _ = self.__get_feature__(preprocessed_audio)
        if self.pre_feature_transforms:
            real = self.pre_feature_transforms(real)
            # print("after pre_feature_transforms:", real.shape)
        if self.post_feature_transforms:
            real = self.post_feature_transforms(real)
            real = real.squeeze(0)
            # print("after post_feature_transforms:", real.shape)
        label_tensor = self.__parse_labels__(lbls)
        # print("after pre_feature_transforms:", real.shape)
        # if self.transform is not None:
        #     real = self.transform(real)
        return real, label_tensor

    def __getitem__(self, item):
        filepath = self.files[item]
        block = self.read_packed_file(filepath)
        if not self.is_val:
            idxs = np.random.permutation(len(block))
        else:
            idxs = np.arange(len(block))
        # batch = torch.zeros(len(read_block), 1, )
        batch_tensors = []
        batch_labels = []
        # for record in read_block:
        for idx in idxs:
            record = block[idx]
            real, label = self.__get_item_helper__(record)
            # if self.mixer is not None:
            #     real, final_label = self.mixer(self, real, label)
            #     if self.mode != "multiclass":
            #         label = final_label
            batch_tensors.append(real)
            batch_labels.append(label)

        # apply transforms here if needed

        return batch_tensors, batch_labels


def packed_collate_fn_multiclass(batches):
    deflated_batch = []
    for batch in batches:
        for jx in range(len(batch[0])):
            deflated_batch.append((batch[0][jx], batch[1][jx]))
    ixs = np.random.permutation(len(deflated_batch))
    deflated_batch = [deflated_batch[ix] for ix in ixs]
    return _collate_fn_multiclass(deflated_batch)


def packed_collate_fn_multilabel(batches):
    deflated_batch = []
    for batch in batches:
        for jx in range(len(batch[0])):
            deflated_batch.append((batch[0][jx], batch[1][jx]))
    ixs = np.random.permutation(len(deflated_batch))
    deflated_batch = [deflated_batch[ix] for ix in ixs]
    return _collate_fn_multilabel(deflated_batch)
