import numpy as np
from audio_utils.common.transforms import NonOverlappingRandomCrop, RandomCrop
from audio_utils.common.base_packed_dataset import BasePackedDataset
from audio_utils.common.utilities import load_audio, _collate_fn_multilabel, _collate_fn_multiclass, _collate_fn_contrastive
from audio_utils.common.utilities import ConstrastiveCroppingType, Features


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
        if preprocessed_audio is None:
            return None, None
        real, _ = self.__get_waveform__(preprocessed_audio)
        if self.pre_feature_transforms:
            real = self.pre_feature_transforms(real)
            # print("after pre_feature_transforms:", real.shape)
        if self.post_feature_transforms:
            real = self.post_feature_transforms(real)
            if self.audio_config.features != Features.RAW:
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
            if real is None or label is None:
                continue
            else:
                batch_tensors.append(real)
                batch_labels.append(label)
            # if self.mixer is not None:
            #     real, final_label = self.mixer(self, real, label)
            #     if self.mode != "multiclass":
            #         label = final_label
        # apply transforms here if needed

        return batch_tensors, batch_labels


class PackedContrastiveDataset(BasePackedDataset):
    def __init__(self,
                 random_cropping_strategy=ConstrastiveCroppingType.PLAIN,
                 **kwargs):
        super(PackedContrastiveDataset, self).__init__(**kwargs)

        if type(random_cropping_strategy) == str:
            self.random_cropping_strategy = ConstrastiveCroppingType(random_cropping_strategy)
        elif type(random_cropping_strategy) == ConstrastiveCroppingType:
            self.random_cropping_strategy = random_cropping_strategy
        else:
            print(random_cropping_strategy)

        if self.random_cropping_strategy == ConstrastiveCroppingType.NO_OVERLAP:
            self.view_cropper = NonOverlappingRandomCrop(self.audio_config.view_size, self.audio_config.sr)
        else:
            self.view_cropper = RandomCrop(self.audio_config.view_size)

    def __get_item_helper__(self, record):
        if self.audio_config.cropped_read:
            dur = record['duration']
        else:
            dur = None

        if (self.random_cropping_strategy == ConstrastiveCroppingType.PLAIN) and self.audio_config.cropped_read:
            # if cropped read is being done
            # just read twice from the buffer
            preprocessed_audio_i = load_audio(record['audio'],
                                              self.audio_config.sr, self.audio_config.min_duration,
                                              read_cropped=self.audio_config.cropped_read,
                                              frames_to_read=self.audio_config.num_frames,
                                              audio_size=dur)
            preprocessed_audio_j = load_audio(record['audio'],
                                              self.audio_config.sr, self.audio_config.min_duration,
                                              read_cropped=self.audio_config.cropped_read,
                                              frames_to_read=self.audio_config.num_frames,
                                              audio_size=dur)
            if preprocessed_audio_i is None or preprocessed_audio_j is None:
                return None, None
            x_i, _ = self.__get_waveform__(preprocessed_audio_i)
            x_j, _ = self.__get_waveform__(preprocessed_audio_j)
        else:
            preprocessed_audio = load_audio(record['audio'],
                                            self.audio_config.sr, self.audio_config.min_duration,
                                            read_cropped=self.audio_config.cropped_read,
                                            frames_to_read=self.audio_config.num_frames,
                                            audio_size=dur)
            if preprocessed_audio is None:
                return None, None
            audio, _ = self.__get_waveform__(preprocessed_audio)
            if self.random_cropping_strategy == ConstrastiveCroppingType.PLAIN:
                # call self.view_cropper, which is a plain random cropper, twice
                x_i = self.view_cropper(audio)
                x_j = self.view_cropper(audio)
            else:
                # call NonOverlappingRandomCrop view_cropper, which return two non-overlapping crops
                x_i, x_j = self.view_cropper(audio)
        if self.pre_feature_transforms:
            x_i = self.pre_feature_transforms(x_i)
            x_j = self.pre_feature_transforms(x_j)
            # print("after pre_feature_transforms:", real.shape)
        if self.post_feature_transforms:
            x_i = self.post_feature_transforms(x_i)
            x_j = self.post_feature_transforms(x_j)
            if self.audio_config.features != Features.RAW:
                x_i = x_i.squeeze(0)
                x_j = x_j.squeeze(0)
        return x_i, x_j

    def __getitem__(self, item):
        filepath = self.files[item]
        block = self.read_packed_file(filepath)
        if not self.is_val:
            idxs = np.random.permutation(len(block))
        else:
            idxs = np.arange(len(block))
        # batch = torch.zeros(len(read_block), 1, )
        anchors = []
        positives = []
        # for record in read_block:
        for idx in idxs:
            record = block[idx]
            anchor, pos = self.__get_item_helper__(record)
            if anchor is None or pos is None:
                continue
            else:
                anchors.append(anchor)
                positives.append(pos)
        return anchors, positives


def deflate_batches(batches):
    deflated_batch = []
    for batch in batches:
        for jx in range(len(batch[0])):
            deflated_batch.append((batch[0][jx], batch[1][jx]))
    return deflated_batch


def packed_collate_fn_multiclass(batches):
    deflated_batch = deflate_batches(batches)
    ixs = np.random.permutation(len(deflated_batch))
    deflated_batch = [deflated_batch[ix] for ix in ixs]
    return _collate_fn_multiclass(deflated_batch)


def packed_collate_fn_multilabel(batches):
    deflated_batch = deflate_batches(batches)
    ixs = np.random.permutation(len(deflated_batch))
    deflated_batch = [deflated_batch[ix] for ix in ixs]
    return _collate_fn_multilabel(deflated_batch)


def packed_collate_fn_contrastive(batches):
    deflated_batch = deflate_batches(batches)
    return _collate_fn_contrastive(deflated_batch)
