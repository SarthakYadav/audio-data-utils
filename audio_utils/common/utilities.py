import os
import io
import enum
import torch
import random
import msgpack
import numpy as np
import soundfile as sf
import msgpack_numpy as msg_np


def _collate_fn_multiclass(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    targets = torch.LongTensor(minibatch_size)

    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets[x] = target
    if channel_size != 1:
        inputs = inputs.unsqueeze(1)
    return inputs, targets


def _collate_fn_multilabel(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    # print("channel size:", channel_size)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, channel_size, max_seqlength)
    targets = []

    for x in range(minibatch_size):
        sample = batch[x]
        real_tensor = sample[0]
        target = sample[1]
        seq_length = real_tensor.size(1)
        # inputs[x] = real_tensor
        inputs[x].narrow(1, 0, seq_length).copy_(real_tensor)
        targets.append(target.unsqueeze(0))
    targets = torch.cat(targets)
    if channel_size != 1:
        inputs = inputs.unsqueeze(1)
    return inputs, targets


def _collate_fn_contrastive(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    channel_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    batch_xi = torch.zeros(minibatch_size, channel_size, max_seqlength)
    batch_xj = torch.zeros(minibatch_size, channel_size, max_seqlength)
    targets = torch.arange(minibatch_size).long()
    for ix in range(minibatch_size):
        sample = batch[ix]
        x_i = sample[0]
        x_j = sample[1]
        seq_length_i = x_i.size(1)
        seq_length_j = x_j.size(1)
        batch_xi[ix].narrow(1, 0, seq_length_i).copy_(x_i)
        batch_xj[ix].narrow(1, 0, seq_length_j).copy_(x_j)

    if channel_size != 1:
        # assume it's not raw waveform signal
        batch_xi = batch_xi.unsqueeze(1)
        batch_xj = batch_xj.unsqueeze(1)
    return batch_xi, batch_xj, targets


def readfile(f):
    with open(f, "rb") as stream:
        data = stream.read()
    return data


def unpack_batch(buffer):
    data = msgpack.unpackb(buffer, object_hook=msg_np.decode)
    return data


def sf_read_wrapper(f, frames_to_read=-1, start_idx=0):
    if os.path.isfile(f):
        x, clip_sr = sf.read(f, frames=frames_to_read, start=start_idx)
    else:
        with io.BytesIO(f) as buf:
            x, clip_sr = sf.read(buf, frames=frames_to_read, start=start_idx)
    return x, clip_sr


def load_audio(f, sr, min_duration: float = 5.,
               read_cropped=False, frames_to_read=-1, audio_size=None):
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    load_full = True
    if read_cropped:
        load_full = False
        assert audio_size
        assert frames_to_read != -1
        if frames_to_read >= audio_size:
            x, clip_sr = sf_read_wrapper(f)
        else:
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
            try:
                x, clip_sr = sf_read_wrapper(f, frames_to_read=frames_to_read, start_idx=start_idx)
            except RuntimeError as ex:
                load_full = True
                print("{} {} {}. Attempting full read..".format(ex, start_idx, frames_to_read))
            if load_full:
                try:
                    x, clip_sr = sf_read_wrapper(f)
                    x = x[start_idx:start_idx+frames_to_read]
                except RuntimeError as ex:
                    print("Catastrophic read failure. {} {} {}".format(ex, start_idx, frames_to_read))
                    return None
        min_samples = frames_to_read
    else:
        try:
            x, clip_sr = sf_read_wrapper(f)
        except RuntimeError as ex:
            print("Catastrophic read failure. {} {}".format(ex, f))
            return None

    x = x.astype('float32')
    assert clip_sr == sr
    # pad if needed
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x


def read_np_audio(f, dtype="float32"):
    if os.path.isfile(f):
        return np.load(f).astype(dtype)
    else:
        return np.frombuffer(f, dtype=dtype)


def load_numpy_buffer(f, sr, min_duration: float = 5.,
                      read_cropped=False, frames_to_read=-1, audio_size=None,
                      dtype="float32"):
    # sr = record['sr']
    if min_duration is not None:
        min_samples = int(sr * min_duration)
    else:
        min_samples = None
    try:
        x = read_np_audio(f, dtype)
    except RuntimeError as ex:
        print("Catastrophic read failure. {}".format(ex))
        return None
    if read_cropped:
        assert frames_to_read != -1
        if not (frames_to_read >= audio_size):
            start_idx = random.randint(0, audio_size - frames_to_read - 1)
            x = x[start_idx:start_idx + frames_to_read]
    if min_samples is not None:
        if len(x) < min_samples:
            tile_size = (min_samples // x.shape[0]) + 1
            x = np.tile(x, tile_size)[:min_samples]
    return x


@enum.unique
class Features(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    RAW = "raw"
    SPECTROGRAM = "spectrogram"
    LOGMEL = "log_mel"


@enum.unique
class TrainingMode(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    CONTRASTIVE = "contrastive"


def _check_transform_input(audio_sample, desired_dims=2):
    if isinstance(audio_sample, np.ndarray):
        audio_sample = torch.from_numpy(audio_sample.astype("float32")).float()
    if audio_sample.dim() == 1:
        audio_sample = audio_sample.reshape(1, -1)
    if audio_sample.dim() != desired_dims:
        raise ValueError("An array/tensor of shape (batch, T) or (1, T) is needed")
    return audio_sample


@enum.unique
class ConstrastiveCroppingType(enum.Enum):
    """Look up for similarity measure in contrastive model."""
    PLAIN = "plain"
    NO_OVERLAP = "no_overlap"


@enum.unique
class RecordFormat(enum.Enum):
    """
    format in which audio is saved in records

    NUMPY: the audio in the records is a numpy array stored as bytes
    ENCODED: the audio in the records is a full audio file of .wav/.flac stored as bytes
    """
    NUMPY = "numpy"
    ENCODED = "encoded"
