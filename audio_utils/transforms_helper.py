import torch
from audio_utils.common.audio_config import AudioConfig
from audio_utils.common.utilities import Features
from audio_utils.common import transforms, feature_transforms


def basic_supervised_transforms(audio_config: AudioConfig):
    # transforms corresponding to initial successful AudioSet supervised training
    # from cola-pytorch commit 21d1df2c
    if audio_config.features == Features.RAW:
        random_clip_size = audio_config.view_size
        val_clip_size = audio_config.view_size
        is_raw = True
    else:
        random_clip_size = audio_config.tr_feature_size
        val_clip_size = audio_config.val_feature_size
        is_raw = False
    pre_transforms = transforms.PeakNormalization()
    train_tfs = {
        'pre': pre_transforms
    }
    val_tfs = {
        "pre": pre_transforms
    }
    mode = "per_instance"
    if audio_config.features == Features.RAW:
        # Raw waveform processor is called either way
        # just add augmentations
        train_tfs['post'] = transforms.Compose(
            [
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.SPECTROGRAM:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            normalize=audio_config.normalize_features,
            log_compress=True,
            mode=mode)

        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.LOGMEL:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            log_compress=False,
            normalize=audio_config.normalize_features,
            mode=mode
        )
        mel_trans = feature_transforms.ToMelScale(audio_config.sr, audio_config.hop_len,
                                                  n_fft=audio_config.n_fft, n_mels=audio_config.n_mels,
                                                  fmin=audio_config.fmin, fmax=audio_config.fmax)
        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.CenterCrop(val_clip_size)
            ])
    return train_tfs, val_tfs


def basic_contrastive_transforms(audio_config: AudioConfig):
    # transforms corresponding to initial successful AudioSet supervised training
    # from cola-pytorch commit 21d1df2c
    if audio_config.features == Features.RAW:
        random_clip_size = audio_config.view_size
        val_clip_size = audio_config.view_size
        is_raw = True
    else:
        random_clip_size = audio_config.tr_feature_size
        val_clip_size = audio_config.val_feature_size
        is_raw = False
    pre_transforms = transforms.Compose([
        transforms.AddGaussianNoise(),
        transforms.PeakNormalization()
    ])
    train_tfs = {
        'pre': pre_transforms
    }
    val_tfs = {
        "pre": pre_transforms
    }
    mode = "per_instance"
    if audio_config.features == Features.RAW:
        # Raw waveform processor is called either way
        # just add augmentations
        train_tfs['post'] = transforms.Compose(
            [
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.SPECTROGRAM:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            normalize=audio_config.normalize_features,
            log_compress=True,
            mode=mode)

        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc,
                transforms.CenterCrop(val_clip_size)
            ])
    elif audio_config.features == Features.LOGMEL:
        spec_gram = feature_transforms.SpectrogramParser(
            window_length=audio_config.win_len,
            hop_length=audio_config.hop_len,
            n_fft=audio_config.n_fft,
            mode=mode)
        spec_post_proc = feature_transforms.SpectrogramPostProcess(
            window_length=audio_config.win_len,
            log_compress=False,
            normalize=audio_config.normalize_features,
            mode=mode
        )
        mel_trans = feature_transforms.ToMelScale(audio_config.sr, audio_config.hop_len,
                                                  n_fft=audio_config.n_fft, n_mels=audio_config.n_mels,
                                                  fmin=audio_config.fmin, fmax=audio_config.fmax)
        train_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.RandomCrop(random_clip_size)
            ])
        val_tfs['post'] = transforms.Compose(
            [
                spec_gram, spec_post_proc, mel_trans,
                transforms.CenterCrop(val_clip_size)
            ])
    return train_tfs, val_tfs