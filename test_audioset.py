import os
import tqdm
import time
from audio_utils.packed_datasets import PackedDataset, packed_collate_fn_multiclass, packed_collate_fn_multilabel
from audio_utils.common.audio_config import AudioConfig
from audio_utils.common import transforms, feature_transforms
from torch.utils.data import DataLoader
from audio_utils.common.utilities import Features
import argparse
import matplotlib.pyplot as plt
import torchaudio


def save_batch(batch, output_fld):
    if not os.path.exists(output_fld):
        os.makedirs(output_fld)
    for ix in range(len(batch)):
        x = batch[ix]
        torchaudio.save(os.path.join(output_fld, "sample_{:03d}.flac".format(ix)), x, 16000)


def save_spec_batch(batch, output_fld):
    if not os.path.exists(output_fld):
        os.makedirs(output_fld)
    for ix in range(len(batch)):
        im_x = batch[ix]
        if im_x.size(0) == 1:       # channel size bruh
            im_x = im_x[0]
        else:
            im_x = im_x.permute(1, 2, 0)

        plt.imshow(im_x.numpy())
        plt.axis("off")
        plt.savefig(os.path.join(output_fld, "sample_{:03d}.png".format(ix)))
        plt.clf()
        plt.cla()


parser = argparse.ArgumentParser()
parser.add_argument("--post_batch", action="store_true")
parser.add_argument("--save_batch", action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    bsize = 2
    ac = {'features': 'log_mel', 'normalize': False, 'sample_rate': 16000, 'min_duration': 5, 'random_clip_size': 5,
          'val_clip_size': 5, 'mixup': True, 'mixup_alpha': 0.1, "cropped_read": True, "fmin": 60, "fmax": 7800, "n_fft": 1024,
          "win_len": 400, "hop_len": 160, "tr_feature_size": 501, "val_feature_size": 501, "n_mels": 64}
    audio_config = AudioConfig()
    audio_config.parse_from_config(ac)
    print(audio_config.features)
    tr = "/media/sarthak/nvme/datasets/audioset_small_blocks/meta/eval.csv"
    lbl_map = "/media/sarthak/nvme/datasets/audioset_small_blocks/meta/lbl_map.json"

    pre_features = transforms.Compose(
        [
            transforms.RandomGain(),
            transforms.AddGaussianNoise(),
            transforms.PeakNormalization(),
        ])
    if args.post_batch:
        mode = "after_batch"
    else:
        mode = "per_instance"
    if audio_config.features == Features.RAW:
        post_feature = None
    else:
        post_feature = transforms.Compose(
            [
                feature_transforms.SpectrogramParser(mode=mode),
                # transforms.UseWithProb(transforms.TimeStretch(hop_length=160, n_freq=201), prob=0.1),
                feature_transforms.SpectrogramPostProcess(mode=mode),
                # feature_transforms.ToMelScale(n_fft=400),
                # transforms.PadToSize(501),
                transforms.CenterCrop(501)
            ]
        )
    if args.post_batch:
        dset = PackedDataset(manifest_path=tr, labels_map=lbl_map, audio_config=audio_config,
                             mode="multilabel", labels_delimiter=";", pre_feature_transforms=pre_features,
                             post_feature_transforms=None, gcs_bucket_path=os.environ.get("GCS_BUCKET_PATH", None))
    else:
        dset = PackedDataset(manifest_path=tr, labels_map=lbl_map, audio_config=audio_config,
                             mode="multilabel", labels_delimiter=";", pre_feature_transforms=pre_features,
                             post_feature_transforms=post_feature, gcs_bucket_path=os.environ.get("GCS_BUCKET_PATH", None))

    loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=8, collate_fn=packed_collate_fn_multilabel)
    lf = len(loader)
    bs_sim_time = bsize * 0.2611
    cnt = 0
    t0 = time.time()

    for batch in tqdm.tqdm(loader):
        x, y = batch
        if args.post_batch:
            feature_x = post_feature(x)
        if cnt == 0:
            print(x.shape, y.shape)
            if args.post_batch:
                print(feature_x.shape)
            if args.save_batch:
                save_batch(x, "./test_batch/audio")
                save_spec_batch(feature_x, "./test_batch/specs")
        cnt += 1
    t1 = time.time()
    print("time taken:", t1-t0)
