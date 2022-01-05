import os
import tqdm
import time
from audio_utils.packed_datasets import PackedContrastiveDataset, packed_collate_fn_contrastive
from audio_utils.common.audio_config import AudioConfig
from audio_utils.common import transforms, feature_transforms
from torch.utils.data import DataLoader
from audio_utils.common.utilities import Features
import argparse
import matplotlib.pyplot as plt
import torchaudio


def save_batch(anchors, positives, output_fld):
    anchors_fld = os.path.join(output_fld, "anchors_spec")
    pos_fld = os.path.join(output_fld, "pos_spec")
    if not os.path.exists(anchors_fld):
        os.makedirs(anchors_fld)
    if not os.path.exists(pos_fld):
        os.makedirs(pos_fld)
    for ix in range(len(anchors)):
        anchor = anchors[ix]
        pos = positives[ix]
        torchaudio.save(os.path.join(anchors_fld, "sample_{:03d}.flac".format(ix)), anchor, 16000)
        torchaudio.save(os.path.join(pos_fld, "sample_{:03d}.flac".format(ix)), pos, 16000)


def fix_tensor_for_matplotlib(x):
    if x.size(0) == 1:  # channel size bruh
        x = x[0]
    else:
        x = x.permute(1, 2, 0)
    return x


def save_spec_batch(anchors, positives, output_fld):
    anchors_fld = os.path.join(output_fld, "anchors_spec")
    pos_fld = os.path.join(output_fld, "pos_spec")
    if not os.path.exists(anchors_fld):
        os.makedirs(anchors_fld)
    if not os.path.exists(pos_fld):
        os.makedirs(pos_fld)
    for ix in range(len(anchors)):
        im_anchor = anchors[ix]
        im_pos = positives[ix]

        im_anchor = fix_tensor_for_matplotlib(im_anchor)
        im_pos = fix_tensor_for_matplotlib(im_pos)

        plt.imshow(im_anchor.numpy())
        plt.axis("off")
        plt.savefig(os.path.join(anchors_fld, "sample_{:03d}.png".format(ix)))
        plt.clf()
        plt.cla()

        plt.imshow(im_pos.numpy())
        plt.axis("off")
        plt.savefig(os.path.join(pos_fld, "sample_{:03d}.png".format(ix)))
        plt.clf()
        plt.cla()


parser = argparse.ArgumentParser()
parser.add_argument("--post_batch", action="store_true")
parser.add_argument("--save_batch", action="store_true")


if __name__ == '__main__':
    args = parser.parse_args()
    bsize = 2
    ac = {'features': 'log_mel', 'normalize': False, 'sample_rate': 16000, 'min_duration': 5, 'random_clip_size': 0.96,
          'val_clip_size': 0.96, 'mixup': True, 'mixup_alpha': 0.1, "cropped_read": True, "fmin": 60, "fmax": 7800, "n_fft": 1024,
          "win_len": 400, "hop_len": 160, "tr_feature_size": 251, "val_feature_size": 251, "n_mels": 64}
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
                feature_transforms.SpectrogramPostProcess(mode=mode, normalize=False, log_compress=False),
                feature_transforms.ToMelScale(n_fft=400),
                # transforms.PadToSize(501),
                # transforms.CenterCrop(501)
            ]
        )
    if args.post_batch:
        dset = PackedContrastiveDataset(manifest_path=tr, labels_map=None,
                                        audio_config=audio_config,
                                        mode="contrastive",
                                        pre_feature_transforms=pre_features,
                                        post_feature_transforms=None,
                                        random_cropping_strategy="plain",
                                        gcs_bucket_path=os.environ.get("GCS_BUCKET_PATH", None))
    else:
        dset = PackedContrastiveDataset(manifest_path=tr, labels_map=None,
                                        mode="contrastive",
                                        audio_config=audio_config,
                                        pre_feature_transforms=pre_features,
                                        post_feature_transforms=post_feature,
                                        random_cropping_strategy="plain",
                                        gcs_bucket_path=os.environ.get("GCS_BUCKET_PATH", None))

    loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=8, collate_fn=packed_collate_fn_contrastive)
    lf = len(loader)
    bs_sim_time = bsize * 0.2611
    cnt = 0
    t0 = time.time()

    for batch in tqdm.tqdm(loader):
        anchor, positive, targets = batch
        if args.post_batch:
            anchor = post_feature(anchor)
            positive = post_feature(positive)
        if cnt == 0:
            print(anchor.shape, positive.shape, targets.shape)
            if args.save_batch:
                save_spec_batch(anchor, positive, "./test_batch/contrastive_specs/")
        cnt += 1
    t1 = time.time()
    print("time taken:", t1-t0)
