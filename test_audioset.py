import os
import tqdm
import time
from audio_utils.packed_datasets import PackedDataset, packed_collate_fn_multiclass, packed_collate_fn_multilabel
from audio_utils.common.audio_config import AudioConfig
from torch.utils.data import DataLoader

if __name__ == '__main__':
    bsize = 2
    ac = {'features': 'log_mel', 'normalize': False, 'sample_rate': 16000, 'min_duration': 5, 'random_clip_size': 5,
          'val_clip_size': 5, 'mixup': True, 'mixup_alpha': 0.1, "cropped_read": True, "fmin": 60, "fmax": 7800, "n_fft": 1024,
          "win_len": 400, "hop_len": 160, "tr_feature_size": 501, "val_feature_size": 501, "n_mels": 64}
    audio_config = AudioConfig()
    audio_config.parse_from_config(ac)
    print(audio_config.features)
    tr = "/home/sarthak/my_disk/Datasets/audioset_16kHz/gcs_meta/balanced_train.csv"
    lbl_map = "/home/sarthak/my_disk/Datasets/audioset_16kHz/gcs_meta/lbl_map.json"
    dset = PackedDataset(manifest_path=tr, labels_map=lbl_map, audio_config=audio_config,
                         mode="multilabel", labels_delimiter=";", gcs_bucket_path=os.environ.get("GCS_BUCKET_PATH",None))
    loader = DataLoader(dset, batch_size=bsize, shuffle=True, num_workers=8, collate_fn=packed_collate_fn_multilabel)
    lf = len(loader)
    bs_sim_time = bsize * 0.2611
    cnt = 0
    t0 = time.time()

    for batch in tqdm.tqdm(loader):
        x, y = batch
        if cnt == 0:
           print(x.shape)
           print(y.shape)
        cnt += 1
        # simulate leaf 64 batchsize time
        #time.sleep(bs_sim_time)
    t1 = time.time()
    print("time taken:", t1-t0)
    #print("effective data time:", (t1-t0) - (bs_sim_time*lf))
