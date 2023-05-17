import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import h5py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import sys
import math

from models.ctran import ctranspath
import openslide

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path

# load visual encoder weights and transforms
def load_ctranspath_clip(ckpt_path, img_size = 224, return_trsforms = True):
    def clean_state_dict_clip(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'attn_mask' in k:
                continue
            if 'visual.trunk.' in k:
                new_state_dict[k.replace('module.visual.trunk.', '')] = v
        return new_state_dict
    
    model = ctranspath(img_size = img_size)
    model.head = nn.Identity()
    state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
    state_dict = clean_state_dict_clip(state_dict)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('missing keys: ', missing_keys)
    print('unexpected keys: ', unexpected_keys)
    
    if return_trsforms:
        trsforms = get_transforms_ctranspath(img_size = img_size)
        return model, trsforms
    return model

# ctranspath transformation
def get_transforms_ctranspath(img_size=224, 
                              mean = (0.485, 0.456, 0.406), 
                              std = (0.229, 0.224, 0.225)):
    trnsfrms = transforms.Compose(
                    [
                     transforms.Resize(img_size),
                     transforms.ToTensor(),
                     transforms.Normalize(mean = mean, std = std)
                    ]
                )
    return trnsfrms

def file_exists(file_id, root, ext = '.h5'):
    return os.path.isfile(os.path.join(root, file_id + ext))

def read_assets_from_h5(h5_path):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs

def compute_patch_level(wsi, level0_mag, target_mag = 20, patch_size = 256):
    custom_downsample = int(level0_mag / target_mag)
    if custom_downsample == 1:
        target_level = 0
        target_patch_size = patch_size
    else:
        all_downsamples = wsi.level_downsamples
        all_downsamples = list(np.around(all_downsamples, 1))
        if custom_downsample in all_downsamples:
            target_level = all_downsamples.index(custom_downsample)
            target_patch_size = patch_size
        else:
            target_level = 0
            target_patch_size = int(patch_size * custom_downsample)
    return target_level, target_patch_size

def compute_patch_args(df, wsi_source, wsi_ext = '.svs', target_mag = 20, patch_size = 256):
    wsi_path = os.path.join(wsi_source, df['slide_id'] + wsi_ext)
    wsi = openslide.open_slide(wsi_path)
    df['patch_level'], df['patch_size'] = compute_patch_level(wsi, df['level0_mag'], target_mag, patch_size)
    return df

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self,
        coords,
        wsi,
        patch_level,
        patch_size,
        custom_transforms=None,
        target_patch_size=-1):
        """
        Args:
            coords (string): coordinates to extract patches from w.r.t. level 0.
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            target_patch_size (int): Custom defined image size before embedding
        """
        self.coords = coords
        self.patch_level = patch_level
        self.patch_size = patch_size
        self.wsi = wsi
        self.roi_transforms = custom_transforms
        self.length = len(coords)
        if target_patch_size > 0:
            self.target_patch_size = (target_patch_size, ) * 2
        else:
            self.target_patch_size = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
        if self.target_patch_size is not None:
            img = img.resize(self.target_patch_size)
        img = self.roi_transforms(img)
        return {'img': img, 'coords': coord}
    
@torch.no_grad()
def extract_features(df, model, trsforms, save_dir = '', wsi_ext = '.svs', 
                     target_patch_size = 256, 
                     dataloader_args = {'batch_size': 128, 'num_workers': 8},
                     device = 'cuda:0'):
    model.to(device)
    model.eval()
    slide_id = df['slide_id']
    wsi_path = os.path.join(wsi_source, slide_id + wsi_ext)
    wsi = openslide.open_slide(wsi_path)
    patch_level = df['patch_level']
    patch_size = df['patch_size']
    h5_path = os.path.join(h5_source, slide_id + '.h5')
    assets, _ = read_assets_from_h5(h5_path)
    coords = assets['coords']
    print(f'slide_id: {slide_id}, n_patches: {len(coords)}')
    save_path = os.path.join(save_dir, 'h5_files', df['slide_id'] + '.h5')
    if os.path.isfile(save_path):
        print(f'{slide_id} already exists')
        return
    dataset = Whole_Slide_Bag_FP(coords,
                                 wsi,
                                 patch_level,
                                 patch_size,
                                 custom_transforms = trsforms,
                                 target_patch_size = target_patch_size)
    dataloader = DataLoader(dataset, shuffle=False, **dataloader_args)
    mode = 'w'
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch['img'].to(device)
            coords = np.array(batch['coords'])
            features = model(imgs)
            features = features.cpu().numpy()
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(save_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'
            
    pt_save_path = os.path.join(save_dir, 'pt_files', df['slide_id'] + '.pt')
    assets, _ = read_assets_from_h5(save_path)
    features = torch.from_numpy(assets['features'])
    print('features: ', features.size())
    torch.save(features, pt_save_path)

import argparse
parser = argparse.ArgumentParser(description='Extract features using patch coordinates')
parser.add_argument('--csv_path', type=str, help='path to slide csv')
parser.add_argument('--h5_source', type=str, help='path to dir containing patch h5s')
parser.add_argument('--wsi_source', type=str, help='path to dir containing wsis')
parser.add_argument('--save_dir', type=str, help='path to save extracted features')
parser.add_argument('--wsi_ext', type=str, default='.svs', help='extension for wsi')
parser.add_argument('--ckpt_path', type=str, help='path to clip ckpt')
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda:n')
args = parser.parse_args()

if __name__ == '__main__':
    h5_source = args.h5_source
    wsi_source = args.wsi_source
    ckpt_path = args.ckpt_path
    feat_save_dir = args.save_dir
    device = args.device

    print(feat_save_dir)
    os.makedirs(feat_save_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_save_dir, 'h5_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_save_dir, 'pt_files'), exist_ok=True)
    
    df = pd.read_csv(args.csv_path)
    assert 'level0_mag' in df.columns, 'level0_mag column missing'

    df['has_h5'] = df['slide_id'].apply(lambda x: file_exists(x, h5_source))
    df['has_h5'].value_counts()
    df['has_slide'] = df['slide_id'].apply(lambda x: file_exists(x, wsi_source, ext='.svs'))
    df = df[df['has_slide']]
    assert df['has_h5'].sum() == len(df['has_h5'])
    assert df['has_slide'].sum() == len(df['has_slide'])

    df = df.apply(lambda x: compute_patch_args(x, wsi_source, wsi_ext='.svs', target_mag = 20, patch_size = 256), axis=1)
    model, trsforms = load_ctranspath_clip(ckpt_path=ckpt_path, 
                                 img_size = 224, 
                                 return_trsforms = True)
    df.apply(lambda x: extract_features(x, save_dir=feat_save_dir, 
                                        wsi_ext=args.wsi_ext, 
                                        model=model, 
                                        trsforms=trsforms, 
                                        device=device), axis=1)
