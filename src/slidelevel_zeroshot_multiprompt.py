import argparse
import os
import logging
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import pickle
import random

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score, classification_report
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('once')

# Custom dependencies
from models.factory import create_model
from zeroshot_utils.zeroshot_path import zero_shot_classifier

label_dicts = {
    'NSCLC_subtyping': {'LUAD': 0, 'LUSC': 1},
    'BRCA_subtyping': {'IDC': 0, 'ILC': 1},
    'RCC_subtyping': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2}
}

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def parse_args():
    parser = argparse.ArgumentParser(description='Slidelevel Zeroshot')
    parser.add_argument('--task', type=str, choices = ['NSCLC_subtyping', 'BRCA_subtyping', 'RCC_subtyping', 'C16'], help='name of dataset')
    parser.add_argument('--embeddings_dir', type=str, default='/ssd/CVPR2023/tcga_rcc_bioclinicalbert_embeddings', help='name of embeddings')
    parser.add_argument('--dataset_split', type=str, help='path to test splits, if no splits, path to test set reference csv')
    parser.add_argument('--model_checkpoint', type=str, default='./logs/ctranspath_448_bioclinicalbert/checkpoints/epoch_50.pt', help='name of ckpt')
    parser.add_argument('--topj', type=int, nargs="+", help='integers for topj pooling')
    parser.add_argument('--avg_pool', type=int, default=0, help='simply mean pool')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for dataloading')
    parser.add_argument('--num_folds',  default=1, type=int, help='number of folds')
    parser.add_argument('--seed',  default=1, type=int, help='number of folds')
    parser.add_argument('--ss',  default=0, type=int, help='enable spatial smoothing')
    parser.add_argument('--ss_k',  default=8, type=int, help='k for knn spatial smoothing')
    parser.add_argument('--index_cache_dir', type=str, default=None)
    parser.add_argument('--save_dir', default='slidelevel_zeroshot_results', type=str, help='dir for saving results')
    parser.add_argument('--prompt_file', type=str, help='path to prompt file')
    parser.add_argument('--device', type=str, default='cuda:0', help='device cuda:n')
    
    # python slidelevel_zeroshot_multiprompt.py --task RCC_subtyping --dataset_root /ssd/CVPR2023/ --dataset_split ./data_csvs/tcga_rcc_zeroshot_example.csv --topj 1 5 50 --prompt_file ./prompts/rcc_prompts.json

    args = parser.parse_args()
    return args

def clean_state_dict_ctranspath(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'attn_mask' in k:
            continue
        new_state_dict[k.replace('module.', '')] = v
    return new_state_dict

class WSI_Classification_Dataset(Dataset):
    def __init__(self, 
                 df, 
                 data_source, 
                 target_transform = None,
                 index_col = 'slide_id',
                 target_col = 'OncoTreeCode', 
                 use_h5 = True,
                 label_map = None):
        """
        Args:
        """
        self.label_map = label_map
        self.data_source = data_source
        self.index_col = index_col
        self.target_col = target_col
        self.target_transform = target_transform
        self.data = df
        self.use_h5 = use_h5

    def __len__(self):
        return len(self.data)

    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]

    def __getitem__(self, idx):
        
        slide_id = self.get_ids(idx)
        label = self.get_labels(idx)

        if self.label_map is not None:
            label = self.label_map[label]
        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.use_h5:
            feat_path = os.path.join(self.data_source, 'h5_files', slide_id + '.h5')
            with h5py.File(feat_path, 'r') as f:
                features = torch.from_numpy(f['features'][:])
                coords = torch.from_numpy(f['coords'][:])
        else:
            feat_path = os.path.join(self.data_source, 'pt_files', slide_id + '.pt')
            features = torch.load(feat_path)
            coords = []
        
        return {'features': features, 'coords': coords, 'label': label}

class AverageMeter(object):   # Copied from core_utils.py
    """Computes and stores the average and current value"""
    def __init__(self, name = 'unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def perm_gpu_f32(pop_size, num_samples, device):
    """Use torch.randperm to generate indices on a 32-bit GPU tensor."""
    return torch.randperm(pop_size, dtype=torch.int32, device=device)[:num_samples]

def spatial_smoothing(logits, index_cache_file):
    I = torch.load(index_cache_file, map_location = logits.device)
    logits = logits[I].mean(dim=1)
    return logits

def topj_pooling(logits, topj):
    """
    logits: N x 1 logit for each patch
    coords: N x 2 coordinates for each patch
    topj: tuple of the top number of patches to use for pooling
    ss: spatial smoothing by k-nn
    ss_k: k in k-nn for spatial smoothing
    """
    # Sums logits across topj patches for each class, to get class prediction for each topj
    maxj = min(max(topj), logits.size(0)) # Ensures j is smaller than number of patches. Unlikely for number of patches to be < 10, but just in case
    values, _ = logits.topk(maxj, 0, True, True) # maxj x C
    preds = {j : values[:min(j, maxj)].sum(dim=0, keepdim=True) for j in topj} # dict of 1 x C logit scores
    preds = {key: val.argmax(dim=1) for key,val in preds.items()} # dict of predicted class indices
    return preds

@torch.no_grad()
def run(model, classifier, dataloader, device, topj, ss = 0, ss_k = 8, avg_pool = False, index_cache_dir = None):                                             # Copied from zeroshot_path, moderate changes
    accs = {}   # Initialize dict of accuracy for each j
    meters = {j: AverageMeter() for j in topj}

    # meters[-1] =  AverageMeter()
    logits_all, targets, coords_all, preds_all = [], [], [], []
    for idx, data in enumerate(dataloader): # batch size is always 1 WSI, 
        image_features = data['features'].to(device, non_blocking=True).squeeze(0)
        target = data['label'].to(device, non_blocking=True)
        coords = data['coords']
        if not isinstance(coords, list):
            coords = coords.squeeze(0).numpy()
        slide_id = dataloader.dataset.get_ids(idx)
        if index_cache_dir is not None:
            index_cache_file = os.path.join(index_cache_dir, slide_id + '.pt')
        else:
            index_cache_file = None
        coords_all.append(coords)

        # Get similarity logits
        image_features = model.visual.head(image_features) # Project to desired dimensions num_patches x 512
        image_features = F.normalize(image_features, dim=-1) 
        logits = image_features @ classifier

        if ss:
            logits = spatial_smoothing(logits, coords, ss_k = ss_k, index_cache_file = index_cache_file)
        
        logits_all.append(logits.cpu().numpy())
        targets.append(target.cpu().numpy())

        if avg_pool:
            preds = {-1: logits.mean(dim=0, keepdim=True).argmax(dim=1)}
        else:
            preds = topj_pooling(logits, topj = topj)

        results = {key: (val == target).float().item() for key, val in preds.items()}
        preds_all.append(preds)

        for j in topj:
            meters[j].update(results[j], n=1) # Update AverageMeters with new results

    # Save raw preds & targets
    targets = np.concatenate(targets, axis=0)
    # Computed balanced accuracy
    preds_all = {key: np.concatenate([i[key].cpu() for i in preds_all], axis=0) for key in preds_all[0].keys()}
    bacc = {key: balanced_accuracy_score(targets, val) for key, val in preds_all.items()}
    cls_rep = {key: classification_report(targets, val, output_dict=True, zero_division=0) for key, val in preds_all.items()}

    # Get final accuracy across all images
    accs = {j: meters[j].avg for j in topj}
 
    return accs, bacc, cls_rep

def zero_shot_eval(model, tokenizer, dataloader, prompts, model_name, topj = (1,5,10), device = 'cuda', 
                    ss = False, ss_k = 8, index_cache_dir = None,
                    avg_pool = False):
    classnames = prompts['classnames']
    templates = prompts['templates']

    idx_to_class = {v:k for k,v in dataloader.dataset.label_map.items()}
    n_classes = len(idx_to_class)
    classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]
    
    classifier = zero_shot_classifier(model, tokenizer, classnames_text, templates, device) # num_classes x 512
    accs, bacc, cls_rep = run(model, classifier, dataloader, device, topj, avg_pool = avg_pool,
                              ss = ss, ss_k = ss_k, index_cache_dir = index_cache_dir)

    return accs, bacc, cls_rep

def save_pkl(filename, save_object):
    writer = open(filename,'wb')
    pickle.dump(save_object, writer)
    writer.close()

def load_pretrained_tokenizer(model_name):
    if 'clinicalbert' in model_name:
        model_name = 'emilyalsentzer/Bio_ClinicalBERT'
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    elif 'pubmed' in model_name:
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
    else:
        raise NotImplementedError
    
    return tokenizer

def main():
    args = parse_args()

    random_seed(args.seed, 0)
    os.environ['TOKENIZERS_PARALLELISM'] = "true"
    logging.getLogger().setLevel(logging.INFO)

    if args.avg_pool > 0 and args.ss > 0:
        args.mode = 'ssmp'
    elif args.avg_pool > 0:
        args.mode = 'mp'
    elif args.ss > 0:
        args.mode = 'ss'
    else:
        args.mode = 'none'
    
    if os.path.isfile(args.dataset_split): # if only a reference file is provided, check that no folds are intended
        assert args.num_folds == 1

    args.model_name = args.model_checkpoint.split('/')[-3]
    save_path = os.path.join(args.save_dir, f'{args.model_name}_{args.task}_{args.mode}_results.json')
    if os.path.isfile(save_path): # skip already computed
        return 0

    os.makedirs(args.save_dir, exist_ok=True)

    model_config = args.model_name
    model = create_model(model_config, device=args.device, override_image_size=None)
    if args.model_checkpoint is not None: # load PPTCLIP checkpoint if applicable
        if os.path.exists(args.model_checkpoint):
            state_dict = torch.load(args.model_checkpoint, map_location='cpu')['state_dict']
            state_dict = clean_state_dict_ctranspath(state_dict)
            missing_keys, _ = model.load_state_dict(state_dict, strict=False)
            assert pd.Series(missing_keys).str.contains('attn_mask').all() # only modules with attn_mask are not loaded
            logging.info(f'Checkpoint {args.model_checkpoint} loaded successfully')
        else:
            logging.error(f'Cannot find model checkpoint {args.model_checkpoint}')
            return 1
    
    model.eval()
    with open(args.prompt_file, 'r') as pf: 
        prompts = json.load(pf)
    num_prompts = len(prompts)

    all_results = {}

    for prompt_idx in (pbar := tqdm(range(num_prompts))):
        pbar.set_description(f'mode {args.mode}')
        prompt = prompts[str(prompt_idx)]
        # Load tokenizer
        tokenizer = load_pretrained_tokenizer(args.model_name)
        
        # Set up dataloader
        # Run each test split
        all_accs = {}
        for fold_idx in range(args.num_folds):
            if os.path.isfile(args.dataset_split):
                csv_path = args.dataset_split
            else:
                csv_path = os.path.join(os.path.join(args.dataset_split, f'test_{fold_idx}.csv'))
            
            df = pd.read_csv(csv_path) # Load split csv
            df.reset_index(drop=True, inplace=True)

            dataset = WSI_Classification_Dataset(
                    df = df,
                    data_source = args.embeddings_dir, 
                    target_transform = None,
                    index_col = 'slide_id',
                    target_col = 'OncoTreeCode', 
                    use_h5 = bool(args.ss),
                    label_map = label_dicts[args.task]
                )
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
            # logits: list of N_1 x n_classes scores for N_1, ..., N_num_slides
            # coords: same shape as logits
            # target: num_slides x 1

            # Eval zeroshot
            if args.avg_pool > 0:
                args.topj = (-1,)
            accs, bacc, cls_rep = zero_shot_eval(
                model, tokenizer, dataloader, prompt, args.model_name, args.topj, 
                ss = args.ss > 0, ss_k = args.ss_k, 
                index_cache_dir = args.index_cache_dir,
                avg_pool = args.avg_pool > 0,
                device=args.device
            )
            all_accs[f'fold_{fold_idx}'] = accs
            all_accs[f'fold_{fold_idx}_bacc'] = bacc
            all_accs[f'fold_{fold_idx}_cls_rep'] = cls_rep
            
            logging.info(f'Model achieved the following results for zero shot evaluation on split {fold_idx}:')
            for j in args.topj:
                logging.info(f'top{j} pooling: {bacc[j]:.3f}')
        
        avgs = {}
        stds = {}
        baccs = {}
        baccs_std = {}
        for j in args.topj:
            avgs[j] = np.array([all_accs[f'fold_{fold_idx}'][j] for fold_idx in range(args.num_folds)]).mean()
            stds[j] = np.array([all_accs[f'fold_{fold_idx}'][j] for fold_idx in range(args.num_folds)]).std()
            baccs[j] = np.array([all_accs[f'fold_{fold_idx}_bacc'][j] for fold_idx in range(args.num_folds)]).mean()
            baccs_std[j] = np.array([all_accs[f'fold_{fold_idx}_bacc'][j] for fold_idx in range(args.num_folds)]).std()
        
        all_accs['avg'] = avgs
        all_accs['std'] = stds
        all_accs['bacc_avg'] = baccs
        all_accs['bacc_std'] = baccs_std

        all_results[prompt_idx] = all_accs

    median_baccs = {key: np.median(np.array([all_results[prompt_idx]['bacc_avg'][key] for prompt_idx in range(num_prompts)])) for key in args.topj}
    all_results['median_baccs'] = median_baccs

    with open(save_path, 'w') as f:
        f.write(json.dumps(all_results, indent=4))

if __name__ == '__main__':
    main()
