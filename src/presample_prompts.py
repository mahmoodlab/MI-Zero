import argparse
import numpy as np
import json
import os

parser = argparse.ArgumentParser(description='generate a set of prompts for zeroshot eval')
parser.add_argument('--iters', type=int, default=50, help='num sampling iters')
parser.add_argument('--dataset', type=str, choices=['rcc', 'brca', 'nsclc'], help='name of dataset')
parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite prompt file if already exists')
args = parser.parse_args()

def main(args):
    if args.dataset == 'rcc':
        classnames_pool = {
            'CHRCC': ['chromophobe renal cell carcinoma', 'renal cell carcinoma, chromophobe type', 'renal cell carcinoma of the chromophobe type', 'chromophobe RCC'],  
            'CCRCC': ['clear cell renal cell carcinoma', 'renal cell carcinoma, clear cell type', 'renal cell carcinoma of the clear cell type', 'clear cell RCC'],
            'PRCC': ['papillary renal cell carcinoma', 'renal cell carcinoma, papillary type', 'renal cell carcinoma of the papillary type', 'papillary RCC']
        }
    elif args.dataset == 'brca':
        classnames_pool = {
            'IDC': ['invasive ductal carcinoma', 'carcinoma of the breast, ductal pattern'],
            'ILC': ['invasive lobular carcinoma', 'carcinoma of the breast, lobular pattern']
        }
    elif args.dataset == 'nsclc':
        classnames_pool = {
            'LUAD': ['adenocarcinoma', 'lung adenocarcinoma', 'adenocarcinoma of the lung', 'pulmonary adenocarcinoma', 'adenocarcinoma, lepidic pattern', 'adenocarcinoma, solid pattern', 'adenocarcinoma, micropapillary pattern', 'adenocarcinoma, acinar pattern', 'adenocarcinoma, papillary pattern'],
            'LUSC': ['squamous cell carcinoma', 'lung squamous cell carcinoma', 'squamous cell carcinoma of the lung', 'pulmonary squamous cell carcinoma']
        }
    else:
        raise NotImplementedError

    templates_pool = [
    'CLASSNAME.',
    'a photomicrograph showing CLASSNAME.',
    'a photomicrograph of CLASSNAME.',
    'an image of CLASSNAME.',
    'an image showing CLASSNAME.', 
    'an example of CLASSNAME.',
    'CLASSNAME is shown.',
    'this is CLASSNAME.',
    'there is CLASSNAME.',
    'a histopathological image showing CLASSNAME.',
    'a histopathological image of CLASSNAME.',
    'a histopathological photograph of CLASSNAME.',
    'a histopathological photograph showing CLASSNAME.',
    'shows CLASSNAME.',
    'presence of CLASSNAME.',
    'CLASSNAME is present.'
    ]


    iters = args.iters
    sizerange = range(1, len(templates_pool)+1)


    path_to_prompts = f'./prompts/{args.dataset}_prompts.json' 
    if not args.overwrite and os.path.isfile(path_to_prompts):
        return

    sampled_prompts = {}
    for i in range(iters):
        size = np.random.choice(sizerange)
        classnames_subset = {k: np.random.choice(v, size=1, replace=False)[0] for k, v in classnames_pool.items()}
        template_subset = np.random.choice(templates_pool, size=size, replace=False).tolist()
        sampled_prompts[i] = {
            'classnames': classnames_subset,
            'templates': template_subset
        }

    json.dump(sampled_prompts, open(path_to_prompts, 'w'), indent=4)

if __name__ == '__main__':
    main(args)
    
