""" timm model adapter

Wraps timm (https://github.com/rwightman/pytorch-image-models) models for use as a vision tower in CLIP model.
"""
from collections import OrderedDict

import torch.nn as nn

import timm_ctp as timm
from timm_ctp.models.layers import Mlp, to_2tuple
from .utils import freeze_batch_norm_2d

class TimmModel(nn.Module):
    def __init__(
            self,
            trunk = None,
            model_name = None,
            embed_dim = -1,
            image_size=512,
            pool='avg',
            proj='linear',
            drop=0.,
            pretrained=False):
        super().__init__()

        self.image_size = to_2tuple(image_size)
        if trunk is not None:
            self.trunk = trunk
        elif model_name is not None:
            self.trunk = timm.create_model(model_name, img_size=image_size, pretrained=pretrained)
        else:
            raise NotImplementedError

        # reset global pool if pool config set, otherwise leave as network default
        reset_kwargs = dict(global_pool=pool) if pool else {}
        self.trunk.reset_classifier(0, **reset_kwargs)
        prev_chs = self.trunk.num_features

        head_layers = OrderedDict()
        if proj == 'linear':
            head_layers['drop'] = nn.Dropout(drop)
            head_layers['proj'] = nn.Linear(prev_chs, embed_dim)
        elif proj == 'mlp':
            head_layers['mlp'] = Mlp(prev_chs, 2 * embed_dim, embed_dim, drop=drop)

        self.head = nn.Sequential(head_layers)


    def lock(self, freeze_bn_stats=False):
        for param in self.trunk.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self.trunk)
        

    def forward(self, x):
        x = self.trunk(x)
        x = self.head(x)
        return x
