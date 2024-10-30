import os
import json
import torch
import logging
import collections

from utils.registry_class import PRETRAIN
from ..modules.unet import UNetSD_T2V_DEMO

import json



@PRETRAIN.register_function()
def load_model_new(
        unet=None, 
        resume_checkpoint=None,
        from_modelscope=False,
        inference=False,
        key_file="key_file.json",
        **kwargs
    ):

    state_dict = torch.load(resume_checkpoint, map_location='cpu')
    
    if isinstance(unet, UNetSD_T2V_DEMO):
        if from_modelscope:
            dict= json.load(open(key_file))
            exchange = dict['exchange']
            exchange_keys = [e[0] for e in exchange]
            copy = dict['copy']
            copy_keys = [e[0] for e in copy]
            new_state_dict = {}
            for k,v in state_dict.items():
                new_k = k
                for keys in exchange_keys:
                    if k.startswith(keys):
                        index = exchange_keys.index(keys)
                        new_k = k.replace(keys,exchange[index][1])
                        break
                new_state_dict[new_k] = v    
                    
            state_dict = new_state_dict
            new_state_dict = {}
            for k,v in state_dict.items():
                new_k = k
                for keys in copy_keys:
                    if k.startswith(keys):
                        index = copy_keys.index(keys)
                        new_k = k.replace(keys,copy[index][1])
                        if new_k in state_dict.keys():
                            raise ValueError(f"key {new_k} already exists")
                new_state_dict[k] = v
                new_state_dict[new_k] = state_dict[k]

            state_dict = new_state_dict
        mkey, ukey = unet.load_state_dict(state_dict, strict=False)
        print(f'load unet with missing key: {mkey} and unexpected key: {ukey}')
    
    print(f'Successfully load model from {resume_checkpoint}')
    return unet



@PRETRAIN.register_function()
def pretrain_from_sd():
    pass


@PRETRAIN.register_function()
def pretrain_ema_model():
    pass
