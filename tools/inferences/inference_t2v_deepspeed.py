import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))

import torch
import numpy as np
from PIL import Image
from einops import rearrange
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.distributed as dist

from ..modules.config import cfg
from utils.registry_class import INFER_ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN

import time
import tqdm

import deepspeed
import torch.nn as nn
import random

from utils.video_op import save_video

@INFER_ENGINE.register_function()
def inference_t2v_deepspeed(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
            
    seed_all(cfg.inference_seed)
    cfg.rank = cfg.local_rank
    deepspeed_worker_inference(cfg)
    return cfg




def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class ModelWrapper(nn.Module):
    def __init__(self, cfg):
        super(ModelWrapper, self).__init__()
        self.clip_encoder = EMBEDDER.build(cfg.embedder)
        if "motion_encoder" in cfg:
            self.motion_encoder = EMBEDDER.build(cfg.motion_encoder)
        else:
            self.motion_encoder = None
        self.diffusion = DIFFUSION.build(cfg.Diffusion) 
        self.autoencoder = AUTO_ENCODER.build(cfg.auto_encoder)
        self.model = MODEL.build(cfg.UNet)

        self.model = PRETRAIN.build(cfg.Pretrain, unet=self.model)
        
        self.freeze()
        self.cfg = cfg

    def freeze(self):
        self.clip_encoder.eval()
        for param in self.clip_encoder.parameters():
                param.requires_grad = False             
                
        self.autoencoder.eval() # freeze
        for param in self.autoencoder.parameters():
            param.requires_grad = False
            
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.motion_encoder is not None:
            self.motion_encoder.eval()
            for param in self.motion_encoder.parameters():
                param.requires_grad = False
            
def deepspeed_worker_inference(cfg):
    '''
    Training worker for each gpu
    '''
    cfg.world_size = int(os.getenv('WORLD_SIZE', '4'))
    worker_seed = cfg.inference_seed + cfg.rank
    dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)
    torch.cuda.set_device(cfg.rank)
    seed_all(worker_seed)
    
    model_name = cfg.model_name

    cfg.log_dir = osp.join(cfg.log_dir, model_name+"_"+cfg.infer_dataset['type'])

    
    os.makedirs(cfg.log_dir, exist_ok=True)

    infer_dataset = DATASETS.build(cfg.infer_dataset)
    data_sampler = torch.utils.data.distributed.DistributedSampler(infer_dataset, num_replicas= cfg.world_size, shuffle=False)
    eval_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, sampler=data_sampler)
    
    model = ModelWrapper(cfg)

    model_engine = deepspeed.init_inference(
        model=model,
        )

    with torch.no_grad():
        zero_y_negative = model_engine.module.clip_encoder(text=cfg.negative_prompt)
        zero_y_negative = zero_y_negative.detach()
    
    model.eval()
    for batch in tqdm.tqdm(eval_dataloader):
        with torch.no_grad():
            with amp.autocast(enabled=cfg.use_fp16):        
                videoids, captions = batch
                batch_size = len(videoids)
                

                y_words = model_engine.module.clip_encoder(text=captions) # bs * 1 *1024 [B, 1, 1024]


                noise = torch.randn([batch_size, 4, cfg.max_frames, int(cfg.resolution[0]/cfg.scale), int(cfg.resolution[1]/cfg.scale)])
                noise = noise.to(cfg.rank)
                if model_engine.module.motion_encoder is not None:
                    _, temporal_y = model_engine.module.motion_encoder(text=captions)
                    _, temporal_zero_y_negative = model_engine.module.motion_encoder(text=cfg.negative_prompt)
                    model_kwargs=[
                        {'y': y_words, 'temporal_y': temporal_y},
                        {'y': zero_y_negative.repeat(batch_size,1,1), 'temporal_y': temporal_zero_y_negative.repeat(batch_size,1,1)}]
                else:
                    model_kwargs = [{'y': y_words}, {'y': zero_y_negative.repeat(batch_size*16,1,1)}]
                    
                print(f'Start to generate videos')
                video_data = model_engine.module.diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model_engine.module.model,
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)
                print(f'Start to decode videos')
                video_data = 1. / cfg.scale_factor * video_data # [1, 4, 32, 46]
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                chunk_size = min(cfg.decoder_bs, video_data.shape[0])
                video_data_list = torch.chunk(video_data, video_data.shape[0]//chunk_size, dim=0)
                decode_data = []
                for vd_data in video_data_list:
                    gen_frames = model_engine.module.autoencoder.decode(vd_data)
                    decode_data.append(gen_frames)
                video_data = torch.cat(decode_data, dim=0)
                video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b = batch_size)
                print(f'Start to save videos')
                save_video(cfg.log_dir, video_data, name_list=videoids, rank=cfg.rank)
        
        
    