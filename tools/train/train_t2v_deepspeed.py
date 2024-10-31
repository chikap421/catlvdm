import os
import os.path as osp
import sys
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))

import torch
import logging
import datetime
import numpy as np
from einops import rearrange
import torch.cuda.amp as amp
from importlib import reload

import torch.distributed as dist

from ..modules.config import cfg
from utils.seed import setup_seed
from utils.distributed import generalized_all_gather, all_reduce
from utils.registry_class import ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN
from torch.nn import functional as F

import time
from ..modules.unet import TemporalTransformer4Cross
import tqdm
from datetime import datetime
import deepspeed
from deepspeed.runtime.utils import memory_status
import deepspeed.comm as comm
import torch.nn as nn

from .loss_utils import eot_loss_with_video

from deepspeed.monitor.monitor import MonitorMaster
from deepspeed.runtime.config import DeepSpeedConfig
from torchvision.models.optical_flow import raft_small

@ENGINE.register_function()
def train_t2v_deepspeed(cfg_update,  **kwargs):
    for k, v in cfg_update.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
            
    setup_seed(cfg.seed)
    cfg.rank = cfg.local_rank
    deepspeed_worker_wrapper(cfg)
    return cfg



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
        self.model  = PRETRAIN.build(cfg.Pretrain, unet=self.model)
        self.clip_visual = EMBEDDER.build(cfg.clip_visual)
        self.resume_step = cfg.resume_step
        self.freeze()
        self.raft_model = raft_small(weights='DEFAULT')
        self.cfg = cfg

    def freeze(self):
        self.clip_encoder.eval()
        for param in self.clip_encoder.parameters():
                param.requires_grad = False
                
        self.autoencoder.eval() # freeze
        for param in self.autoencoder.parameters():
            param.requires_grad = False
    
    def configure_parameters(self, freeze=True):
        params_list =[]
        if not freeze:
            print("allow unet to be updated")
            for param in self.model.parameters():
               param.requires_grad = True
            params_list.extend(self.model.parameters())
        else:
            for module in self.model.modules():
                if isinstance(module, TemporalTransformer4Cross):
                    for param in module.parameters():
                        param.requires_grad = True
                    params_list.append({'params': module.parameters()})
                    # params_list.extend(module.parameters())
        
        
        if self.motion_encoder:
            for param in self.motion_encoder.parameters():
                param.requires_grad = True
            params_list.append({'params': self.motion_encoder.parameters(),'lr':cfg.motion_lr})
            
        return params_list

    
    
    
    
    def forward(self, batch, zero_y_negative):
        videos, captions = batch
        batch_size, frames_num, _, _, _ = videos.shape

        encoded_chunks = []
        with amp.autocast(enabled=True):
            with torch.no_grad():
                for chunk in torch.chunk(videos, chunks=self.cfg.chunk_size, dim=0):
                    chunk = rearrange(chunk, 'b f c h w -> (b f) c h w')
                    chunk = chunk.to(self.cfg.rank)
                    latent_z = self.autoencoder.encode_first_stage(chunk, self.cfg.scale_factor).detach()
                    encoded_chunks.append(latent_z)

            video_data = torch.cat(encoded_chunks, dim=0)
            video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b=batch_size) 

            opti_timesteps = getattr(cfg, 'opti_timesteps', cfg.Diffusion.schedule_param.num_timesteps)
            t_round = torch.randint(0, opti_timesteps, (batch_size, ),device=self.cfg.rank) # why torch.long
            
            # preprocess
            with torch.no_grad():
                y_words = self.clip_encoder(text=captions)
            
            if self.motion_encoder:
                tokenids,temporal_y_words = self.motion_encoder(text=captions)
                sample_idx, eot_idx = (tokenids == 49407).nonzero(as_tuple=True)

            if self.clip_visual:
                medium_frames = videos[:, 16//2 , :, :, :].squeeze(1)
                image_embeddings = self.clip_visual(medium_frames)
                eot_tokens = temporal_y_words[sample_idx, eot_idx, :]
                regularization_loss = -F.cosine_similarity(eot_tokens, image_embeddings, dim=-1)            

            y_words[torch.rand(y_words.size(0)) < cfg.p_zero, :] = zero_y_negative
            
            model_kwargs = {'y': y_words, 'temporal_y': temporal_y_words, 'eot_idx': eot_idx}
            
            diffusion_loss,video_motion_loss, attention_store = self.diffusion.loss(
                    x0=video_data, 
                    t=t_round, 
                    model=self.model, 
                    model_kwargs=model_kwargs, 
                    use_div_loss=cfg.use_div_loss,
                    )
            
            text_motion_loss = eot_loss_with_video(self.raft_model, videos.to(self.cfg.rank),attention_store)
            
        return diffusion_loss, video_motion_loss, text_motion_loss, regularization_loss    


         


def deepspeed_worker_wrapper(cfg):
    '''
    Training worker for each gpu
    '''
    torch.backends.cudnn.benchmark = True

    # [Log] Save logging
    time_str = datetime.now().strftime('%m-%d_%H:%M')
    log_dir = generalized_all_gather(cfg.log_dir)[0]
    exp_name = osp.basename(cfg.cfg_file).split('.')[0]
    cfg.log_dir = osp.join(cfg.log_dir, exp_name+time_str)
    os.makedirs(cfg.log_dir, exist_ok=True)
    if cfg.rank == 0:
        log_file = osp.join(cfg.log_dir, 'log.txt')
        cfg.log_file = log_file
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)
        logging.info(f'Save all the file in to dir {cfg.log_dir}')


   
    train_dataset = DATASETS.build(cfg.train_dataset)  
    model = ModelWrapper(cfg)
    params_list = model.configure_parameters(freeze=cfg.freeze)
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        args=cfg,
        model=model,
        model_parameters=params_list,
        training_data=train_dataset,
        )
    
    monitor = model_engine.monitor

    cfg.world_size = dist.get_world_size()

    resume_step = model_engine.module.resume_step
    with torch.no_grad():
        zero_y_negative = model_engine.module.clip_encoder(text=cfg.negative_prompt)
        zero_y_negative = zero_y_negative.detach()
    model_engine.module.model.zero_y = zero_y_negative
    train_rank_iter = iter(train_dataloader)
    model.train()
    for step in tqdm.tqdm(range(resume_step, resume_step + cfg.num_steps)):    
        try:
            batch = next(train_rank_iter)
        except StopIteration:
            train_rank_iter = iter(train_dataloader)
            batch = next(train_rank_iter)

        diffusion_loss, video_motion_loss, text_motion_loss, regularization_loss = model_engine(batch, zero_y_negative)
        # rank = dist.get_rank()
        events = [("Train/Samples/diffusion_loss", diffusion_loss.mean().item(),model_engine.global_samples),
                  ("Train/Samples/video_motion_loss", video_motion_loss.mean().item(),model_engine.global_samples),
                  ("Train/Samples/text_motion_loss", text_motion_loss.mean().item(),model_engine.global_samples),
                  ("Train/Samples/regularization_loss", regularization_loss.mean().item(),model_engine.global_samples),
                  ]
        loss = diffusion_loss + 0.1 * video_motion_loss + 0.1 * text_motion_loss + 0.3 * regularization_loss
        loss = loss.mean()
        monitor.write_events(events)
        model_engine.backward(loss)
        model_engine.step()


        if cfg.rank == 0 and step % cfg.log_interval == 0: 
            memory_status(f"Memory Usage Summary Step: {step}")

        if step == cfg.num_steps or step % cfg.save_ckp_interval == 0:
            os.makedirs(osp.join(cfg.log_dir, 'checkpoints'), exist_ok=True)
            if cfg.rank == 0:
                local_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}.pth')
                logging.info(f'Begin to Save model to {local_model_path}')
                save_dict = {
                    'unet_state_dict': model_engine.module.model.state_dict(),
                    'step': step}
                torch.save(save_dict, local_model_path)
                print(f'Save model to {local_model_path}')
                if "motion_encoder" in cfg:
                    temporal_model_path = osp.join(cfg.log_dir, f'checkpoints/non_ema_{step:08d}_motion_encoder.pth')
                    temporal_state_dict = model_engine.module.motion_encoder.model.state_dict()
                    torch.save(temporal_state_dict,temporal_model_path)
                    print(f'Save temporal model to {temporal_model_path}')
                    
                
        
