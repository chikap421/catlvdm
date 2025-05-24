import os
import os.path as osp
import sys
import uuid
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
import tqdm
from einops import rearrange
import torch.cuda.amp as amp
from datetime import datetime

sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))

from ..modules.config import cfg
from utils.registry_class import INFER_ENGINE, MODEL, DATASETS, EMBEDDER, AUTO_ENCODER, DISTRIBUTION, VISUAL, DIFFUSION, PRETRAIN
from utils.video_op import save_video

@INFER_ENGINE.register_function()
def inference_t2v_deepspeed(cfg_update, **kwargs):
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
        for p in self.clip_encoder.parameters():
            p.requires_grad = False

        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if self.motion_encoder is not None:
            self.motion_encoder.eval()
            for p in self.motion_encoder.parameters():
                p.requires_grad = False

# Uncomment to Use for webvid dataset
def deepspeed_worker_inference(cfg):
    """
    Worker function for T2V inference using deepspeed.
    """
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Initialize distributed
    cfg.world_size = int(os.getenv('WORLD_SIZE', '4'))
    worker_seed = cfg.inference_seed + cfg.rank
    dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)
    torch.cuda.set_device(cfg.rank)
    seed_all(worker_seed)

    # Build dataset
    infer_dataset = DATASETS.build(cfg.infer_dataset)
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        infer_dataset,
        num_replicas=cfg.world_size,
        shuffle=False
    )
    eval_dataloader = torch.utils.data.DataLoader(
        infer_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        sampler=data_sampler
    )

    # Initialize model & engine
    model = ModelWrapper(cfg)
    model_engine = deepspeed.init_inference(model=model)

    with torch.no_grad():
        negative_feat = model_engine.module.clip_encoder(text=cfg.negative_prompt).detach()

    model.eval()

    for batch in tqdm.tqdm(eval_dataloader):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=cfg.use_fp16):
                # Unpack from dataset => (videoid, page_dir, prompt) or (videoid, prompt)
                if len(batch) == 3:
                    videoids, page_dirs, prompts = batch
                elif len(batch) == 2:
                    videoids, prompts = batch
                    page_dirs = [None] * len(videoids)
                else:
                    raise ValueError(f"Unexpected batch size {len(batch)} in inference.")
                batch_size = len(videoids)

                # Encode prompts
                y_words = model_engine.module.clip_encoder(text=prompts)

                # Generate noise
                noise = torch.randn([
                    batch_size, 4, cfg.max_frames,
                    int(cfg.resolution[0] / cfg.scale),
                    int(cfg.resolution[1] / cfg.scale)
                ], device=cfg.rank)

                # Build model_kwargs
                if model_engine.module.motion_encoder is not None:
                    _, temporal_y = model_engine.module.motion_encoder(text=prompts)
                    _, temporal_zero_neg = model_engine.module.motion_encoder(text=cfg.negative_prompt)
                    model_kwargs = [
                        {'y': y_words, 'temporal_y': temporal_y},
                        {'y': negative_feat.repeat(batch_size, 1, 1), 'temporal_y': temporal_zero_neg.repeat(batch_size, 1, 1)}
                    ]
                else:
                    model_kwargs = [
                        {'y': y_words},
                        {'y': negative_feat.repeat(batch_size, 1, 1)}
                    ]

                print("Start to generate videos")
                video_data = model_engine.module.diffusion.ddim_sample_loop(
                    noise=noise,
                    model=model_engine.module.model,
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0
                )

                print("Start to decode videos")
                video_data = video_data / cfg.scale_factor
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                chunk_size = min(cfg.decoder_bs, video_data.size(0))
                chunks = torch.chunk(video_data, video_data.size(0) // chunk_size, dim=0)

                decode_data = []
                for c_data in chunks:
                    frames_dec = model_engine.module.autoencoder.decode(c_data)
                    decode_data.append(frames_dec)

                video_data = torch.cat(decode_data, dim=0)
                video_data = rearrange(video_data, '(b f) c h w -> b c f h w', b=batch_size)

                print("Start to save videos")
                name_list = [
                    f"{page_dirs[i]}/{videoids[i]}.mp4" if page_dirs[i] else f"{videoids[i]}.mp4"
                    for i in range(batch_size)
                ]

                save_video(
                    local_path=cfg.log_dir,
                    gen_video=video_data,
                    name_list=name_list,
                    rank=cfg.rank,
                    dataset_type=cfg.infer_dataset["type"]
                )
