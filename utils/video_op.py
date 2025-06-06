import os
import os.path as osp
import sys
import cv2
import glob
import math
import torch
import gzip
import copy
import time
import json
import pickle
import base64
import imageio
import hashlib
import requests
import binascii
import zipfile
import subprocess
# import skvideo.io
import numpy as np
from io import BytesIO
import urllib.request
import torch.nn.functional as F
import torchvision.utils as tvutils
from multiprocessing.pool import ThreadPool as Pool
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont


def gen_text_image(captions, text_size):
    num_char = int(38 * (text_size / text_size))
    font_size = int(text_size / 20)
    font = ImageFont.truetype('data/font/DejaVuSans.ttf', size=font_size)
    text_image_list = []
    for text in captions:
        txt_img = Image.new("RGB", (text_size, text_size), color="white") 
        draw = ImageDraw.Draw(txt_img)
        lines = "\n".join(text[start:start + num_char] for start in range(0, len(text), num_char))
        draw.text((0, 0), lines, fill="black", font=font)
        txt_img = np.array(txt_img)
        text_image_list.append(txt_img)
    text_images = np.stack(text_image_list, axis=0)
    text_images = torch.from_numpy(text_images)
    return text_images

@torch.no_grad()
def save_video_refimg_and_text(
    local_path,
    ref_frame,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    nrow=4, 
    save_fps=8,
    retry=5):
    ''' 
    gen_video: BxCxFxHxW
    '''
    nrow = max(int(gen_video.size(0) / 2), 1)
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 8x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 8x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 8x16x256x256x3

    ref_frame = ref_frame.unsqueeze(2)
    ref_frame = ref_frame.mul_(vid_std).add_(vid_mean)
    ref_frame = ref_frame.repeat_interleave(repeats=gen_video.size(2), dim=2) # 8x16x256x256x3
    ref_frame.clamp_(0, 1)
    ref_frame = ref_frame * 255.0
    ref_frame = rearrange(ref_frame, 'b c f h w -> b f h w c')
    
    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([ref_frame, images, text_images], dim=3)

    images = rearrange(images, '(r j) f h w c -> f (r h) (j w) c', r=nrow)
    images = [(img.numpy()).astype('uint8') for img in images]

    for _ in [None] * retry:
        try:
            if len(images) == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                local_path = local_path + '.mp4'
                frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                for fid, frame in enumerate(images):
                    tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                    cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                os.system(cmd); os.system(f'rm -rf {frame_dir}')
                os.system(f'rm -rf {local_path}')
                
                
                # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # writer = cv2.VideoWriter(local_path, fourcc, save_fps, (images[0].shape[1], images[0].shape[0]))
                # for frame in images:
                #     writer.write(frame[:,:,::-1])
                # writer.release()
            exception = None
            break
        except Exception as e:
            exception = e
            continue


@torch.no_grad()
def save_i2vgen_video(
    local_path,
    image_id,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    ''' 
    Save both the generated video and the input conditions.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    text_images = gen_text_image(captions, text_size) # Tensor 1x256x256x3
    text_images = text_images.unsqueeze(1) # Tensor 1x1x256x256x3
    text_images = text_images.repeat_interleave(repeats=gen_video.size(2), dim=1) # 1x16x256x256x3

    image_id = image_id.unsqueeze(2) # B, C, F, H, W
    image_id = image_id.repeat_interleave(repeats=gen_video.size(2), dim=2) # 1x3x32x256x448
    image_id = image_id.mul_(vid_std).add_(vid_mean)  # 32x3x256x448
    image_id.clamp_(0, 1)
    image_id = image_id * 255.0
    image_id = rearrange(image_id, 'b c f h w -> b f h w c')

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = torch.cat([image_id, images, text_images], dim=3)
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]

    exception = None
    for _ in [None] * retry:
        try:
            frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
            os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
            for fid, frame in enumerate(images):
                tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
            os.system(cmd); os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception


@torch.no_grad()
def save_i2vgen_video_safe(
    local_path,
    gen_video, 
    captions, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps = 8
):
    '''
    Save only the generated video, do not save the related reference conditions, and at the same time perform anomaly detection on the last frame.
    '''
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1) #ncfhw

    gen_video = gen_video.mul_(vid_std).add_(vid_mean)  # 8x3x16x256x384
    gen_video.clamp_(0, 1)
    gen_video = gen_video * 255.0

    images = rearrange(gen_video, 'b c f h w -> b f h w c')
    images = images[0]
    images = [(img.numpy()).astype('uint8') for img in images]
    num_image = len(images)
    exception = None
    for _ in [None] * retry:
        try:
            if num_image == 1:
                local_path = local_path + '.png'
                cv2.imwrite(local_path, images[0][:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            else:
                frame_dir = os.path.join(os.path.dirname(local_path), '%s_frames' % (os.path.basename(local_path)))
                os.system(f'rm -rf {frame_dir}'); os.makedirs(frame_dir, exist_ok=True)
                for fid, frame in enumerate(images):
                    if fid == num_image-1: # Fix known bugs.
                        ratio = (np.sum((frame >= 117) & (frame <= 137)))/(frame.size)
                        if ratio > 0.4: continue
                    tpth = os.path.join(frame_dir, '%04d.png' % (fid+1))
                    cv2.imwrite(tpth, frame[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} -i {frame_dir}/%04d.png -vcodec libx264 -crf 17  -pix_fmt yuv420p {local_path}'
                os.system(cmd) 
                os.system(f'rm -rf {frame_dir}')
            break
        except Exception as e:
            exception = e
            continue
    
    if exception is not None:
        raise exception

@torch.no_grad()
def save_video(
    local_path,
    gen_video, 
    name_list=None,
    dataset_type=None,
    rank=0, 
    mean=[0.5, 0.5, 0.5], 
    std=[0.5, 0.5, 0.5], 
    text_size=256, 
    retry=5,
    save_fps=8
):
    """
    Saves each generated video in 'gen_video' (shape [B, C, F, H, W]) 
    into subfolders indicated by 'name_list'.
    - local_path is the base directory, e.g. "/scratch/.../inference/<unique_run_folder>"
    - name_list[index] might be "027701_027750/1053841541.mp4"
      so final path is local_path + name_list[index]
      => e.g. "/scratch/.../inference/<unique_run_folder>/027701_027750/1053841541.mp4"
    """
    if name_list is None:
        raise ValueError("save_video requires a non-None name_list.")

    # Convert to [0..255]
    vid_mean = torch.tensor(mean, device=gen_video.device).view(1, -1, 1, 1, 1)
    vid_std = torch.tensor(std, device=gen_video.device).view(1, -1, 1, 1, 1)
    gen_video = gen_video.mul_(vid_std).add_(vid_mean).clamp_(0, 1).mul_(255.0)

    # Reshape => [B, F, H, W, C]
    gen_video = rearrange(gen_video, 'b c f h w -> b f h w c')
    batch_size = gen_video.size(0)

    for index, images in enumerate(gen_video):
        images = images.cpu().numpy().astype('uint8')  # shape => [F, H, W, C]
        num_image = images.shape[0]
        exception = None

        # Build the final output path
        # E.g. local_path="/scratch/.../20250124_043255_e56bf453"
        # name_list[index]="027701_027750/1053841541.mp4"
        # => output_file="/scratch/.../20250124_043255_e56bf453/027701_027750/1053841541.mp4"
        if any(x in dataset_type for x in ["WebVid", "UCF101"]):
            output_file = os.path.join(local_path, name_list[index])
        else:
            output_file = os.path.join(local_path, os.path.basename(name_list[index]))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        for _ in [None] * retry:
            try:
                if num_image == 1:
                    # Single frame => .png
                    png_path = output_file + ".png"
                    cv2.imwrite(png_path, images[0][:, :, ::-1],
                                [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                else:
                    # Multiple frames => create a consistent temp folder under local_path
                    # instead of using os.path.dirname(output_file)
                    # to avoid accidentally writing "/temp_0".
                    frame_dir = os.path.join(local_path, f"temp_{rank}")
                    
                    # Clean up old temp folder if it exists
                    os.system(f"rm -rf {frame_dir}")
                    os.makedirs(frame_dir, exist_ok=True)

                    # Write out each frame
                    for fid, frame in enumerate(images):
                        tpth = os.path.join(frame_dir, f"{fid+1:04d}.png")
                        cv2.imwrite(tpth, frame[:, :, ::-1],
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                    cmd = (
                        f'ffmpeg -y -f image2 -loglevel quiet -framerate {save_fps} '
                        f'-i {frame_dir}/%04d.png -vcodec libx264 -crf 17 -pix_fmt yuv420p "{output_file}"'
                    )
                    os.system(cmd)
                    os.system(f"rm -rf {frame_dir}")  # Remove temp frames
                break
            except Exception as e:
                exception = e
                continue

        if exception is not None:
            raise exception

