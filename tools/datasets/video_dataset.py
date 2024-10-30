import os
import cv2
import json
import torch
import random
import logging
import tempfile
import numpy as np
from copy import copy
from PIL import Image
from torch.utils.data import Dataset
from utils.registry_class import DATASETS

import csv
import torchvision.transforms as transforms
from decord import VideoReader


@DATASETS.register_class()
class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            resolution=[256,256], sample_n_frames=16,
            is_image=False,
            **kwargs,
        ):
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        # zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image

        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize(resolution),
            transforms.Resize(resolution[0], interpolation=Image.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        capture = cv2.VideoCapture(video_dir)
        frame_rate = capture.get(cv2.CAP_PROP_FPS)
        stride = round(frame_rate / 3)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader


        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)

        return pixel_values, name





   
@DATASETS.register_class()
class VisualDataset(Dataset):
    def __init__(
            self,
            prompt_dir_path,
            **kwargs,
        ):
        
        self.prompt_dir_path = prompt_dir_path
        self.prompt_list = os.listdir(prompt_dir_path)
        self.length = len(self.prompt_list)

        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        videoid = self.prompt_list[idx].split('.')[0]
        videoid = videoid+".mp4"    
        prompt_path = os.path.join(self.prompt_dir_path, self.prompt_list[idx])
        prompt = open(prompt_path, 'r').readline()
        

        return videoid, prompt
    
    
@DATASETS.register_class()
class VisualDatasetRepeat(Dataset):
    def __init__(
            self,
            csv_path,
            repeat_times=5,
            **kwargs,
        ):
        
        self.prompt_path = csv_path
        self.repeat_times = repeat_times    
        with open(csv_path, 'r') as csvfile:
            self.data = list(csv.DictReader(csvfile))
        self.prompt = [item['prompt'] for item in self.data]    
        self.video_names = [p.replace(" ","_") + "-" + str(i)+".mp4" for p in self.prompt for i in range(repeat_times)]
        self.length = len(self.video_names)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_names = self.video_names[idx]
        prompt = self.prompt[idx//self.repeat_times]
        return video_names, prompt