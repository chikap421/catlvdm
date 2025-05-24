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
class WebVid(Dataset):
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
        
        video_dir    = os.path.join(self.video_folder, page_dir, f"{videoid}.mp4")
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
class InferenceDatasetWebVid(Dataset):
    def __init__(self, csv_path, repeat_times=1, **kwargs):
        """
        A simple dataset that reads from csv_path and repeats each row 'repeat_times' times.

        We assume the CSV has columns: videoid, page_dir, name, etc.
        We'll return (videoid, page_dir, prompt).
        """
        self.repeat_times = repeat_times
        with open(csv_path, 'r') as f:
            self.data = list(csv.DictReader(f))

        self.videoids  = [row['videoid'] for row in self.data]
        self.page_dirs = [row['page_dir'] for row in self.data]
        self.prompts   = [row['name'] for row in self.data]

        self.length = len(self.data) * repeat_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_times
        videoid  = self.videoids[base_idx]
        page_dir = self.page_dirs[base_idx]
        prompt   = self.prompts[base_idx]
        return videoid, page_dir, prompt

@DATASETS.register_class()
class InferenceDatasetMSRVTT(Dataset):
    def __init__(
            self,
            json_path,
            repeat_times=1,
            seed=42,  # Add a seed for reproducibility
            **kwargs,
        ):
        self.json_path = json_path
        self.repeat_times = repeat_times
        self.seed = seed  # Store the seed

        # Load dataset from JSON file
        with open(json_path, 'r') as jsonfile:
            self.dataset = json.load(jsonfile)

        # Extract video names
        self.video_names = [video_id + ".mp4" for video_id in self.dataset.keys()]
        self.length = len(self.video_names)

        # 🔥 Set a fixed random seed to make results deterministic
        random.seed(self.seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        captions = self.dataset.get(video_name.replace(".mp4", ""), {}).get("captions", [""])  
        return video_name, random.choice(captions)  # Random but deterministic due to fixed seed

@DATASETS.register_class()
class InferenceDatasetMSVD(Dataset):
    def __init__(
            self,
            json_path,
            repeat_times=1,
            seed=42,  # Add a seed for reproducibility
            **kwargs,
        ):
        self.json_path = json_path
        self.repeat_times = repeat_times
        self.seed = seed  # Store the seed

        # Load dataset from JSON file
        with open(json_path, 'r') as jsonfile:
            self.dataset = json.load(jsonfile)

        # Extract video names (MSVD format: "<video_id>_<start>_<end>")
        self.video_names = [video_id + ".mp4" for video_id in self.dataset.keys()]
        self.length = len(self.video_names)

        # 🔥 Set a fixed random seed to make results deterministic
        random.seed(self.seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        captions = self.dataset.get(video_name.replace(".mp4", ""), {}).get("captions", [""])  
        return video_name, random.choice(captions)  # Random but deterministic due to fixed seed


@DATASETS.register_class()
class InferenceDatasetTGIF(Dataset):
    def __init__(
            self,
            csv_path,
            repeat_times=1,
            **kwargs,
        ):
        self.csv_path = csv_path
        self.repeat_times = repeat_times

        # Load dataset from CSV file
        self.dataset = {}
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                video_id, caption, _, _ = row  # Extract relevant columns
                self.dataset[video_id] = caption

        # Extract video names (TGIF uses GIFs, but we assume ".mp4" for consistency)
        self.video_names = [video_id + ".mp4" for video_id in self.dataset.keys()]
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        caption = self.dataset.get(video_name.replace(".mp4", ""), "")  # Retrieve caption
        return video_name, caption

@DATASETS.register_class()
class InferenceDatasetUCF101(Dataset):
    def __init__(self, csv_path, repeat_times=1, **kwargs):
        """
        A dataset for UCF101 with class label prompts.

        CSV format: each row contains [video_name, class_label]
        Example:
            v_ApplyEyeMakeup_g01_c01.mp4, ApplyEyeMakeup
        """
        self.repeat_times = repeat_times
        self.video_names = []
        self.class_labels = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header if present
            for row in reader:
                if len(row) < 2:
                    continue
                self.video_names.append(row[0])
                self.class_labels.append(row[1])

        self.length = len(self.video_names) * repeat_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_times
        return self.video_names[base_idx], self.class_labels[base_idx]

@DATASETS.register_class()
class InferenceDatasetTest(Dataset):
    def __init__(self, csv_path, repeat_times=10, **kwargs):
        """
        A dataset class for custom test prompts.
        CSV format: id,prompt
        Example:
            0,A small bird sits atop a blooming flower stem.
        """
        self.repeat_times = repeat_times
        self.video_names = []
        self.prompts = []

        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) < 2:
                    continue
                video_id = row[0].strip()
                prompt = ",".join(row[1:]).strip()
                self.video_names.append(f"{video_id}.mp4")
                self.prompts.append(prompt)

        self.length = len(self.video_names) * repeat_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_times
        return self.video_names[base_idx], self.prompts[base_idx]

@DATASETS.register_class()
class InferenceDatasetVBench(Dataset):
    def __init__(self, json_path, repeat_times=5, **kwargs):
        """
        Dataset for evaluating generated videos using VBench prompts.
        Each video is saved as <prompt>-<repeat_index>.mp4
        """
        self.repeat_times = repeat_times
        with open(json_path, 'r') as f:
            self.prompts = json.load(f)

        self.length = len(self.prompts) * repeat_times

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        base_idx = idx // self.repeat_times
        repeat_idx = idx % self.repeat_times

        raw_prompt = self.prompts[base_idx]
        sanitized_prompt = raw_prompt
        video_name = f"{sanitized_prompt}-{repeat_idx}"
        return video_name, raw_prompt


@DATASETS.register_class()
class InferenceDatasetEvalCrafter(Dataset):
    def __init__(self, csv_path, **kwargs):
        """
        Dataset for EvalCrafter inference.
        CSV format: video_id,caption
        Output video names: 0000, 0001, ...
        """
        import csv
        self.entries = []

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append((row["video_id"].strip(), row["caption"].strip()))

        self.length = len(self.entries)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name, caption = self.entries[idx]
        return video_name, caption