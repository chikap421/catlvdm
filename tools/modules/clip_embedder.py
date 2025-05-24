import os
import torch
import math
import logging
import open_clip
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from utils.registry_class import EMBEDDER

import time
from PIL import Image


@EMBEDDER.register_class()
class FrozenOpenCLIPEmbedder(nn.Module):
    """
    Uses OpenCLIP Transformer Encoder for text with multiple noise injection methods.
    """
    LAYERS = ["last", "penultimate"]

    def __init__(self, pretrained, arch="ViT-H-14", device="cuda", max_length=77, 
                 freeze=True, layer="last", noise_type="none", noise_ratio=0.0):
        """
        Initializes the text encoder with optional noise injection.
        
        Args:
            pretrained (str): Path to the pretrained OpenCLIP model.
            arch (str): OpenCLIP architecture (default is "ViT-H-14").
            device (str): Device to use ("cuda" or "cpu").
            max_length (int): Maximum token length.
            freeze (bool): Whether to freeze model parameters.
            layer (str): Layer to extract features from ["last", "penultimate"].
            noise_type (str): Type of noise to apply ["none", "uniform", "gaussian", "gap", "bcni", "san", "tani"].
            noise_ratio (float): Strength of the noise perturbation.
        """
        super().__init__()
        assert layer in self.LAYERS
        self.device = device
        self.max_length = max_length
        self.noise_type = noise_type.lower()
        self.noise_ratio = noise_ratio
        self.prev_embedding = None

        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
        del model.visual  # Remove visual component (not needed for text encoding)
        self.model = model
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = 0 if layer == "last" else 1

    def freeze(self):
        """Freezes model parameters to prevent updates during training."""
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        """
        Encodes the input text and applies noise if specified.
        
        Args:
            text (str): Input text sequence.
        
        Returns:
            torch.Tensor: Encoded text representation with optional noise.
        """
        tokens = open_clip.tokenize(text).to(self.device)
        embedding = self.encode_with_transformer(tokens)

        # Apply selected noise technique
        embedding = self.apply_noise(embedding)
        self.prev_embedding = embedding.detach()  # Store for next frame
        return embedding

    def encode_with_transformer(self, text):
        """
        Passes the tokenized input through the transformer encoder.
        
        Args:
            text (torch.Tensor): Tokenized input text.
        
        Returns:
            torch.Tensor: Encoded text representation.
        """
        x = self.model.token_embedding(text) + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # (N, L, D) → (L, N, D)
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # (L, N, D) → (N, L, D)
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        """
        Passes data through the transformer layers up to the target depth.
        
        Args:
            x (torch.Tensor): Input tensor.
            attn_mask (torch.Tensor): Attention mask.
        
        Returns:
            torch.Tensor: Processed tensor.
        """
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def apply_noise(self, embedding):
        """
        Applies the selected noise perturbation to the embedding.
        
        Args:
            embedding (torch.Tensor): Encoded text representation.
        
        Returns:
            torch.Tensor: Perturbed embedding.
        """
        if self.noise_type == "none" or self.noise_ratio == 0:
            return embedding

        prev_embedding = getattr(self, "prev_embedding", None)  # Retrieve previous frame embedding
        self.prev_embedding = embedding.detach()  # Store current embedding for next frame

        if self.noise_ratio > 0:
            if self.noise_type == "gaussian":
                noise = self.gaussian_noise(embedding, self.noise_ratio)
            elif self.noise_type == "uniform":
                noise = self.uniform_noise(embedding, self.noise_ratio)
            elif self.noise_type == "gap":
                noise = self.gap(embedding)
            elif self.noise_type == "bcni":
                noise = self.bcni(embedding)
            elif self.noise_type == "tani":
                if prev_embedding is None:
                    return embedding  # No noise for the first frame
                noise = self.tani(embedding, prev_embedding)
            elif self.noise_type == "sacn":
                noise = self.sacn(embedding)
            elif self.noise_type == "hscan":
                noise = self.hscan(embedding)
            else:
                raise ValueError("Invalid noise type. Choose from 'gaussian', 'uniform', 'gap', 'bcni', 'tani', 'sacn', 'hscan'.")
            return embedding + noise

    def gaussian_noise(self, x, variance=1):
        """Applies Gaussian noise, scaled by noise_ratio."""
        dims = max(x.numel() // x.shape[0], 1) 
        std = math.sqrt(variance * self.noise_ratio) / math.sqrt(dims)
        return torch.randn_like(x) * std

    def uniform_noise(self, x, alpha=5):
        """Applies uniform noise, scaled by noise_ratio."""
        dims = max(x.numel() // x.shape[0], 1) 
        mag_norm = (alpha * self.noise_ratio) / math.sqrt(dims)
        return torch.empty_like(x).uniform_(-mag_norm, mag_norm)

    def gap(self, embedding):
        """Applies Gradient-Aligned Perturbation (GAP)."""
        grad_norm = torch.norm(embedding, dim=-1, keepdim=True)
        return grad_norm * torch.randn_like(embedding) * self.noise_ratio

    def bcni(self, embedding):
        """Applies Batch-Centered Noise Injection (BCNI)."""
        batch_mean = embedding.mean(dim=0, keepdim=True)
        distance = torch.norm(embedding - batch_mean, dim=-1, keepdim=True)
        return distance * (torch.rand_like(embedding) - 0.5) * 2 * self.noise_ratio

    def tani(self, embedding, prev_embedding):
        """
        Computes Temporal-Aware Noise Injection (TANI).
        
        Args:
            embedding (torch.Tensor): Current frame embedding.
            prev_embedding (torch.Tensor): Previous frame embedding.

        Returns:
            torch.Tensor: Perturbation noise.
        """
        delta = embedding - prev_embedding  # Compute temporal difference
        delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-6  # Avoid zero division
        noise = delta * torch.randn_like(embedding) * self.noise_ratio / delta_norm
        return noise

    def sacn(self, embedding):
        """
        Spectrum-Aware Contextual Noise (SACN) injection.
        Combines frequency domain analysis with contextual scaling.
    
        Args:
            embedding (torch.Tensor): Input embedding of shape (B, L, D) or (B, D)
        
        Returns:
            torch.Tensor: Contextualized spectral noise
        """
        # Convert to full precision for SVD computation
        orig_dtype = embedding.dtype
        orig_shape = embedding.shape
        embedding_fp32 = embedding.float()
        
        # Reshape to 2D for SVD if needed
        if len(orig_shape) == 3:
            B, L, D = orig_shape
            embedding_fp32 = embedding_fp32.reshape(-1, D)
        
        # 1. Compute spectral decomposition
        U, S, V = torch.linalg.svd(embedding_fp32, full_matrices=False)
        
        # 2. Generate frequency-aware noise
        freq_weights = torch.exp(-torch.arange(S.shape[-1], device=S.device) / S.shape[-1])
        spectral_noise = torch.randn_like(S) * freq_weights
        
        # 3. Context-awareness through local statistics
        local_scale = embedding_fp32.pow(2).mean(-1, keepdim=True)
        
        # 4. Combine spectral and contextual components
        noise = torch.matmul(
            U * spectral_noise.unsqueeze(-1),  # Scale singular vectors
            V
        ) * torch.sqrt(local_scale)
        
        # Reshape back to original shape if needed
        if len(orig_shape) == 3:
            noise = noise.reshape(orig_shape)
        
        # Convert back to original dtype and apply noise ratio
        return (noise * self.noise_ratio).to(orig_dtype)
    
    def hscan(self, embedding):
        """
        Hierarchical Spectrum-Context Adaptive Noise (HSCAN) injection.
        Combines hierarchical spectral analysis with multi-scale contextual awareness.
        
        Args:
            embedding (torch.Tensor): Input embedding of shape (B, L, D) or (B, D)
        
        Returns:
            torch.Tensor: Multi-scale contextualized noise
        """
        orig_dtype = embedding.dtype
        orig_shape = embedding.shape
        x = embedding.float()
        
        if len(orig_shape) == 3:
            B, L, D = orig_shape
            x = x.reshape(-1, D)
        
        # 1. Multi-scale decomposition
        scales = [1.0, 0.5, 0.25]  # Different analysis scales
        noise_components = []
        
        for scale in scales:
            # Compute scaled features
            scaled_x = x * scale
            
            # Spectral analysis
            U, S, V = torch.linalg.svd(scaled_x, full_matrices=False)
            
            # Progressive frequency weighting
            freq_weights = torch.exp(-torch.arange(S.shape[-1], 
                                                     device=S.device) / (S.shape[-1] * scale))
            
            # Generate scale-specific noise
            spectral_noise = torch.randn_like(S) * freq_weights
            
            # Local statistics at current scale
            local_stats = scaled_x.pow(2).mean(-1, keepdim=True).sqrt()
            
            # Combine components
            scale_noise = torch.matmul(
                U * spectral_noise.unsqueeze(-1),
                V
            ) * local_stats
            
            noise_components.append(scale_noise)
        
        # 2. Adaptive mixing of scales
        attention_weights = torch.softmax(
            torch.stack([n.pow(2).mean() for n in noise_components]), dim=0
        )
        
        # 3. Combine with learned importance
        final_noise = sum(w * n for w, n in zip(attention_weights, noise_components))
        
        # 4. Add residual connection for stability
        residual = self.gaussian_noise(x, variance=0.1)
        final_noise = final_noise + 0.1 * residual
        
        # Reshape and return
        if len(orig_shape) == 3:
            final_noise = final_noise.reshape(orig_shape)
        
        return (final_noise * self.noise_ratio).to(orig_dtype)
    

@EMBEDDER.register_class()
class FrozenOpenCLIPVisualEmbedder(nn.Module):
    """
    Uses OpenCLIP Transformer Encoder for visual embeddings with multiple noise injection methods.
    """
    LAYERS = ["last", "penultimate"]

    def __init__(self, pretrained, vit_resolution=(224, 224), arch="ViT-H-14", device="cuda",
                 freeze=True, layer="last", noise_type="none", noise_ratio=0.0):
        """
        Initializes the visual encoder with optional noise injection.

        Args:
            pretrained (str): Path to the pretrained OpenCLIP model.
            vit_resolution (tuple): Resolution for input images (default 224x224).
            arch (str): OpenCLIP architecture (default is "ViT-H-14").
            device (str): Device to use ("cuda" or "cpu").
            freeze (bool): Whether to freeze model parameters.
            layer (str): Layer to extract features from ["last", "penultimate"].
            noise_type (str): Type of noise to apply ["none", "uniform", "gaussian", "gap", "gap++", "bcni", "bcni++", "tani"].
            noise_ratio (float): Strength of the noise perturbation.
        """
        super().__init__()
        assert layer in self.LAYERS
        self.device = device
        self.noise_type = noise_type.lower()
        self.noise_ratio = noise_ratio
        self.vit_resolution = vit_resolution
        self.prev_embedding = None

        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=pretrained
        )
        del model.transformer  # Remove transformer component (not needed for images)
        self.model = model
        self.preprocessor = preprocess  # OpenCLIP preprocessing

        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = 0 if layer == "last" else 1

    def freeze(self):
        """Freezes model parameters to prevent updates during training."""
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        Encodes the input image and applies noise if specified.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Encoded image representation with optional noise.
        """
        PIL_images = [T.ToPILImage()(img) for img in image]
        preprocessed_images = torch.stack([self.preprocessor(img) for img in PIL_images]).to(self.device)
        embedding = self.model.encode_image(preprocessed_images)

        # Apply selected noise technique
        embedding = self.apply_noise(embedding)
        self.prev_embedding = embedding.detach()  # Store for next frame
        return embedding

    def apply_noise(self, embedding):
        """
        Applies the selected noise perturbation to the embedding.
        
        Args:
            embedding (torch.Tensor): Encoded text representation.
        
        Returns:
            torch.Tensor: Perturbed embedding.
        """
        if self.noise_type == "none" or self.noise_ratio == 0:
            return embedding

        prev_embedding = getattr(self, "prev_embedding", None)  # Retrieve previous frame embedding
        self.prev_embedding = embedding.detach()  # Store current embedding for next frame

        if self.noise_ratio > 0:
            if self.noise_type == "gaussian":
                noise = self.gaussian_noise(embedding, self.noise_ratio)
            elif self.noise_type == "uniform":
                noise = self.uniform_noise(embedding, self.noise_ratio)
            elif self.noise_type == "gap":
                noise = self.gap(embedding)
            elif self.noise_type == "bcni":
                noise = self.bcni(embedding)
            elif self.noise_type == "tani":
                if prev_embedding is None:
                    return embedding  # No noise for the first frame
                noise = self.tani(embedding, prev_embedding)
            elif self.noise_type == "sacn":
                noise = self.sacn(embedding)
            elif self.noise_type == "hscan":
                noise = self.hscan(embedding)
            else:
                raise ValueError("Invalid noise type. Choose from 'gaussian', 'uniform', 'gap', 'bcni', 'tani', 'sacn', 'hscan'.")
            return embedding + noise

    def gaussian_noise(self, x, variance=1):
        """Applies Gaussian noise, scaled by noise_ratio."""
        dims = max(x.numel() // x.shape[0], 1)
        std = math.sqrt(variance * self.noise_ratio) / math.sqrt(dims)
        return torch.randn_like(x) * std

    def uniform_noise(self, x, alpha=5):
        """Applies uniform noise, scaled by noise_ratio."""
        dims = max(x.numel() // x.shape[0], 1)
        mag_norm = (alpha * self.noise_ratio) / math.sqrt(dims)
        return torch.empty_like(x).uniform_(-mag_norm, mag_norm)

    def gap(self, embedding):
        """Applies Gradient-Aligned Perturbation (GAP)."""
        grad_norm = torch.norm(embedding, dim=-1, keepdim=True)
        return grad_norm * torch.randn_like(embedding) * self.noise_ratio

    def bcni(self, embedding):
        """Applies Batch-Centered Noise Injection (BCNI)."""
        batch_mean = embedding.mean(dim=0, keepdim=True)
        distance = torch.norm(embedding - batch_mean, dim=-1, keepdim=True)
        return distance * (torch.rand_like(embedding) - 0.5) * 2 * self.noise_ratio

    def tani(self, embedding, prev_embedding):
        """
        Computes Temporal-Aware Noise Injection (TANI).
        
        Args:
            embedding (torch.Tensor): Current frame embedding.
            prev_embedding (torch.Tensor): Previous frame embedding.

        Returns:
            torch.Tensor: Perturbation noise.
        """
        delta = embedding - prev_embedding  # Compute temporal difference
        delta_norm = torch.norm(delta, dim=-1, keepdim=True) + 1e-6  # Avoid zero division
        noise = delta * torch.randn_like(embedding) * self.noise_ratio / delta_norm
        return noise

    def sacn(self, embedding):
        """
        Spectrum-Aware Contextual Noise (SACN) injection.
        Combines frequency domain analysis with contextual scaling.
    
        Args:
            embedding (torch.Tensor): Input embedding of shape (B, L, D) or (B, D)
        
        Returns:
            torch.Tensor: Contextualized spectral noise
        """
        # Convert to full precision for SVD computation
        orig_dtype = embedding.dtype
        orig_shape = embedding.shape
        embedding_fp32 = embedding.float()
        
        # Reshape to 2D for SVD if needed
        if len(orig_shape) == 3:
            B, L, D = orig_shape
            embedding_fp32 = embedding_fp32.reshape(-1, D)
        
        # 1. Compute spectral decomposition
        U, S, V = torch.linalg.svd(embedding_fp32, full_matrices=False)
        
        # 2. Generate frequency-aware noise
        freq_weights = torch.exp(-torch.arange(S.shape[-1], device=S.device) / S.shape[-1])
        spectral_noise = torch.randn_like(S) * freq_weights
        
        # 3. Context-awareness through local statistics
        local_scale = embedding_fp32.pow(2).mean(-1, keepdim=True)
        
        # 4. Combine spectral and contextual components
        noise = torch.matmul(
            U * spectral_noise.unsqueeze(-1),  # Scale singular vectors
            V
        ) * torch.sqrt(local_scale)
        
        # Reshape back to original shape if needed
        if len(orig_shape) == 3:
            noise = noise.reshape(orig_shape)
        
        # Convert back to original dtype and apply noise ratio
        return (noise * self.noise_ratio).to(orig_dtype)
    
    def hscan(self, embedding):
        """
        Hierarchical Spectrum-Context Adaptive Noise (HSCAN) injection.
        Combines hierarchical spectral analysis with multi-scale contextual awareness.
        
        Args:
            embedding (torch.Tensor): Input embedding of shape (B, L, D) or (B, D)
        
        Returns:
            torch.Tensor: Multi-scale contextualized noise
        """
        orig_dtype = embedding.dtype
        orig_shape = embedding.shape
        x = embedding.float()
        
        if len(orig_shape) == 3:
            B, L, D = orig_shape
            x = x.reshape(-1, D)
        
        # 1. Multi-scale decomposition
        scales = [1.0, 0.5, 0.25]  # Different analysis scales
        noise_components = []
        
        for scale in scales:
            # Compute scaled features
            scaled_x = x * scale
            
            # Spectral analysis
            U, S, V = torch.linalg.svd(scaled_x, full_matrices=False)
            
            # Progressive frequency weighting
            freq_weights = torch.exp(-torch.arange(S.shape[-1], 
                                                     device=S.device) / (S.shape[-1] * scale))
            
            # Generate scale-specific noise
            spectral_noise = torch.randn_like(S) * freq_weights
            
            # Local statistics at current scale
            local_stats = scaled_x.pow(2).mean(-1, keepdim=True).sqrt()
            
            # Combine components
            scale_noise = torch.matmul(
                U * spectral_noise.unsqueeze(-1),
                V
            ) * local_stats
            
            noise_components.append(scale_noise)
        
        # 2. Adaptive mixing of scales
        attention_weights = torch.softmax(
            torch.stack([n.pow(2).mean() for n in noise_components]), dim=0
        )
        
        # 3. Combine with learned importance
        final_noise = sum(w * n for w, n in zip(attention_weights, noise_components))
        
        # 4. Add residual connection for stability
        residual = self.gaussian_noise(x, variance=0.1)
        final_noise = final_noise + 0.1 * residual
        
        # Reshape and return
        if len(orig_shape) == 3:
            final_noise = final_noise.reshape(orig_shape)
        
        return (final_noise * self.noise_ratio).to(orig_dtype)

@EMBEDDER.register_class()
class MotionEncoder(nn.Module):
    """
    Fine tune OpenCLIP transformer encoder
    """
    LAYERS = [
        # //"pooled",
        "last",
        "penultimate"
    ]
    def __init__(self, pretrained, arch="ViT-H-14", device="cuda", max_length=77,
                 freeze=True, layer="last",from_incomplete=False):
        super().__init__()
        assert layer in self.LAYERS
        if from_incomplete:
            path = self.prepare(pretrained)
            model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=path)
            os.remove(path)
        else:
            model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=pretrained)
            
        del model.visual
        self.model = model
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.model.train()
        
    def prepare(self,pretrained_path):
        x = torch.load(pretrained_path,map_location="cpu")
        y = torch.load("./models/modelscopet2v/open_clip_pytorch_model.bin",map_location="cpu")
        for k,v in y.items():
            if k.startswith("visual."):
                x[k] = v
        t = time.time()
        torch.save(x,f"temp_{t}.pth")
        return f"temp_{t}.pth"
    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, text):

        tokens = open_clip.tokenize(text)
        # print(tokens)
        z = self.encode_with_transformer(tokens.to(self.device))
        return tokens, z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask = None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)