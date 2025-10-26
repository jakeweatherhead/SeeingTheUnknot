# The MIT License (MIT)
# Copyright © 2025 Jake Weatherhead

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import cv2
from abc import ABC, abstractmethod
from functools import singledispatch
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
from PIL import Image

from config import config
from config.config import CNNConfig, ViTConfig
import constants.constant as C
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from schema.dataset import KnotDataset


class Cam(ABC):
    """Abstract base class for saliency map generators."""
    
    def __init__(
        self, 
        model_name: str
    ):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
    
    def load_model(
        self, 
        model_path: str
    ) -> None:
        """Load the trained model from checkpoint."""
        self._model = timm.create_model(
            self.model_name, 
            pretrained=False, 
            num_classes=2
        )
        ckpt = torch.load(model_path, map_location=self.device)
        self._model.load_state_dict(ckpt)
        self._model = self._model.to(self.device).eval()
    
    def preprocess_image(
        self, 
        image_path: str
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """Preprocess image for model input."""
        image = Image.open(image_path).convert('RGB')
        rgb_img = np.array(image, dtype=np.float32) / 255.0
        
        input_tensor = preprocess_image(
            img=rgb_img,
            mean=C.IMAGENET_CHANNEL_MEANS,
            std=C.IMAGENET_CHANNEL_STDS
        ).to(self.device)
        
        return rgb_img, input_tensor
    
    def predict_with_model(
        self, 
        input_tensor: torch.Tensor
    ) -> Tuple[int, float, str]:
        """Make prediction with the model and return predicted class."""
        with torch.no_grad():
            outputs = self._model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(outputs, 1).item()
            confidence = probabilities[predicted_class].item()
        
        class_name = ["KNOT", "UNKNOT"][predicted_class]
        return predicted_class, confidence, class_name
    
    @abstractmethod
    def compute_saliency_map(
        self, 
        input_tensor: torch.Tensor, 
        predicted_class: int
    ) -> np.ndarray:
        """Compute saliency map for the given input."""
        ...
    
    def save_saliency_map(
        self, 
        cam_image: np.ndarray, 
        image_path: str, 
        true_label: int, 
        predicted_label: str, 
        confidence: float,
        output_dir: str
    ) -> None:
        """Save saliency map with consistent naming and directory structure."""
        knot_type = 'knots' if true_label == 0 else 'unknots'
        
        # Determine confusion matrix category
        true_class = true_label
        predicted_class = 0 if predicted_label == "KNOT" else 1
        
        if knot_type == 'knots':  # True label is knot
            confusion_category = 'TN' if true_class == predicted_class else 'FP'
        else:  # True label is unknot
            confusion_category = 'TP' if true_class == predicted_class else 'FN'
        
        final_output_dir = os.path.join(output_dir, knot_type, confusion_category)
        os.makedirs(final_output_dir, exist_ok=True)
        
        true_label_str = ['KNOT', 'UNKNOT'][true_label]
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(
            final_output_dir, 
            f'{image_name}_T_{true_label_str}_P_{predicted_label}_C_{confidence:.3f}.jpg'
        )
        cv2.imwrite(output_path, cam_image)
    
    def process_single_image(
        self, 
        image_path: str, 
        true_label: int, 
        output_dir: str
    ) -> None:
        """Process a single image and save its saliency map."""
        try:
            rgb_img, input_tensor = self.preprocess_image(image_path)
            predicted_class, confidence, predicted_label = self.predict_with_model(input_tensor)
            grayscale_cam = self.compute_saliency_map(input_tensor, predicted_class)
            
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            
            self.save_saliency_map(
                cam_image, image_path, true_label, predicted_label, confidence, output_dir
            )
            
        except Exception as e:
            raise Exception(f"Error processing {image_path}: {str(e)}")
    
    def run_cam(
        self, 
        dataset: KnotDataset, 
        ckpt_paths: List[Path]
    ) -> None:
        """
        Generate saliency maps for all images in dataset.
        
        Args:
            dataset: Dataset containing image paths and labels
            ckpt_paths: Paths to checkpoint models
        """
        for ckpt_path in ckpt_paths:
            self.load_model(str(ckpt_path))
            output_dir = ckpt_path.parent / "saliency_maps"
            
            for image_id in range(len(dataset)):
                self.process_single_image(
                    image_path=dataset.image_paths[image_id], 
                    true_label=dataset.labels[image_id], 
                    output_dir=str(output_dir)
                )


class CNNCam(Cam):
    """CNN saliency map generator using Grad-CAM."""
    
    def __init__(
        self, 
        model_name: str = "convnext_base.fb_in22k_ft_in1k"
    ):
        super().__init__(model_name)
    
    def compute_saliency_map(
        self, 
        input_tensor: torch.Tensor, 
        predicted_class: int
    ) -> np.ndarray:
        """Compute Grad-CAM for the CNN model."""
        cam = GradCAM(model=self._model, target_layers=[self._model.stages[-1]])
        targets = [ClassifierOutputTarget(predicted_class)]
        
        return cam(
            input_tensor=input_tensor, 
            targets=targets
        )[0, :]  # First image, all spatial dimensions


class ViTCam(Cam):
    """ViT saliency map generator using attention rollout."""
    
    def __init__(
        self, 
        model_name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k"
    ):
        super().__init__(model_name)
    
    def compute_saliency_map(
        self, 
        input_tensor: torch.Tensor, 
        predicted_class: int
    ) -> np.ndarray:
        """Compute attention rollout."""
        attention_weights = []
        
        def attention_hook(module, input, output):
            B, N, C = input[0].shape
            qkv = module.qkv(input[0]).reshape(
                B, N, 3, module.num_heads, C // module.num_heads
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)
            
            attention_weights.append(attn.cpu().detach())
        
        hooks = []
        for block in self._model.blocks:
            hook = block.attn.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self._model(input_tensor)
        
        for hook in hooks:
            hook.remove()
        
        attention_maps = []
        for attn in attention_weights:
            attn = attn.mean(dim=1)
            
            residual_att = torch.eye(attn.size(-1))
            aug_att_mat = attn + residual_att
            aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)
            
            attention_maps.append(aug_att_mat[0])
        
        rollout = attention_maps[0]
        for attention_map in attention_maps[1:]:
            rollout = torch.matmul(rollout, attention_map)
        
        cls_attention = rollout[0, 1:]
        
        patches_per_side = 14  # 224/16 for ViT

        attention_map = cls_attention.reshape(patches_per_side, patches_per_side)
        attention_map = attention_map.numpy()
        attention_map = cv2.resize(attention_map, (224, 224))
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        return attention_map


@singledispatch
def generate_saliency_maps(
    config: config.Config, 
    **params
) -> None:
    """
    Generate saliency maps for the appropriate architecture branching on config type.
    
    Args:
        config: Configuration object (CNNConfig or ViTConfig)
        **params: Additional arguments passed to the generator
    """
    ...

@generate_saliency_maps.register
def _(
    config: CNNConfig, 
    **params
) -> None:
    """Generate saliency maps for the CNN model"""
    generator = CNNCam(model_name=config.model_name)
    generator.run_cam(**params)


@generate_saliency_maps.register
def _(
    config: ViTConfig,
    **params
) -> None:
    """Generate saliency maps for the ViT model."""
    generator = ViTCam(model_name=config.model_name)
    generator.run_cam(**params)
