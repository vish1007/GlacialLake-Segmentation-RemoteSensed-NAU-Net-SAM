import os
import torch
import torchvision.transforms as T
import rasterio
import numpy as np
from torch.utils.data import Dataset
import rasterio
import torchvision.transforms as T
from transformers import SamProcessor

# Load once globally
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

class SAMDataset(Dataset):
    def __init__(self, image_dir, mask_dir, prompt_dir=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.prompt_dir = prompt_dir
        self.transform = transform

        self.image_names = sorted([f for f in os.listdir(image_dir) if f.endswith('.TIF')])
        self.mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.TIF')])
        self.prompt_names = sorted([f for f in os.listdir(prompt_dir) if f.endswith('.TIF')])

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # ---- Read image ----
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Shape: (9, H, W)
            orig_image = image.copy()  
        # Separate RGB and extra bands
        rgb_image = (image[:3] * 255).astype(np.uint8)  # (3, H, W)
        extra_bands = image[3:]  # (6, H, W)

        # Apply SamProcessor on RGB bands (returns 1024x1024)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))  # (H, W, 3)
        rgb_processed = processor.image_processor(rgb_image, return_tensors="pt")["pixel_values"][0]  # (3, 1024, 1024)

        # Resize extra bands to (6, 1024, 1024)
        extra_bands_resized = torch.nn.functional.interpolate(
            torch.from_numpy(extra_bands).unsqueeze(0).float(),  # Add batch dim → (1, 6, H, W)
            size=(1024, 1024),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # Back to (6, 1024, 1024)

        # Combine RGB and resized extra bands → (9, 1024, 1024)
        full_image_tensor = torch.cat([rgb_processed, extra_bands_resized], dim=0)

        # ---- Read mask ----
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # ---- Read prompt ----
        prompt_path = os.path.join(self.prompt_dir, self.prompt_names[idx])
        with rasterio.open(prompt_path) as src:
            prompt = src.read(1)

        return {
            "pixel_values": full_image_tensor,   # (9, 1024, 1024)
            "ground_truth_mask": mask,
            "prompt": prompt,
            "file_name": self.image_names[idx],
            "orig_image":orig_image
        }

# Define transformations
transform = T.Compose([
    T.ToTensor(),                      
    T.Resize((1024, 1024)),            
])
