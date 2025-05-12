import os
import torch
import rasterio
import random
from torch.utils.data import Dataset
from torchvision import transforms

import os
import rasterio
import torch
from torch.utils.data import Dataset
import os
import torch
import rasterio
import numpy as np
from torch.utils.data import Dataset
class glacialLakeDataset_fil(Dataset):
    def __init__(self, data_directory, transform=None, pixel_threshold=20): ## pixel threshold included so it will take lakes patches which has greater than equal to 20 pixels
        self.data_dir = data_directory
        self.transform = transform
        self.pixel_threshold = pixel_threshold
        print("Pixel_threshold",self.pixel_threshold)
        image_files = os.listdir(os.path.join(data_directory, "images"))
        mask_files = os.listdir(os.path.join(data_directory, "masks"))

        image_paths = [
            os.path.join(data_directory, "images", file) 
            for file in image_files 
            if file.lower().endswith('.tif')
        ]
        mask_paths = [
            os.path.join(data_directory, "masks", file) 
            for file in mask_files 
            if file.lower().endswith('.tif')
        ]

        image_mask_mapping = {}
        for mask_path in mask_paths:
            mask_filename = os.path.basename(mask_path)
            image_filename = mask_filename.replace('.tif', '.tif')
            for img_path in image_paths:
                if os.path.basename(img_path) == image_filename:
                    image_mask_mapping[img_path] = mask_path
                    break

        # ✅ Filter pairs based on lake pixel threshold
        self.image_paths = []
        self.mask_paths = []
        for img_path, mask_path in image_mask_mapping.items():
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # Read mask
                if np.sum(mask > 0) >= self.pixel_threshold:  # Count lake pixels
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        # After filtering
        print(f"Total valid image–mask pairs (pixel threshold ≥ {self.pixel_threshold}): {len(self.image_paths)}")

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in images and masks!"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        with rasterio.open(image_path) as src:
            image_data = src.read()
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)

        if self.transform:
            image_data = self.transform(image_data)
            mask_data = self.transform(mask_data)

        image_data = torch.tensor(image_data, dtype=torch.float32)
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)

        return image_data, mask_data


class GlacialLakeDataset_min_max(Dataset):
    def __init__(self, data_directory, transform=None):
        self.data_dir = data_directory  
        self.transform = transform

        # Get image and mask paths
        image_files = os.listdir(os.path.join(data_directory, "images"))
        mask_files = os.listdir(os.path.join(data_directory, "masks"))

        self.image_paths = [
            os.path.join(data_directory, "images", file) 
            for file in image_files 
            if file.lower().endswith('.tif')
        ]
        
        self.mask_paths = [
            os.path.join(data_directory, "masks", file) 
            for file in mask_files 
            if file.lower().endswith('.tif')
        ]

        # Map images to masks
        self.image_mask_mapping = {os.path.basename(img): img for img in self.image_paths}
        self.mask_paths = [self.image_mask_mapping[os.path.basename(mask)] for mask in self.mask_paths if os.path.basename(mask) in self.image_mask_mapping]

        # Ensure equal number of images and masks
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between images and masks"

    def __len__(self):
        return len(self.image_paths)

    def min_max_normalize(self, band):
        """ Normalize band using min-max scaling to [0,1] """
        return (band - band.min()) / (band.max() - band.min() + 1e-6)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the TIFF image
        with rasterio.open(image_path) as src:
            image_data = src.read()  # Shape: (8, H, W)

        # Load the mask
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)  # (H, W)

        with rasterio.open(image_path) as src:
        
                # Read RGBNIR and SWIR channels
                rgbns_channels = src.read([1, 2, 3]).astype('float64')
                
                nir = src.read(4).astype('float64')
                swir = src.read(5).astype('float64')
                swir2 =src.read(6).astype('float64')
                green = src.read(2).astype('float64')
                slope=src.read(8).astype('float64')
                dem=src.read(7).astype('float64')
                rgb_channels = rgbns_channels / (65536)
                nir=nir/65536
                swir=swir/65636
                swir2=swir2/65536
                slope=slope/90
                dem=dem/65536
                # Calculate NDWI and NDSI
                ndwi = (green - nir) / (green + nir + 1e-10)
                ndsi = (green - swir) / (green + swir + 1e-10)  # Normalize slope separately
                # Stack all channels including NDWI and NDSI
                ndwi_expanded = np.expand_dims(ndwi, axis=0)
                ndsi_expanded = np.expand_dims(ndsi, axis=0)
                nir_expanded = np.expand_dims(nir, axis=0)
                swir_expanded = np.expand_dims(swir, axis=0)
                swir2_expanded = np.expand_dims(swir2, axis=0)
                slope_exp     = np.expand_dims(slope, axis=0)
                dem_exp     = np.expand_dims(dem, axis=0)
                stacked_image = np.concatenate([rgb_channels, ndwi_expanded,ndsi_expanded,slope_exp,dem_exp,nir_expanded,swir_expanded,swir2_expanded], axis=0)
                stacked_image = stacked_image.astype('float32')
        # Stack processed bands
        # processed_image = np.stack([R, G, B, NIR, SWIR, SWIR2, DEM, SLOPE, NDWI, NDSI], axis=0)

        # Apply transformations if any
        if self.transform:
            stacked_image = self.transform(stacked_image)
            mask_data = self.transform(mask_data)

        # Convert to tensors
        processed_image = torch.tensor(stacked_image, dtype=torch.float32)
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)  # Add channel dim

        return processed_image, mask_data

class glacialLakeDataset(Dataset):
    def __init__(self, data_directory, transform=None):
        self.data_dir = data_directory  # the training, validation, test directory
        self.transform = transform

        # List the file paths of your TIFF images and masks
        image_files = os.listdir(os.path.join(data_directory, "images"))
        mask_files = os.listdir(os.path.join(data_directory, "masks"))

        # Filter the files to include only .TIF files
        self.image_paths = [
            os.path.join(data_directory, "images", file) 
            for file in image_files 
            if file.lower().endswith('.tif')  # Include only .TIF files (case insensitive)
        ]
        
        self.mask_paths = [
            os.path.join(data_directory, "masks", file) 
            for file in mask_files 
            if file.lower().endswith('.tif')  # Include only .TIF files (case insensitive)
        ]

        # Create a dictionary to map images to masks based on filename
        self.image_mask_mapping = {}
        for mask_path in self.mask_paths:
            mask_filename = os.path.basename(mask_path)
            # Assuming the mask filename corresponds to the image filename (without .tif extension)
            image_filename = mask_filename.replace('.tif', '.tif')  # Adjust if mask filename differs
            for img_path in self.image_paths:
                if os.path.basename(img_path) == image_filename:
                    self.image_mask_mapping[img_path] = mask_path
                    break

        # Extract image paths and corresponding mask paths into lists
        self.image_paths = list(self.image_mask_mapping.keys())
        self.mask_paths = list(self.image_mask_mapping.values())

        # Ensure that the number of images and masks match
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load the TIFF files 
        with rasterio.open(image_path) as src:
            image_data = src.read()  # Read the image data as a NumPy array

        with rasterio.open(mask_path) as src:  # Open the mask image using rasterio
            mask_data = src.read(1)  # Read the mask data as a NumPy array

        # Apply transformations if specified
        if self.transform:
            image_data = self.transform(image_data)
            mask_data = self.transform(mask_data)

        # Convert to tensors
        image_data = torch.tensor(image_data, dtype=torch.float32)
        mask_data = torch.tensor(mask_data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image_data, mask_data


class JointTransform:
    def __init__(self, prob=0.5, degrees=15):
        self.prob = prob
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Random Horizontal Flip
        if random.random() < self.prob:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        # Random Vertical Flip
        if random.random() < self.prob:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # Random Rotation
        angle = random.uniform(-self.degrees, self.degrees)
        image = transforms.functional.rotate(image, angle)
        mask = transforms.functional.rotate(mask, angle)

        return {'image': image, 'mask': mask}

# Apply the joint transformations using the custom transform class


# Dataset wrapper to apply transformations
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, transform=None):
        self.original_dataset = original_dataset
        self.transform = transform

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, mask = self.original_dataset[idx]
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['mask']
