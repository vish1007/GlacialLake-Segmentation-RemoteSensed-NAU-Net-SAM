#vishalnaunet
''' Code        :   Training experiment for NAU-Net
    Author      :   Abhijith P Mahadevan    '''
import warnings
from rasterio.errors import NotGeoreferencedWarning

# Suppress the NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
import time
start_time = time.time()
import sys
sys.path.append("..")
import os
# All library imports
import numpy as np
import rasterio
import torch
import torchvision
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from models.slope_aware_naunet_seed import NAU_Net
from utils.loss import dice_loss
from utils.dataset_maker import glacialLakeDataset_fil
from utils.metrics import precision, recall, iou, f1score 
from utils.dataset_maker import TransformedDataset
from utils.dataset_maker import JointTransform

pix_thresh=20 # This is for filtering the patches.It will take that patches whose pixels is greater than equal to 20 pixels
model_name = "slope_aware_naunet_vishal" 
excel_dir="/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/5.divideby65536"
y=model_name
save_directory=f'/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/5.divideby65536/models_of_65536/{y}'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

da="/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/5.divideby65536"
training_directory = f"{da}/training"
validation_directory = f"{da}/validation"
testing_directory =f"{da}/test"
epochs =150

md=NAU_Net(n_class=1)
criterion = dice_loss()
print("No error for directory installation")
# import monai
# seg_loss = monai.losses.DiceCELoss(
#     sigmoid=False,
#     squared_pred=True,
#     reduction="mean"  # Adjust contribution of Dice Los,    # Adjust contribution of Cross-Entropy Loss
# )



# ##Global mean and standard deviation
# def calculate_band_statistics(dir):
#     # Initialize lists to store the bands
#     band1_list = []
#     band2_list = []
#     band3_list = []
#     band4_list = []

#     # Loop through the images and extract each band
#     for filename in os.listdir(dir):
#         if filename.endswith('.TIF'):  # or other relevant image extensions
#             image_path = os.path.join(dir,filename)

#             with rasterio.open(image_path) as src:
#                     # Check if the image has at least 4 bands
#                         band1 = src.read(1).flatten()  # Reading the first band
#                         band2 = src.read(2).flatten()  # Reading the second band
#                         band3 = src.read(3).flatten() # Reading the third band
#                         band4 = src.read(4).flatten()  # Reading the fourth band

#                         band1_list.extend(band1)
#                         band2_list.extend(band2)
#                         band3_list.extend(band3)
#                         band4_list.extend(band4)
            
#     # Initialize lists to store means and standard deviations
#     means = []
#     std_devs = []
#     print(f"Band 1 max :{max(band1_list)}   , min:{min(band1_list)}")
#     print(f"Band 2 max :{max(band2_list)}   , min:{min(band2_list)}")
#     print(f"Band 3 max :{max(band3_list)}   , min:{min(band3_list)}")
#     print(f"Band 4 max :{max(band4_list)}   , min:{min(band4_list)}")
#     # Calculate mean and standard deviation for each band
#     for band_flat in [band1_list, band2_list, band3_list]:
#         mean = np.mean(band_flat, axis=0)
#         std_dev = np.std(band_flat, axis=0)
#         means.append(mean)
#         std_devs.append(std_dev)

#     return means, std_devs

# Training Image directory
# dir = '/home/user/Documents/2017/datav/eastern_himalaya_with_ndwi_only_stand_512/training/images'
# means, std_devs = calculate_band_statistics(dir)

# # # print results to confirm
# for i, (mean, std_dev) in enumerate(zip(means, std_devs), 1):
#     print(f"Band {i} - Mean: {mean.mean()}, Std Dev: {std_dev.mean()}")




# #Data Transformation
# def apply_transformations(dataset, transformations):
#     """
#     Apply transformations to the dataset.

#     Args:
#         dataset (torch.utils.data.Dataset): The dataset to apply transformations to.
#         transformations (callable): The transformations to apply.

#     Returns:
#         torch.utils.data.Dataset: The dataset with applied transformations.
#     """
#     class TransformedDataset(torch.utils.data.Dataset):
#         def __init__(self, original_dataset, transform=None):
#             self.original_dataset = original_dataset
#             self.transform = transform

#         def __len__(self):
#             return len(self.original_dataset)

#         def __getitem__(self, idx):
#             image, mask = self.original_dataset[idx]
#             if self.transform:
#                 # Apply transformations to image only
#                 image = self.transform(image)
#                 # Masks are not transformed
#                 mask = torch.tensor(mask, dtype=torch.float32)
#             return image, mask
        

#     return TransformedDataset(dataset, transformations)



# Training Setup
# Initialize the datasets
training_data = glacialLakeDataset_fil(data_directory=training_directory, pixel_threshold=pix_thresh)
validation_data = glacialLakeDataset_fil(data_directory=validation_directory, pixel_threshold=pix_thresh)
# testing_data = glacialLakeDataset_fil(data_directory=testing_directory)




print("No error for dataset creation")
#apply transform

# train_transforms = transforms.Compose([

 
#     transforms.Normalize(mean=means, std=std_devs)  # Normalize with the calculated mean and std
# ])

# # Validation and testing transforms remain the same
# val_test_transforms = transforms.Compose([
#     transforms.Normalize(mean=means, std=std_devs)
# ])

# # # Apply transformations
# training_data_transformed = apply_transformations(training_data, train_transforms)
# validation_data_transformed = apply_transformations(validation_data, val_test_transforms)
# testing_data_transformed = apply_transformations(testing_data, val_test_transforms)

# ##Data Aug
# train_transforms = JointTransform(prob=0.5, degrees=15)
# training_data_transformed = TransformedDataset(training_data_normalized, transform=train_transforms)
#Initialize the dataloaders
import torch
import numpy as np

# Set seed globally first
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Optional: define worker_init_fn
def worker_init_fn(worker_id):
    # Ensure each worker gets a reproducible seed
    np.random.seed(SEED + worker_id)
    torch.manual_seed(SEED + worker_id)

# Create a reproducible generator
g = torch.Generator()
g.manual_seed(SEED)

# Updated dataloaders
training_dataloader = DataLoader(
    dataset=training_data,
    batch_size=4,
    num_workers=40,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    generator=g
)

validation_dataloader = DataLoader(
    dataset=validation_data,
    batch_size=4,
    num_workers=40,
    pin_memory=True,
    shuffle=False,  # Or True if desired, but needs generator
    worker_init_fn=worker_init_fn,
    generator=g
)

# testing_dataloader = DataLoader(dataset=testing_data, batch_size=16, num_workers=40, pin_memory=True,shuffle=None)
import matplotlib.pyplot as plt

# Check if augmentation is applied by visualizing some samples
import matplotlib.pyplot as plt

def show_augmented_samples(dataloader):
    # Get a batch of data
    data_iter = iter(dataloader)
    images, masks = next(data_iter)
    
    num_samples = 4  # Number of samples to display
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i in range(num_samples):  # Show the first 'num_samples' images and masks
        img = images[i].numpy().transpose(1, 2, 0)  # Convert CHW to HWC for display
        mask = masks[i].numpy().squeeze()  # Remove channel dimension for masks
        
        # Plot image
        axs[i, 0].imshow(img)
        axs[i, 0].set_title(f"Image {i+1}")
        axs[i, 0].axis('off')
        
        # Plot mask
        axs[i, 1].imshow(mask, cmap='gray')  # Use 'gray' colormap for masks
        axs[i, 1].set_title(f"Mask {i+1}")
        axs[i, 1].axis('off')
    
    # plt.tight_layout()
    # plt.show()


#show_augmented_samples(training_dataloader)

print("No error for dataloader creation")

# Initialize the model
torch.cuda.empty_cache()
model = md
print(type(model))

# Initialize loss and optimizer

print(type(criterion))
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
import torch

# Best trial configuration for base model 
# best_config = {
#     'lr': 0.0012152504202467165,  # Learning rate
#     'batch_size': 4,              # Batch size
#     'adam_b1': 0.09981680383588154,  # Beta1 for Adam optimizer
#     'adam_b2': 0.47128506696294736   # Beta2 for Adam optimizer
# }

# Best trial configuration for slope aware model 
best_config= {
        "lr": 0.001,
        "batch_size": 4,
    }


# Define your optimizer with the best hyperparameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=best_config['lr'],
)

train_losses = []
train_ious = []
train_precisions = []
train_recalls = []
train_F1scores = []
val_losses = []
val_ious = []
val_precisions = []
val_recalls = []
val_F1scores = []

# # Iterate through model parameters and check which ones are trainable
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name} is trainable")
#     else:
#         print(f"{name} is frozen")


# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("GPU in use")
else:
    print("GPU not in use")

model.to(device)
criterion.to(device)
#Directory to save the best model


# Initialize the best validation loss to infinity
best_val_loss=float('inf')
best_model_path=os.path.join(save_directory,'best_model.pth')
best_val_iou =0.0
print("Training in progress")
# Training loop
for epoch_number in range(epochs):
    torch.cuda.empty_cache()
    print("Epoch    :   ", epoch_number)

    # Training session begins
    model.train(True)  # Set the model to training mode
    total_train_loss = 0.0
    total_train_iou = 0.0
    total_train_precision = 0.0
    total_train_recall = 0.0
    total_train_F1scores = 0.0

    for i, data in enumerate(training_dataloader):
        # print("Epoch    :   ", epoch_number)
        # print("Batch    :   ", i)
        images, groundtruth_masks = data  # Obtain the images and masks (batch of image-mask pairs)
        images, groundtruth_masks = images.to(device), groundtruth_masks.to(device) 
        groundtruth_masks = groundtruth_masks.clone().detach().float()
        optimizer.zero_grad()  # Set the gradient to zero
        predicted_mask = model(images)
  
        #print(predicted_mask.shape, groundtruth_masks.shape)
    
        loss = criterion(predicted_mask, groundtruth_masks)  # Calculate the loss
        
        loss.backward()  # Calculate the gradients
        optimizer.step()  # Adjust the weights
        
        # Training session report
        batch_iou, batch_precision, batch_recall, batch_F1 = 0.0, 0.0, 0.0, 0.0
        for i in range(images.shape[0]):
                input_image = images[i].unsqueeze(0)
                target_mask = groundtruth_masks[i]
                output_mask = model(input_image)
                prediction = (output_mask > 0.5).float()

                tp = (prediction * target_mask).sum()
                fp = (prediction * (1 - target_mask)).sum()
                fn = ((1 - prediction) * target_mask).sum()

                # Calculate metrics for the single image
                iou_score = iou(tp, fp, fn)
                precision_score = precision(tp, fp)
                recall_score = recall(tp, fn)
                F1_score = f1score(precision_score, recall_score)

                # Accumulate metrics for the batch
                batch_iou += iou_score
                batch_precision += precision_score
                batch_recall += recall_score
                batch_F1 += F1_score
        # Calculate average metrics for the batch
        num_images_in_batch = images.shape[0]
        batch_iou /= num_images_in_batch
        batch_precision /= num_images_in_batch
        batch_recall /= num_images_in_batch
        batch_F1 /= num_images_in_batch

        # Accumulate total training metrics
        total_train_loss += loss.item()
        total_train_iou += batch_iou
        total_train_precision += batch_precision
        total_train_recall += batch_recall
        total_train_F1scores += batch_F1

    # Calculate average loss and IoU for the epoch
    average_train_loss = total_train_loss / len(training_dataloader)
    average_train_iou = total_train_iou / len(training_dataloader)
    average_train_precision = total_train_precision / len(training_dataloader)
    average_train_recall = total_train_recall / len(training_dataloader)
    average_train_F1score = total_train_F1scores / len(training_dataloader)
    train_losses.append(average_train_loss)
    train_ious.append(average_train_iou)
    train_precisions.append(average_train_precision)
    train_recalls.append(average_train_recall)
    train_F1scores.append(average_train_F1score)
    print(f"Epoch {epoch_number}")
    print(f"Train Epoch {epoch_number}, Avg Loss: {average_train_loss:.4f}, Avg IoU: {average_train_iou:.4f}, Avg F1score: {average_train_F1score:.4f}, Avg Precision: {average_train_precision:.4f}, Avg Recall: {average_train_recall:.4f}")
    
    # Training session ends

    # Validation session begins
    model.eval()
    total_val_loss = 0.0
    total_val_iou = 0.0
    total_val_precision = 0.0
    total_val_recall = 0.0
    total_val_F1scores = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(validation_dataloader):
            vimages, vgroundtruth_masks = vdata
            vimages, vgroundtruth_masks = vimages.to(device), vgroundtruth_masks.to(device)
            vgroundtruth_masks = vgroundtruth_masks.clone().detach().float()
            vpredicted_masks = model(vimages)
            vloss = criterion(vpredicted_masks, vgroundtruth_masks)

            batch_iou = 0.0
            batch_precision = 0.0
            batch_recall = 0.0
            batch_F1 = 0.0

            for i in range(vimages.shape[0]):
                input_image = vimages[i].unsqueeze(0)
                target_mask = vgroundtruth_masks[i]
                output_mask = model(input_image)
                prediction = (output_mask > 0.5).float()

                val_tp = (prediction * target_mask).sum()
                val_fp = (prediction * (1 - target_mask)).sum()
                val_fn = ((1 - prediction) * target_mask).sum()

                # Calculate metrics for the single image
                val_iou = iou(val_tp, val_fp, val_fn)
                val_precision = precision(val_tp, val_fp)
                val_recall = recall(val_tp, val_fn)
                val_F1_score = f1score(val_precision, val_recall)

                # Accumulate metrics for the batch
                batch_iou += val_iou
                batch_precision += val_precision
                batch_recall += val_recall
                batch_F1 += val_F1_score

            # Calculate average metrics for the batch
            num_images_in_batch = vimages.shape[0]
            batch_iou /= num_images_in_batch
            batch_precision /= num_images_in_batch
            batch_recall /= num_images_in_batch
            batch_F1 /= num_images_in_batch

            # Accumulate total validation metrics
            total_val_loss += vloss.item()
            total_val_iou += batch_iou
            total_val_precision += batch_precision
            total_val_recall += batch_recall
            total_val_F1scores += batch_F1

    # Calculate average loss and IoU for the epoch
    average_val_loss = total_val_loss / len(validation_dataloader)
    average_val_iou = total_val_iou / len(validation_dataloader)
    average_val_precision = total_val_precision / len(validation_dataloader)
    average_val_recall = total_val_recall / len(validation_dataloader)
    average_val_F1score = total_val_F1scores / len(validation_dataloader)
    val_losses.append(average_val_loss)
    val_ious.append(average_val_iou)
    val_precisions.append(average_val_precision)
    val_recalls.append(average_val_recall)
    val_F1scores.append(average_val_F1score)
    print(f"Validation Epoch {epoch_number}")
    print(f"Val Epoch {epoch_number}, Avg Loss: {average_val_loss:.4f}, Avg IoU: {average_val_iou:.4f}, Avg F1 score: {average_val_F1score:.4f}, Avg Precision: {average_val_precision:.4f}, Avg Recall: {average_val_recall:.4f}")
   
    #Check if this is the best model so far

    # Save the best model only if the average validation IoU improves by more than 1% (absolute change)
    if average_val_iou > best_val_iou :  # 1% more than the best_val_iou
        best_val_iou = average_val_iou
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved at epoch {epoch_number} with validation IoU {best_val_iou:.4f}")
        best_epoch = epoch_number  # Store the best epoch number
        best_train_iou = average_train_iou
        best_train_f1 = average_train_F1score
        best_train_precision = average_train_precision
        best_train_recall = average_train_recall
        best_val_iou = average_val_iou
        best_val_f1 = average_val_F1score
        best_val_precision = average_val_precision
        best_val_recall = average_val_recall

    # Validation session ends

    # Store validation metrics
 # Plotting the training and validation metrics
print(f"New best model saved at epoch {epoch_number} with validation IoU {best_val_iou:.4f}")
epochs_range = range(epochs)

plt.figure(figsize=(12, 8))

# Plot Loss
plt.subplot(2, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Loss')

# Plot IoU
plt.subplot(2, 3, 2)
plt.plot(epochs_range, train_ious, label='Train IoU')
plt.plot(epochs_range, val_ious, label='Val IoU')
plt.legend(loc='upper right')
plt.title('IoU')

# Plot Precision
plt.subplot(2, 3, 3)
plt.plot(epochs_range, train_precisions, label='Train Precision')
plt.plot(epochs_range, val_precisions, label='Val Precision')
plt.legend(loc='upper right')
plt.title('Precision')

# Plot Recall
plt.subplot(2, 3, 4)
plt.plot(epochs_range, train_recalls, label='Train Recall')
plt.plot(epochs_range, val_recalls, label='Val Recall')
plt.legend(loc='upper right')
plt.title('Recall')

# Plot F1 Score
plt.subplot(2, 3, 5)
plt.plot(epochs_range, train_F1scores, label='Train F1 Score')
plt.plot(epochs_range, val_F1scores, label='Val F1 Score')
plt.legend(loc='upper right')
plt.title('F1 Score')

# Adjust layout
plt.tight_layout()

# Save the plot to a specific location
save_path = f'{save_directory}/tra_val.png'
plt.savefig(save_path)

# Show plot
# plt.show()

##############################################################################Testing##################################################
import os
import torch
import rasterio
import numpy as np
import pandas as pd
# from models.nau_net_4 import NAU_Net
# from models.nau_net_4 import NAU_Net
# from utils.loss import dice_loss
# from utils.metrics import precision, recall, iou, f1score
from PIL import Image
import cv2
from scipy.ndimage import label

# Directories for input and output
parent=f"{testing_directory}"
test_images_dir = f"{parent}/images"
test_masks_dir = f"{parent}/masks"
# output_dir = f"{parent}/prompt_eastern_model_on_central_dataset"
output_dir=f"{save_directory}/model_out"
os.makedirs(output_dir, exist_ok=True)

# Load the pre-trained model
b_model = md
model_path = f"{save_directory}/best_model.pth"
b_model.load_state_dict(torch.load(model_path))
# b_model = torch.load(model_path)

b_model.eval()
b_model.to('cuda')

# Loss function
criterion = dice_loss().to('cuda')

# Post-processing functions
def apply_post_processing(mask, kernel_size=3):
    """
    Apply morphological closing to refine the binary mask.
    """
    mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    return (mask > 0).astype(np.uint8)

def remove_false_positives_connected_components(mask, image_shape, min_component_size=100):
    """
    Remove connected components not touching the boundary of the image and smaller than a given size.
    :param mask: Binary mask
    :param image_shape: Shape of the image (height, width)
    :param min_component_size: Minimum size of connected components to retain
    :return: Modified mask
    """
    labeled_mask, num_features = label(mask)
    mask_height, mask_width = image_shape
    final_mask = np.zeros_like(mask)

    for i in range(1, num_features + 1):
        # Extract current connected component
        component = (labeled_mask == i).astype(np.uint8)

        # Check if the component touches any boundary
        touches_boundary = np.any(component[0, :]) or np.any(component[-1, :]) or np.any(component[:, 0]) or np.any(component[:, -1])

        # Calculate component size (number of pixels)
        component_size = np.sum(component)

        # Retain the component if it touches the boundary or is large enough
        if touches_boundary or component_size >= min_component_size:
            final_mask = np.logical_or(final_mask, component)

    return final_mask.astype(np.uint8)

# Prepare to collect metrics
results = []
total_iou = total_iou_post = 0.0
total_precision = total_precision_post = 0.0
total_recall = total_recall_post = 0.0
total_f1 = total_f1_post = 0.0
num_images = 0

# Post-processing parameters
apply_post_processing_flag = True  # Toggle post-processing
morph_kernel_size = 5  # Adjust kernel size here
min_component_size = 35 # Minimum size of components to keep

for image_filename in os.listdir(test_images_dir):
    if image_filename.endswith(".TIF"):  # Ensure we process only image files
        image_path = os.path.join(test_images_dir, image_filename)
        mask_path = os.path.join(test_masks_dir, image_filename)  # Same name for mask

        # Read the image and mask
        with rasterio.open(image_path) as img:
            image = img.read().astype(np.float32)
            image_tensor = torch.tensor(image).to('cuda')

        with rasterio.open(mask_path) as msk:
            mask = msk.read(1).astype(np.float32)
            if np.sum(mask > 0) < pix_thresh:
                print(f"Skipping {image_filename}: < {pix_thresh} lake pixels--{np.sum(mask > 0)}")
                continue
            mask_tensor = torch.tensor(mask).unsqueeze(0).to('cuda')

        # Model prediction
        with torch.no_grad():
            prediction = b_model(image_tensor.unsqueeze(0))
            prediction = (prediction > 0.5).float()

            # Convert prediction to numpy
            predicted_mask = prediction.squeeze().cpu().numpy()

            # Post-process the predicted mask
            if apply_post_processing_flag:
                predicted_mask_post = apply_post_processing(predicted_mask, morph_kernel_size)
                predicted_mask_post = remove_false_positives_connected_components(predicted_mask_post, predicted_mask_post.shape, min_component_size)
            else:
                predicted_mask_post = predicted_mask

            # Calculate metrics (without post-processing)
            val_tp = (prediction * mask_tensor).sum()
            # print("True Poitives",val_tp)
            val_fp = (prediction * (1 - mask_tensor)).sum()
            val_fn = ((1 - prediction) * mask_tensor).sum()
            val_iou = iou(val_tp, val_fp, val_fn)
            val_precision = precision(val_tp, val_fp)
            val_recall = recall(val_tp, val_fn)
            val_f1_score = f1score(val_precision, val_recall)

            # Calculate metrics (with post-processing)
            mask_tensor_np = mask_tensor.cpu().numpy()
            val_tp_post = (predicted_mask_post * mask_tensor_np).sum()
            val_fp_post = (predicted_mask_post * (1 - mask_tensor_np)).sum()
            val_fn_post = ((1 - predicted_mask_post) * mask_tensor_np).sum()
            val_iou_post = iou(val_tp_post, val_fp_post, val_fn_post)
            val_precision_post = precision(val_tp_post, val_fp_post)
            val_recall_post = recall(val_tp_post, val_fn_post)
            val_f1_score_post = f1score(val_precision_post, val_recall_post)

            # Collect metrics for each image
            results.append({
                "Image": image_filename,
                "IoU": val_iou,
                "Precision": val_precision,
                "Recall": val_recall,
                "F1-Score": val_f1_score,
                "IoU_Post": val_iou_post,
                "Precision_Post": val_precision_post,
                "Recall_Post": val_recall_post,
                "F1-Score_Post": val_f1_score_post,
            })

            # Accumulate metrics
            total_iou += val_iou
            total_precision += val_precision
            total_recall += val_recall
            total_f1 += val_f1_score
            total_iou_post += val_iou_post
            total_precision_post += val_precision_post
            total_recall_post += val_recall_post
            total_f1_post += val_f1_score_post
            num_images += 1

            # # Save the predicted mask (with and without post-processing)
            # Create directories for saving predicted masks (with and without post-processing)
            os.makedirs(os.path.join(output_dir, "witout_post"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "with_post"), exist_ok=True)

            # Save the predicted mask (without post-processing)
            predicted_mask_path = os.path.join(f"{output_dir}/witout_post", f"{image_filename}")
            Image.fromarray((predicted_mask).astype(np.uint8)).save(predicted_mask_path)

            # Save the predicted mask (with post-processing)
            predicted_mask_post_path = os.path.join(f"{output_dir}/with_post", f"{image_filename}")
            Image.fromarray((predicted_mask_post).astype(np.uint8)).save(predicted_mask_post_path)
            # Calculate mean metrics
mean_iou = total_iou / num_images if num_images > 0 else 0
mean_precision = total_precision / num_images if num_images > 0 else 0
mean_f1 = total_f1 / num_images if num_images > 0 else 0
mean_recall = total_recall / num_images if num_images > 0 else 0


mean_iou_post = total_iou_post / num_images if num_images > 0 else 0
mean_precision_post = total_precision_post / num_images if num_images > 0 else 0
mean_f1_post = total_f1_post / num_images if num_images > 0 else 0
mean_recall_post = total_recall_post / num_images if num_images > 0 else 0
import pandas as pd
from datetime import datetime

# Define the model name (you can modify this dynamically if needed)
 # Example model name

# Create a DataFrame with the metrics
results_df = pd.DataFrame({
    "Model Name": [model_name],
    "IoU": [round(mean_iou, 4)],
    "Precision": [round(mean_precision, 4)],
    "Recall": [round(mean_recall, 4)],
    "F1-Score": [round(mean_f1, 4)],
    "Post-Processed IoU": [round(mean_iou_post, 4)],
    "Post-Processed Precision": [round(mean_precision_post, 4)],
    "Post-Processed Recall": [round(mean_recall_post, 4)],
    "Post-Processed F1-Score": [round(mean_f1_post, 4)],
    "best_epoch":[epoch_number],
    "Best_Train_IoU": [round(best_train_iou, 4)],
    "Best_Train_Precision": [round(best_train_precision, 4)],
    "Best_Train_Recall": [round(best_train_recall, 4)],
    "Best_Train_F1-Score": [round(best_train_f1, 4)],
    "Best_Val_IoU": [round(best_val_iou, 4)],
    "Best_Val_Precision": [round(best_val_precision, 4)],
    "Best_Val_Recall": [round(best_val_recall, 4)],
    "Best_Val_F1-Score": [round(best_val_f1, 4)]
})

# Define the path to the Excel file
metrics_file = os.path.join(excel_dir, "metrics_results.xlsx")

# Check if the Excel file already exists
if os.path.exists(metrics_file):
    # Load the existing Excel file
    existing_df = pd.read_excel(metrics_file)
    # Append the new results to the existing DataFrame
    updated_df = pd.concat([existing_df, results_df], ignore_index=True)
else:
    # Create a new DataFrame if the file doesn't exist
    updated_df = results_df

# Save the updated DataFrame to the Excel file
updated_df.to_excel(metrics_file, index=False)

# Print all metrics
print(f"Mean IoU: {mean_iou:.4f}, Mean Precision: {mean_precision:.4f}, Mean F1: {mean_f1:.4f}, Mean Recall: {mean_recall:.4f}")
print(f"Post-Processed Mean IoU: {mean_iou_post:.4f}, Mean Precision: {mean_precision_post:.4f}, Mean F1: {mean_f1_post:.4f}, Mean Recall: {mean_recall_post:.4f}")

print("Processing complete. Predictions and metrics saved.")

