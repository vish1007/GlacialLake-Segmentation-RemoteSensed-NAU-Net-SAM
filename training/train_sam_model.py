import random
import os
import monai
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
from torch.optim import Adam
from tqdm import tqdm
from transformers import SamModel
from utils.metrics import precision, recall, iou, f1score
# Load SAM Model
from transformers import SamModel, SamConfig
from torch.utils.data import DataLoader
from utils.sam_datasetmaker import SAMDataset
import warnings
from rasterio.errors import NotGeoreferencedWarning

# Suppress TensorFlow and Rasterio logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

################################################################################Dataset_Loading_Start###########################################################################
# where you want to save the model and plots training and validation curve
sav="/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/simple_cnn_model/sam_model"

### Input images of training validation and test 
par = "/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/simple_cnn_model/training"
train_image_dir = f"{par}/images"
train_mask_dir = f"{par}/masks"
train_prompt_dir = f"{par}/prompt/witout_post"

par = "/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/simple_cnn_model/validation"
val_image_dir = f"{par}/images"
val_mask_dir = f"{par}/masks"
val_prompt_dir = f"{par}/prompt/witout_post"


# par = "/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/eastern_himalaya-128/simple_cnn_model/test"
# test_image_dir = f"{par}/images"
# test_mask_dir = f"{par}/masks"
# test_prompt_dir = f"{par}/prompt/witout_post"


# Create datasets
train_dataset = SAMDataset(train_image_dir, train_mask_dir, train_prompt_dir)
val_dataset = SAMDataset(val_image_dir, val_mask_dir, val_prompt_dir)
# test_dataset = SAMDataset(test_image_dir, test_mask_dir, test_prompt_dir)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


for batch in train_dataloader:
    print("Pixel Values (Images) Shape:", batch["pixel_values"].shape)
    print("Ground Truth Mask Shape:", batch["ground_truth_mask"].shape)
    print("Prompt Shape:", batch["prompt"].shape)
    print("Prompt Shape:", batch["file_name"])
    print("orig Shape:", batch["orig_image"].shape)
    # print("First Image Tensor:\n", batch["pixel_values"][0])
    # print("First Mask Tensor:\n", batch["ground_truth_mask"][0])
    # print("First Prompt Tensor:\n", batch["prompt"][0])
    break  # Print only the first batch



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU if used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # More reproducible, but may be slower

set_seed(42)


######################################################################################Dataset_loading_end#######################################################

# ✅ Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

###############################################################################Architecture intial layer changes Starts ##################################################################
# ✅ Initialize SAM Model from Scratch
config = SamConfig()  # Load default SAM config
config.num_channels = 9
model = SamModel.from_pretrained("facebook/sam-vit-base")
# model = SamModel(config).to(device)  # Initialize model

# ✅ Locate the First Conv Layer (Patch Embedding)
conv1 = model.vision_encoder.patch_embed.projection  # First convolutional layer

# ✅ Create a New Conv Layer with 4 Input Channels
new_conv1 = nn.Conv2d(
    in_channels=9,  # Change from 3 → 4 channels
    out_channels=conv1.out_channels,  # Keep same output channels (768)
    kernel_size=conv1.kernel_size,  # Same kernel size (16, 16)
    stride=conv1.stride,
    padding=conv1.padding,
    bias=conv1.bias is not None  # Keep bias if it existed before
)
# # ✅ Apply Kaiming Initialization to the 4th Channel
with torch.no_grad():
    new_conv1.weight[:, :3] = conv1.weight  # Copy existing RGB weights
    init.kaiming_normal_(new_conv1.weight[:, 3:].unsqueeze(1), mode='fan_in', nonlinearity='relu')  # Kaiming Init

# ✅ Replace Old Conv Layer with the New One
model.vision_encoder.patch_embed.projection = new_conv1
model.vision_encoder.patch_embed.num_channels = 9

###############################################################################Architecture intial layer changes end##################################################################

# Freeze the vision encoder (backbone)
for name, param in model.vision_encoder.named_parameters():
    param.requires_grad = True  # Freeze vision encoder

# Freeze the prompt encoder
for name, param in model.prompt_encoder.named_parameters():
    param.requires_grad = True  # Freeze prompt encoder

# Keep only the mask decoder trainable
for name, param in model.mask_decoder.named_parameters():
    param.requires_grad = True  # Train decoder only


# Now create the optimizer with all parameters
optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=0)

# print("\n🔍 Trainable Parameters:\n")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"✅ {name} | Shape: {param.shape}")

seg_loss = monai.losses.DiceCELoss(
    sigmoid=True,
    squared_pred=True,
    reduction="mean",
  # Reduced weight for Dice Loss
    # Increased weight for Cross-Entropy Loss
)
num_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"✅ GPU is in use: {torch.cuda.get_device_name(0)}")
else:
    print("❌ GPU is not in use, running on CPU.")
model.to(device)

# Store metrics for training and validation
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
best_val_iou=0.0
# Function to unfreeze layers

   
patience = 7
no_improvement_epochs = 0
# Training loop Starts 
for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0
    total_train_precision = 0.0
    total_train_recall = 0.0
    total_train_F1scores = 0.0

    # Iterate through the training data loader
    for batch in tqdm(train_dataloader):
        # Lists to store batch-level metrics
            batch_precisions, batch_recalls, batch_f1s, batch_ious = [], [], [], []
            pixel_values = batch["pixel_values"].to(device)
            input_masks = batch["prompt"].float().to(device).unsqueeze(1)
            input_masks = F.interpolate(input_masks, size=(256, 256), mode='bilinear', align_corners=True)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device).unsqueeze(1)
            # Get model predictions
            outputs = model(pixel_values=pixel_values, input_masks=input_masks, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = F.interpolate(predicted_masks, size=(128, 128), mode='bilinear', align_corners=True)
            # Compute loss for the batch
            loss = seg_loss(predicted_masks, ground_truth_masks)
            optimizer.zero_grad()  # Zero gradients before backward pass
            loss.backward()        # Backpropagation
            optimizer.step()       # Update model parameters
    
            # Apply sigmoid to get probabilities
            predicted_masks = torch.sigmoid(predicted_masks)

            # For each image in the batch
            batch_iou, batch_precision, batch_recall, batch_F1 = 0.0, 0.0, 0.0, 0.0
            for i in range(predicted_masks.size(0)):
                pred_mask = predicted_masks[i].detach().cpu().numpy().flatten() > 0.5  # Binarize prediction
                gt_mask = ground_truth_masks[i].detach().cpu().numpy().flatten()
                tp = (pred_mask * gt_mask).sum()
                fp = (pred_mask * (1 - gt_mask)).sum()
                fn = ((1 - pred_mask) * gt_mask).sum()

                iou_score = iou(tp, fp, fn)
                precision_score = precision(tp, fp)
                recall_score = recall(tp, fn)
                F1_score = f1score(precision_score, recall_score)
              
                # Accumulate metrics for the batch
                batch_iou += iou_score
                batch_precision += precision_score
                batch_recall += recall_score
                batch_F1 += F1_score
            num_images_in_batch = predicted_masks.size(0)
            # print(num_images_in_batch)
            batch_iou /= num_images_in_batch
            batch_precision /= num_images_in_batch
            batch_recall /= num_images_in_batch
            batch_F1 /= num_images_in_batch

            total_train_loss += loss.item()
            total_train_iou += batch_iou
            total_train_precision += batch_precision
            total_train_recall += batch_recall
            total_train_F1scores += batch_F1
        
        # Calculate average loss and IoU for the epoch
    average_train_loss = total_train_loss / len(train_dataloader)
    print(len(train_dataloader))
    average_train_iou = total_train_iou / len(train_dataloader)
    average_train_precision = total_train_precision / len(train_dataloader)
    average_train_recall = total_train_recall / len(train_dataloader)
    average_train_F1score = total_train_F1scores / len(train_dataloader)
    train_losses.append(average_train_loss)
    train_ious.append(average_train_iou)
    train_precisions.append(average_train_precision)
    train_recalls.append(average_train_recall)
    train_F1scores.append(average_train_F1score)
    print(f"Epoch {epoch}")
    print(f"Train Epoch {epoch}, Avg Loss: {average_train_loss:.4f}, Avg IoU: {average_train_iou:.4f}, Avg Precision: {average_train_precision:.4f}, Avg Recall: {average_train_recall:.4f}")

    # Validation phase starts
    model.eval()
    total_val_loss = 0.0
    total_val_iou = 0.0
    total_val_precision = 0.0
    total_val_recall = 0.0
    total_val_F1scores = 0.0
    # Begin validation
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            # Forward pass
            pixel_values = batch["pixel_values"].to(device)
            input_masks = batch["prompt"].float().to(device).unsqueeze(1)
            input_masks = F.interpolate(input_masks, size=(256, 256), mode='bilinear', align_corners=True)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device).unsqueeze(1)

            # Model inference
            outputs = model(pixel_values=pixel_values, input_masks=input_masks, multimask_output=False)
            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = F.interpolate(predicted_masks, size=(128, 128), mode='bilinear', align_corners=True)
        
            # Compute loss for the batch
            vloss = seg_loss(predicted_masks, ground_truth_masks)
           
            # Apply sigmoid to get probabilities
            predicted_masks = torch.sigmoid(predicted_masks)
            batch_iou = 0.0
            batch_precision = 0.0
            batch_recall = 0.0
            batch_F1 = 0.0
            # For each image in the batch
            for i in range(predicted_masks.size(0)):
                pred_mask = predicted_masks[i].detach().cpu().numpy().flatten() > 0.5  # Binarize prediction
                gt_mask = ground_truth_masks[i].detach().cpu().numpy().flatten()
                val_tp = (pred_mask * gt_mask).sum()
                val_fp = (pred_mask * (1 - gt_mask)).sum()
                val_fn = ((1 - pred_mask) * gt_mask).sum()
                # Calculate metrics for the single image
                val_iou = iou(val_tp, val_fp, val_fn)
                val_precision = precision(val_tp, val_fp)
                val_recall = recall(val_tp, val_fn)
                val_F1_score = f1score(val_precision, val_recall)
     
                # Store metrics for this image
                batch_iou += val_iou
                batch_precision += val_precision
                batch_recall += val_recall
                batch_F1 += val_F1_score

           
            # Calculate average metrics for the batch
            num_images_in_batch = predicted_masks.size(0)
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
    average_val_loss = total_val_loss / len(val_dataloader)
    average_val_iou = total_val_iou / len(val_dataloader)
    average_val_precision = total_val_precision / len(val_dataloader)
    average_val_recall = total_val_recall / len(val_dataloader)
    average_val_F1score = total_val_F1scores / len(val_dataloader)
    val_losses.append(average_val_loss)
    val_ious.append(average_val_iou)
    val_precisions.append(average_val_precision)
    val_recalls.append(average_val_recall)
    val_F1scores.append(average_val_F1score)
    print(f"Validation Epoch {epoch}")
    print(f"Val Epoch {epoch}, Avg Loss: {average_val_loss:.4f}, Avg IoU: {average_val_iou:.4f}, Avg Precision: {average_val_precision:.4f}, Avg Recall: {average_val_recall:.4f}")
    

    # Save the model if the current IoU is the best so far
    if average_val_iou > best_val_iou:
        best_val_iou = average_val_iou
        
        os.makedirs(sav,exist_ok=True)
        torch.save(model.state_dict(),f"{sav}/model.pth")
        print(f"Best model saved with IoU: {best_val_iou:.4f}")
        no_improvement_epochs=0
    else:
        no_improvement_epochs += 1
        print(f"No improvement for {no_improvement_epochs} epochs.")
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement in validation IoU for {patience} epochs.")
        break
    torch.cuda.empty_cache()
# Plotting metrics
epochs_range = range(len(val_ious))

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
save_path = f'{sav}/tra_val.png'
plt.savefig(save_path)

# Show plot

plt.savefig(f"{sav}/metric_graph.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()