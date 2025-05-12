from tqdm import tqdm
import os
from statistics import mean
import torch
from PIL import Image
from torch.optim import Adam
import monai
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from utils.loss import dice_loss
import torch.nn as nn
import torch.nn.init as init
from models.simple_cnn import simple
# Initialize the model from scratch (random weights)
from transformers import SamModel, SamConfig
from utils.test_datsetmaker import SAMDataset
from utils.metrics import iou,precision,recall,f1score
from torch.utils.data import DataLoader
import warnings
import pandas as pd
import pandas as pd
from datetime import datetime

# Ignore all warnings
warnings.filterwarnings("ignore")

from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=UserWarning)


##################################################################Here specify the directory ######################################################

path = '/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels'  # <-- change this

# List only folders (not files)
folder_list = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
# or you can give folder or particular himalaya range
folder_list=["simple_cnn_model"]
iou_matrix = pd.DataFrame(index=folder_list, columns=folder_list)
for md in folder_list:
    print(md)
    for tes in folder_list:
        print(f"model-{md}-Testing-{tes}")
        par = f"/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/{tes}/final_model_all/test"
        test_image_dir = f"{par}/images"
        test_mask_dir = f"{par}/masks"
    

        model_path = f"/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/{md}/final_model_all/best_model.pth"
        # Create datasets
        sam_model_path=f"/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/{md}/final_model_all/sam_model_all_bands/model.pth"
        test_dataset = SAMDataset(test_image_dir, test_mask_dir)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


        # Create datasets

        
        # You want to save the output or not 
        save=True
        excel_dir="/home/user/Documents/2017/dataset/128/latest_dataset"
        
        save_dir = f'/home/user/Documents/2017/dataset/128/latest_dataset/greater_than_20_pixels/{md}/final_model_all/sam_model_all_bands/out_try'
        ##################################################################-----------Model-----------------############################


        # ✅ Check for GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"

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



        # ✅ Apply Kaiming Initialization to the 4th Channel
        with torch.no_grad():
            new_conv1.weight[:, :3] = conv1.weight  # Copy existing RGB weights
            init.kaiming_normal_(new_conv1.weight[:, 3:].unsqueeze(1), mode='fan_in', nonlinearity='relu')  # Kaiming Init

        # ✅ Replace Old Conv Layer with the New One
        model.vision_encoder.patch_embed.projection = new_conv1

        # ✅ Print Updated Model Structure
        # print(model.vision_encoder)
        # print(model.vision_encoder.patch_embed.projection)
        # print(f"Updated config num_channels: {model.config.num_channels}")
        # print(f"Updated input layer: {model.vision_encoder.patch_embed.projection}")

        # print("\n✅ SAM Model now correctly accepts 4-channel input!")
        # print("🔹 Vision Encoder Configuration:")
        # # print(f"Input Channels: {config.num_channels}")
        # print(f"Expected input channels: {model.config.num_channels}")

        model.vision_encoder.patch_embed.num_channels = 9

        if device == "cuda":
            print(f"✅ GPU is in use: {torch.cuda.get_device_name(0)}")
        else:
            print("❌ GPU is not in use, running on CPU.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model Naunet
        b_model = simple()

        b_model.load_state_dict(torch.load(model_path))
        b_model.eval()
        b_model.to(device)
    
        # Move the model to the appropriate device

        # seg_loss=dice_loss()
        seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        # Load the saved model weights
        model.load_state_dict(torch.load(sam_model_path, map_location=device))
        
        model.to(device)
        model.eval()
        total_test_loss = 0.0
        total_test_iou = 0.0
        total_test_precision = 0.0
        total_test_recall = 0.0
        total_test_F1scores = 0.0
        # Begin testidation
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                # Forward pass
                nau_net_input=batch["orig_image"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                # input_points=batch["input_points"].to(device)
            
                ground_truth_masks = batch["ground_truth_mask"].float().to(device).unsqueeze(1)
                file_name=batch["file_name"]
            
                # naunet__model

                prediction = b_model(nau_net_input)
                prediction = (prediction > 0.5).float()
                input_masks = prediction.float().to(device)
                # print(input_masks.shape)
                input_masks = F.interpolate(input_masks, size=(256, 256), mode='bilinear', align_corners=True)


                # Model inference SAm model
                outputs = model(pixel_values=pixel_values,input_masks=input_masks,multimask_output=False)
                predicted_masks = outputs.pred_masks.squeeze(1)
                predicted_masks = F.interpolate(predicted_masks, size=(128, 128), mode='bilinear', align_corners=True)
                # predicted_masks = torch.sigmoid(predicted_masks)
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
                    pred_mask = predicted_masks[i].detach().cpu().numpy().flatten() > 0.5# Binarize prediction
                    gt_mask = ground_truth_masks[i].detach().cpu().numpy().flatten()
                    test_tp = (pred_mask * gt_mask).sum()
                    test_fp = (pred_mask * (1 - gt_mask)).sum()
                    test_fn = ((1 - pred_mask) * gt_mask).sum()
                    # Calculate metrics for the single image
                    test_iou = iou(test_tp, test_fp, test_fn)
                    test_precision = precision(test_tp, test_fp)
                    test_recall = recall(test_tp, test_fn)
                    test_F1_score = f1score(test_precision, test_recall)
                    # # Calculate metrics for this image
                    # precision = precision_score(gt_mask, pred_mask, average="binary", zero_division=0)
                    # recall = recall_score(gt_mask, pred_mask, average="binary", zero_division=0)
                    # f1 = f1_score(gt_mask, pred_mask, average="binary", zero_division=0)
                    # iou = jaccard_score(gt_mask, pred_mask, average="binary", zero_division=0)
                    # Save the predicted and ground truth mask images
                    pred_mask = (predicted_masks[i].detach().cpu().numpy() > 0.5).astype(np.uint8)
                
                    # Ensure the mask has the correct 2D shape (128, 128)
                    if pred_mask.ndim == 3:  # In case of an extra singleton dimension
                        pred_mask = pred_mask.squeeze()  # Remove singleton dimensions

                    # Create the image filename (e.g., image_0_predicted.TIF)
                    image_filename = f"{file_name[i]}"
                    
                    # Save the predicted mask as a .TIF file using PIL
                    if save:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir,exist_ok=True)
                        predicted_mask_path = os.path.join(save_dir, image_filename)

                        # Save the mask using PIL directly
                    
                        Image.fromarray(pred_mask).save(predicted_mask_path)
                            # Store metrics for this image
                    batch_iou += test_iou
                    batch_precision += test_precision
                    batch_recall += test_recall
                    batch_F1 += test_F1_score

                
                # Calculate average metrics for the batch
                num_images_in_batch = predicted_masks.size(0)
                batch_iou /= num_images_in_batch
                batch_precision /= num_images_in_batch
                batch_recall /= num_images_in_batch
                batch_F1 /= num_images_in_batch

                    # Accumulate total testidation metrics
                total_test_loss += vloss.item()
                total_test_iou += batch_iou
                total_test_precision += batch_precision
                total_test_recall += batch_recall
                total_test_F1scores += batch_F1


            # Calculate average loss and IoU for the epoch
        average_test_loss = total_test_loss / len(test_dataloader)
        average_test_iou = total_test_iou / len(test_dataloader)
        average_test_precision = total_test_precision / len(test_dataloader)
        average_test_recall = total_test_recall / len(test_dataloader)
        average_test_F1score = total_test_F1scores / len(test_dataloader)

        print(f"Avg Loss: {average_test_loss:.4f}, Avg IoU: {average_test_iou:.4f}, Avg Precision: {average_test_precision:.4f}, Avg Recall: {average_test_recall:.4f},Avg F1 score:{average_test_F1score}")


        # Create a DataFrame with the metrics
        results_df = pd.DataFrame({
            "Model Name": [md],
            "Test_dataset":[tes],
            "IoU": [round(average_test_iou, 4)],
            "Precision": [round(average_test_precision, 4)],
            "Recall": [round(average_test_recall, 4)],
            "F1-Score": [round(average_test_F1score, 4)],
        })
        iou_matrix.loc[md, tes] = round(average_test_iou, 4)
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
# After all loops, save it
metrics_file = os.path.join(excel_dir, "metrics_results_iou_only.xlsx")
iou_matrix.to_excel(metrics_file)