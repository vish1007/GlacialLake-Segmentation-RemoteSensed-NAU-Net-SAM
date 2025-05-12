''' Code        : Loss functions
    Description : This file contains loss function used for experiments'''

# all library imports
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Dice loss
class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self, predicted, target):
        numerator = torch.sum(predicted * target)
        denominator = torch.sum(predicted) + torch.sum(target)
        loss = 1 - (2*(numerator) / (denominator + 1e-6)) 
        return loss
# import torch
# import torch.nn as nn
# from scipy.ndimage import label  # For connected components

# class dice_loss_per_lake(nn.Module):  ## By Vishal
#     def __init__(self):
#         super(dice_loss_per_lake, self).__init__()

#     def forward(self, predicted, target):
#         # Apply sigmoid to raw scores for soft predictions
#         predicted = torch.sigmoid(predicted)

#         # Convert target to numpy for connected component labeling
#         target_np = target.cpu().numpy()
#         labeled_target, num_lakes = label(target_np)  # Label each connected component (lake)

#         # Convert labeled target back to a tensor
#         labeled_target = torch.tensor(labeled_target, device=target.device, dtype=torch.float32)

#         per_lake_losses = []
#         for lake_id in range(1, num_lakes + 1):  # Skip background (label 0)
#             # Create a mask for the current lake
#             lake_mask = (labeled_target == lake_id).float()

#             # Compute soft Dice loss for the current lake
#             numerator = 2 * torch.sum(predicted * lake_mask)
#             denominator = torch.sum(predicted) + torch.sum(lake_mask)
#             lake_loss = 1 - (numerator / (denominator + 1e-6))
            
#             per_lake_losses.append(lake_loss)

#         # Compute average Dice loss across all lakes
#         if len(per_lake_losses) > 0:
#             avg_loss = torch.mean(torch.stack(per_lake_losses))
#         else:
#             # No lakes in target, return 0 loss
#             avg_loss = torch.tensor(0.0, device=target.device)

#         return avg_loss
# class dice_loss_per_lake(nn.Module):
#         def __init__(self):
#             super(dice_loss_per_lake, self).__init__()

#         def forward(self, predicted, target):
#             # Apply sigmoid to raw scores for soft predictions
#             predicted = torch.sigmoid(predicted)

#             # Initialize a variable to accumulate the total loss across the entire batch
#             total_loss = 0.0

#             # Iterate through each image in the batch (batch size 4)
#             for i in range(predicted.size(0)):  # Loop over batch size
#                 # Get the target for the current image in the batch
#                 target_np = target[i].cpu().numpy()
#                 labeled_target, num_lakes = label(target_np)  # Label each connected component (lake)

#                 # Convert labeled target back to a tensor
#                 labeled_target = torch.tensor(labeled_target, device=target.device, dtype=torch.float32)

#                 per_lake_losses = []
#                 for lake_id in range(1, num_lakes + 1):  # Skip background (label 0)
#                     # Create a mask for the current lake
#                     lake_mask = (labeled_target == lake_id).float()

#                     # Compute soft Dice loss for the current lake
#                     numerator = 2 * torch.sum(predicted[i] * lake_mask)
#                     denominator = torch.sum(predicted[i]) + torch.sum(lake_mask)
#                     lake_loss = 1 - (numerator / (denominator + 1e-6))

#                     per_lake_losses.append(lake_loss)

#                 # Average Dice loss for the current image
#                 if len(per_lake_losses) > 0:
#                     avg_loss = torch.mean(torch.stack(per_lake_losses))  # Average the per-lake losses for this image
#                 else:
#                     # No lakes in target, no loss for this image
#                     avg_loss = torch.tensor(0.0, device=target.device)

#                 # Add the averaged loss for this image to the total loss
#                 total_loss += avg_loss

#             # Return the total loss across all images in the batch
#             return total_loss



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Apply sigmoid to inputs if needed
        #inputs = inputs.sigmoid()
        
        # Calculate focal loss
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)  # probability of correct class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        # Reduction
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

    

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-6):
        # Apply sigmoid to get probabilities
        #inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Compute intersection and union
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        # Compute IoU
        iou = (intersection + smooth) / (union + smooth)

        # Compute IoU loss
        return 1 - iou
    

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, eps=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        true_pos = (inputs * targets).sum()
        false_neg = (targets * (1 - inputs)).sum()
        false_pos = ((1 - targets) * inputs).sum()
        tversky = (true_pos + self.eps) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.eps)
        return 1 - tversky


# Sigmoid Focal loss
class sigmoidfocal_loss(nn.Module):
    '''Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.'''
    def __init__(self):
        super(sigmoidfocal_loss, self).__init__()

    def forward(inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.6, gamma: float = 2, reduction: str = "none",) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'")
        return loss

# Instance IOU loss
class instanceiou_loss(nn.Module):
    def __init__(self):
        super(instanceiou_loss, self).__init__()

    def forward(self, predicted_masks,actual_masks):
        actual_masks = actual_masks.squeeze(1)
        predicted_masks = predicted_masks.squeeze(1)

        # Calculate loss for each sample in the batch
        batch_loss = 0.0
        batch_size = actual_masks.shape[0]
        for i in range(batch_size):
            actual_mask = actual_masks[i]
            predicted_mask = predicted_masks[i]
            iou=(((actual_mask*predicted_mask).sum()))/(actual_mask.sum()+predicted_mask.sum()) #*2 removed
            avg_modified_instance_iou = self.compute_instance_iou_loss(actual_mask, predicted_mask)
            batch_loss += (1-0.5*avg_modified_instance_iou-iou)
        
        # Calculate mean loss across the batch
        avg_loss = batch_loss / batch_size
        return avg_loss

    def compute_instance_iou_loss(self, actual_mask, predicted_mask):
        # Convert actual binary mask to instance-based mask
        device = torch.device('cuda:0')
        labels, actual_labels = cv2.connectedComponents(actual_mask.detach().cpu().numpy().astype(np.uint8))

        # Calculate IoU for each instance
        actual_labels= torch.tensor(actual_labels).to(device)
        total_modified_instance_iou = 0.0
        for label in range(labels-1):
            actual_instance_mask = (actual_labels == (label+1)).to(torch.uint8) #(label+1) to exclude bgd 0 

            intersection =(predicted_mask * actual_instance_mask).sum()
            actual_instance_mask_sum = (actual_instance_mask).sum()

            # IoU for this instance
            modified_instance_iou = (intersection / (actual_instance_mask_sum + 1e-6) ) # Add epsilon to avoid division by zero

            # Add IoU to total loss
            total_modified_instance_iou+= modified_instance_iou  # You can adjust the loss function as needed

        # Calculate average loss
        if labels > 1:
            avg_modified_instance_iou = total_modified_instance_iou / (labels-1)
        else:
            #avg_modefied_instance_iou = torch.tensor(0.0, device=actual_mask.device)
            avg_modified_instance_iou=0.0

        return avg_modified_instance_iou