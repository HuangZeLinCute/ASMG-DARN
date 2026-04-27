import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
import torchvision.models as models



class ReconstructionLoss(nn.Module):
    """
    Simplified image reconstruction loss function - includes only MSE and SSIM
    Contains:
    - MSE Loss (basic reconstruction)
    - SSIM Loss (structural similarity)

    Supports dynamic weight adjustment
    """
    def __init__(self, 
                 mse_weight: float = 1.0,
                 ssim_weight: float = 0.2):
        """
        Args:
            mse_weight: MSE loss weight
            ssim_weight: SSIM loss weight
        """
        super(ReconstructionLoss, self).__init__()
        self.mse = nn.MSELoss()
        
        # Initial weights - only use MSE and SSIM
        self.weights = {
            'mse': mse_weight,
            'ssim': ssim_weight
        }
    
    def update_weights(self, new_weights: dict):
        """
        Update loss weights
        
        Args:
            new_weights: new weight dictionary, e.g. {'ssim': 0.3, 'edge': 0.5}
        """
        for loss_name, weight in new_weights.items():
            if loss_name in self.weights:
                self.weights[loss_name] = weight
    
    def get_weights(self) -> dict:
        """Get current weights"""
        return self.weights.copy()

    def forward(self, pred, target, shadow_mask=None):
        """
        Args:
            pred: [B, 3, H, W] predicted image
            target: [B, 3, H, W] target image
            shadow_mask: [B, 1, H, W] shadow mask (optional)
        """
        # Basic loss
        loss_mse = self.mse(pred, target)
        
        # Structural similarity loss
        loss_ssim = 1 - structural_similarity_index_measure(pred, target, data_range=1.0)
        
        # Calculate total loss using dynamic weights
        total_loss = (self.weights['mse'] * loss_mse + 
                     self.weights['ssim'] * loss_ssim)

        return total_loss, loss_mse, loss_ssim


if __name__ == '__main__':
    # Assume input is batch=2, 3-channel RGB image, 256x256
    pred = torch.rand((2, 3, 512, 512))  # model output
    target = torch.rand((2, 3, 512, 512))  # GT image

    # Define Loss
    criterion = ReconstructionLoss(ssim_weight=0.2)

    # Forward pass
    loss, loss_mse, loss_ssim = criterion(pred, target)

    print("Total Loss:", loss.item())
    print("MSE Loss:", loss_mse.item())
    print("SSIM Loss:", loss_ssim.item())