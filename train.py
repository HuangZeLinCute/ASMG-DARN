import warnings
import torch
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.functional.regression import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from config import Config
from data import get_data
from models import *
from utils import *
from utils import losses

warnings.filterwarnings('ignore')


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def manage_best_checkpoints(best_checkpoints, current_rmse, current_epoch, model_session, save_dir, max_keep=2):
    """
    Manage the best checkpoint files, keeping only the top two
    Args:
        best_checkpoints: current best checkpoints list [(rmse, epoch, filepath), ...]
        current_rmse: current epoch RMSE
        current_epoch: current epoch
        model_session: model session name
        save_dir: save directory
        max_keep: maximum number of checkpoints to keep
    Returns:
        updated_best_checkpoints: updated best checkpoints list
        should_save: whether current checkpoint should be saved
    """
    should_save = False
    
    # If maximum keep limit not reached, add directly
    if len(best_checkpoints) < max_keep:
        should_save = True
    else:
        # Find current worst checkpoint
        worst_rmse = max(best_checkpoints, key=lambda x: x[0])[0]
        if current_rmse < worst_rmse:
            should_save = True
    
    if should_save:
        # Generate new checkpoint file path
        new_checkpoint_path = os.path.join(save_dir, f"{model_session}_epoch_{current_epoch}.pth")
        
        # Add to list
        best_checkpoints.append((current_rmse, current_epoch, new_checkpoint_path))
        
        # Sort by RMSE (ascending)
        best_checkpoints.sort(key=lambda x: x[0])
        
        # If exceeding max retention count, delete the worst
        while len(best_checkpoints) > max_keep:
            worst_checkpoint = best_checkpoints.pop()  # Remove the last one (worst)
            worst_filepath = worst_checkpoint[2]
            
            # Delete file
            if os.path.exists(worst_filepath):
                try:
                    os.remove(worst_filepath)
                    print(f"Deleting poor weight file: {os.path.basename(worst_filepath)} (RMSE: {worst_checkpoint[0]:.4f})")
                except Exception as e:
                    print(f"Failed to delete file {worst_filepath}: {e}")
        
        # Print currently retained best checkpoints
        print(f"Currently retained best weight files:")
        for i, (rmse, epoch, filepath) in enumerate(best_checkpoints):
            print(f"   {i+1}. {os.path.basename(filepath)} - RMSE: {rmse:.4f} (Epoch {epoch})")
    
    return best_checkpoints, should_save


def plot_metrics(epochs, psnr_list, ssim_list, rmse_list, save_dir="."):
    """
    Plot training metrics:
    - PSNR + RMSE in one figure
    - SSIM in separate figure
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Figure 1: PSNR + RMSE ----------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: PSNR
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PSNR ↑", color='tab:blue')
    ax1.plot(epochs, psnr_list, 'o-', color='tab:blue', label='PSNR ↑')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Right axis: RMSE
    ax2 = ax1.twinx()
    ax2.set_ylabel("RMSE ↓", color='tab:red')
    ax2.plot(epochs, rmse_list, '^-', color='tab:red', label='RMSE ↓')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Merge legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    plt.title("Training Metrics (PSNR & RMSE)")
    fig.tight_layout()
    save_path1 = os.path.join(save_dir, "metrics_psnr_rmse.png")
    plt.savefig(save_path1)
    plt.close()
    print(f"PSNR+RMSE figure saved to {save_path1}")

    # ---------------- Figure 2: SSIM ----------------
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, ssim_list, 's-', color='tab:orange', label='SSIM ↑')
    plt.xlabel("Epoch")
    plt.ylabel("SSIM ↑")
    plt.title("Training Metrics (SSIM)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    save_path2 = os.path.join(save_dir, "metrics_ssim.png")
    plt.savefig(save_path2)
    plt.close()
    print(f"SSIM figure saved to {save_path2}")


def train():
    # Configuration
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    # Initialize Accelerator
    if getattr(opt.OPTIM, "WANDB", False):
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(project_name=getattr(opt.OPTIM, "WANDB_PROJECT", "default_project"))
    else:
        accelerator = Accelerator()

    if accelerator.is_local_main_process:
        os.makedirs(opt.TRAINING.SAVE_DIR, exist_ok=True)

    # Data loading
    train_dataset = get_data(opt.TRAINING.TRAIN_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'train', opt.TRAINING.ORI,
                             {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=opt.TRAINING.NUM_WORKERS,
        drop_last=True,
        pin_memory=True
    )

    val_dataset = get_data(opt.TRAINING.VAL_DIR, opt.MODEL.INPUT, opt.MODEL.TARGET, 'test', opt.TRAINING.ORI,
                           {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H})
    testloader = DataLoader(dataset=val_dataset, batch_size=opt.TRAINING.VAL_BATCH_SIZE, shuffle=False,
                            num_workers=opt.TRAINING.VAL_NUM_WORKERS, drop_last=False, pin_memory=True)

    # Model and loss
    model = Model()
    print(f"Using complete model (ASMG + DARN + DocumentBoundaryAttention)")
    
    # Get loss configuration from config
    loss_config = opt.get_loss_config()
    
    print(f"Using loss configuration: {opt.LOSS.TYPE}")
    print(f"Configuration parameters: {loss_config}")
    
    # Use simplified loss function
    criterion = losses.ReconstructionLoss(**loss_config)
    
    # Learning rate configuration
    initial_lr = opt.OPTIM.LR_INITIAL
    min_lr = opt.OPTIM.LR_MIN
    print(f"Training mode: using standard learning rate LR={initial_lr:.2e}, MIN_LR={min_lr:.2e}")
    
    optimizer_b = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999),
                              eps=1e-8, weight_decay=0.01)
    scheduler_b = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_b, opt.OPTIM.NUM_EPOCHS, eta_min=min_lr
    )

    start_epoch = 1
    best_epoch = 1
    best_rmse = 100

    # Early stopping parameters
    patience = getattr(opt.TRAINING, "PATIENCE", 20)
    patience_counter = 0
    
    # Keep the best two weight files
    best_checkpoints = []  # Store the best two checkpoint information [(rmse, epoch, filepath), ...]

    # checkpoint
    resume_path = getattr(opt.TRAINING, "RESUME", None)
    if resume_path and os.path.exists(resume_path):
        print(f"=> Loading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")

        state_dict = checkpoint['state_dict']
        # Remove "module." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[len("module."):]] = v
            else:
                new_state_dict[k] = v

        # Load modified state_dict
        model.load_state_dict(new_state_dict)

        if 'optimizer_state_dict' in checkpoint:
            optimizer_b.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_rmse = checkpoint.get('best_rmse', best_rmse)
            best_epoch = checkpoint.get('best_epoch', best_epoch)
            
            print(f"=> Continuing training from epoch {start_epoch} (best_rmse={best_rmse:.4f}, best_epoch={best_epoch})")
        else:
            print("=> Only found model parameters, will train optimizer from scratch in fine-tune mode")
    else:
        print("=> No checkpoint found, will train from scratch")

    # prepare
    trainloader, testloader = accelerator.prepare(trainloader, testloader)
    model = accelerator.prepare(model)
    optimizer_b, scheduler_b = accelerator.prepare(optimizer_b, scheduler_b)

    size = len(testloader)

    # Record metrics
    epoch_list, psnr_list, ssim_list, rmse_list = [], [], [], []
    
    # Normal training mode: model trains and updates weights normally
    print(f"Normal training mode: model trains and updates weights normally")
    
    # Create or open RMSE record file
    rmse_log_path = os.path.join(opt.TRAINING.SAVE_DIR, f"{opt.MODEL.SESSION}_rmse_log.txt")
    if accelerator.is_local_main_process:
        # If resuming training, use append mode; otherwise create new file
        log_mode = 'a' if (resume_path and os.path.exists(resume_path)) else 'w'
        with open(rmse_log_path, log_mode) as f:
            if log_mode == 'w':
                f.write("Epoch,PSNR,SSIM,RMSE\n")  # Write header
            else:
                f.write(f"\n# Resumed training from epoch {start_epoch}\n")
        print(f"RMSE log will be saved to: {rmse_log_path}")

    try:
        for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
            # Normal training
            model.train()
            for _, data in enumerate(tqdm(trainloader, disable=not accelerator.is_local_main_process)):
                inp, gray, tar = data[0].contiguous(), data[1].contiguous(), data[2]
                optimizer_b.zero_grad()

                # forward
                res = model(gray, inp)

                # Use simplified loss function (only includes MSE and SSIM)
                train_loss, loss_mse, loss_ssim = criterion(res, tar)

                # backward
                accelerator.backward(train_loss)
                optimizer_b.step()
            
            scheduler_b.step()

            # Validation
            if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
                model.eval()
                psnr, ssim, rmse = 0, 0, 0
                for _, data in enumerate(tqdm(testloader, disable=not accelerator.is_local_main_process)):
                    inp, gray, tar = data[0].contiguous(), data[1].contiguous(), data[2]
                    with torch.no_grad():
                        res = model(gray, inp)
                    res, tar = accelerator.gather((res, tar))
                    psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
                    ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
                    rmse += mean_squared_error(torch.mul(res, 255).flatten(),
                                               torch.mul(tar, 255).flatten(),
                                               squared=False).item()
                psnr /= size
                ssim /= size
                rmse /= size

                # Record
                epoch_list.append(epoch)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                rmse_list.append(rmse)

                # Manage best weight files (keep only the best two)
                best_checkpoints, should_save = manage_best_checkpoints(
                    best_checkpoints, rmse, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR, max_keep=2
                )
                
                # Save model & early stopping
                if should_save:
                    checkpoint_data = {
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer_b.state_dict(),
                        'epoch': epoch,
                        'best_rmse': min(rmse, best_rmse),
                        'best_epoch': epoch if rmse < best_rmse else best_epoch,
                    }
                    
                    save_checkpoint(checkpoint_data, epoch, opt.MODEL.SESSION, opt.TRAINING.SAVE_DIR)
                
                # Update best record
                if rmse < best_rmse:
                    best_epoch = epoch
                    best_rmse = rmse
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping triggered! No improvement for {patience} consecutive validation cycles (best RMSE={best_rmse:.4f}, best epoch={best_epoch})")
                        raise KeyboardInterrupt

                # Log (only execute when wandb is enabled)
                if getattr(opt.OPTIM, "WANDB", False):
                    accelerator.log({"PSNR": psnr, "SSIM": ssim, "RMSE": rmse}, step=epoch)

                # Console output
                if accelerator.is_local_main_process:
                    output_str = (f"epoch: {epoch}, RMSE:{rmse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, "
                                f"best RMSE: {best_rmse:.4f}, best epoch: {best_epoch}")
                    print(output_str)
                    
                    # Record RMSE to file
                    with open(rmse_log_path, 'a') as f:
                        f.write(f"{epoch},{psnr:.6f},{ssim:.6f},{rmse:.6f}\n")

    except KeyboardInterrupt:
        print("\n==> Training stopped, starting to save metric plots ...")

    # Display final retained weight files
    if accelerator.is_local_main_process and len(best_checkpoints) > 0:
        print(f"\nTraining completed! Final retained best weight files:")
        for i, (rmse, epoch, filepath) in enumerate(best_checkpoints):
            status = "🥇 Best" if i == 0 else "🥈 Second best"
            print(f"   {status}: {os.path.basename(filepath)} - RMSE: {rmse:.4f} (Epoch {epoch})")
        
        # Recommend using best weights for testing
        if len(best_checkpoints) > 0:
            best_checkpoint_path = best_checkpoints[0][2]
            print(f"\nRecommend using best weights for testing:")
            print(f"   python test.py TESTING.WEIGHT \"{best_checkpoint_path}\"")

    # Plot metric figures
    if len(epoch_list) > 0:
        plot_metrics(epoch_list, psnr_list, ssim_list, rmse_list,
             save_dir=opt.TRAINING.SAVE_DIR)

    accelerator.end_training()


if __name__ == '__main__':
    train()
