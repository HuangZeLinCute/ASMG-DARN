import os
import warnings

warnings.filterwarnings('ignore')

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional.regression import mean_squared_error
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data import get_data
from models import *
from utils import *


def test():
    opt = Config('config.yml')
    seed_everything(opt.OPTIM.SEED)

    accelerator = Accelerator()
    device = accelerator.device

    # LPIPS: automatically normalize to [-1,1]
    criterion_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=True
    ).to(device)

    # Data Loader
    val_dir = opt.TRAINING.VAL_DIR
    val_dataset = get_data(
        val_dir,
        opt.MODEL.INPUT,
        opt.MODEL.TARGET,
        'test',
        opt.TRAINING.ORI,
        {'w': opt.TRAINING.PS_W, 'h': opt.TRAINING.PS_H}
    )

    testloader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.TESTING.BATCH_SIZE,
        shuffle=False,
        num_workers=opt.TESTING.NUM_WORKERS,
        drop_last=False,
        pin_memory=True
    )

    # Model & Metrics
    model = Model()
    load_checkpoint(model, opt.TESTING.WEIGHT)

    model, testloader = accelerator.prepare(model, testloader)
    model.eval()

    size = len(testloader)
    stat_psnr = 0
    stat_ssim = 0
    stat_lpips = 0
    stat_rmse = 0

    for _, test_data in enumerate(tqdm(testloader)):
        # data: [targets, inputs, filename]
        inp = test_data[0].contiguous()
        gray = test_data[1].contiguous()
        tar = test_data[2]

        with torch.no_grad():
            res = model(gray, inp)

        # Force clip results to avoid exceeding [0,1] range
        res = torch.clamp(res, 0.0, 1.0)
        tar = torch.clamp(tar, 0.0, 1.0)

        if opt.TESTING.SAVE_IMAGES:
            os.makedirs("result", exist_ok=True)
            # Save each image in the batch
            for i in range(res.size(0)):
                save_image(res[i], os.path.join("result", test_data[3][i]))

        # Metrics calculation
        stat_psnr += peak_signal_noise_ratio(res, tar, data_range=1).item()
        stat_ssim += structural_similarity_index_measure(res, tar, data_range=1).item()
        stat_lpips += criterion_lpips(res, tar).item()
        stat_rmse += mean_squared_error(
            torch.mul(res, 255).flatten(),
            torch.mul(tar, 255).flatten(),
            squared=False
        ).item()

    stat_psnr /= size
    stat_ssim /= size
    stat_lpips /= size
    stat_rmse /= size

    print(f"RMSE: {stat_rmse:.4f}, PSNR: {stat_psnr:.4f}, SSIM: {stat_ssim:.4f}, LPIPS: {stat_lpips:.4f}")


if __name__ == '__main__':
    test()
