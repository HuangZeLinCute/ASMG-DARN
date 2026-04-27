import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from models.mask import AdaptiveShadowMaskGenerator
from models.refine import DocumentAwareRefinementNetwork


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mask = AdaptiveShadowMaskGenerator()

        self.refine = DocumentAwareRefinementNetwork(bilinear=True)

    def forward(self, bin_x, x):
        """
        Args:
            bin_x : [B, 1, H, W]   # input grayscale image (for threshold segmentation)
            x     : [B, 3, H, W]   # input original RGB image

        Returns:
            res   : [B, 3, H, W]   # output restored/enhanced image
        """
        mask = self.mask(bin_x)

        x_res = torch.cat((mask, x), dim=1)

        res = self.refine(x_res)    # [B, 5, H, W] -> [B, 3, H, W]

        return res


if __name__ == '__main__':
    # Grayscale image (bin_x), single channel
    bin_x = torch.randn(1, 1, 512, 512).cuda()  # [B, 1, H, W]

    # Color image (x)
    x = torch.randn(1, 3, 512, 512).cuda()  # [B, 3, H, W]

    model = Model().cuda()
    output = model(bin_x, x)

    print(f'bin_x size: {bin_x.size()}')        # [1, 1, 512, 512]
    print(f'x size: {x.size()}')                # [1, 3, 512, 512]
    print(f'output size: {output.size()}')      # [1, 3, 512, 512]

    # model = Model().cuda()
    # img = Image.open('test.jpg').convert('RGB')
    # img = TF.to_tensor(img).cuda()
    # img = TF.resize(img, (512, 512)).unsqueeze(0)
    # g_img = TF.rgb_to_grayscale(img)
    # out = model(g_img, img)
    # print(out.shape)
