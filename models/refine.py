import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentBoundaryAttention(nn.Module):
    """
    Document-aware boundary attention module
    Specifically designed for document shadow removal task, solving black edge problems
    """
    
    def __init__(self, in_channels, reduction=16):
        super(DocumentBoundaryAttention, self).__init__()
        self.in_channels = in_channels
        
        # 1. Boundary detection branch
        self.edge_detector = EdgeDetectionBranch(in_channels)
        
        # 2. Document region analysis branch
        self.document_analyzer = DocumentRegionBranch(in_channels)
        
        # 3. Boundary smoothing branch
        self.boundary_smoother = BoundarySmoothingBranch(in_channels, reduction)
        
        # 4. Feature fusion
        self.feature_fusion = FeatureFusionModule(in_channels)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            refined_x: [B, C, H, W] boundary-optimized features
        """
        # 1. Detect boundary regions
        edge_map = self.edge_detector(x)
        
        # 2. Analyze document regions (text vs background)
        region_map = self.document_analyzer(x)
        
        # 3. Generate boundary smoothing weights
        smooth_weights = self.boundary_smoother(x, edge_map, region_map)
        
        # 4. Apply boundary optimization
        refined_x = self.feature_fusion(x, smooth_weights, edge_map, region_map)
        
        return refined_x


class EdgeDetectionBranch(nn.Module):
    """Boundary detection branch - detect shadow boundaries"""
    
    def __init__(self, in_channels):
        super(EdgeDetectionBranch, self).__init__()
        
        # Learned edge detector
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Detect boundary regions
        Returns:
            edge_map: [B, 1, H, W] boundary probability map
        """
        edge_map = self.edge_conv(x)
        return edge_map


class DocumentRegionBranch(nn.Module):
    """Document region analysis branch - distinguish text regions from background regions"""
    
    def __init__(self, in_channels):
        super(DocumentRegionBranch, self).__init__()
        
        # Document region analysis network
        self.region_analyzer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 2, 1),  # 2 channels: text probability + background probability
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        Analyze document regions
        Returns:
            region_map: [B, 2, H, W] region probability map (text, background)
        """
        region_map = self.region_analyzer(x)
        return region_map


class BoundarySmoothingBranch(nn.Module):
    """Boundary smoothing branch - generate boundary smoothing weights"""
    
    def __init__(self, in_channels, reduction=16):
        super(BoundarySmoothingBranch, self).__init__()
        
        mid_channels = max(8, in_channels // reduction)
        
        # Boundary smoothing weight generator
        self.smooth_generator = nn.Sequential(
            nn.Conv2d(in_channels + 1 + 2, mid_channels, 3, padding=1),  # +1(edge) +2(region)
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_map, region_map):
        """
        Generate boundary smoothing weights
        Args:
            x: [B, C, H, W] input features
            edge_map: [B, 1, H, W] edge map
            region_map: [B, 2, H, W] region map
        Returns:
            smooth_weights: [B, C, H, W] smoothing weights
        """
        # Concatenate all information
        combined = torch.cat([x, edge_map, region_map], dim=1)
        
        # Generate smoothing weights
        smooth_weights = self.smooth_generator(combined)
        
        return smooth_weights


class FeatureFusionModule(nn.Module):
    """Feature fusion module - apply boundary optimization"""
    
    def __init__(self, in_channels):
        super(FeatureFusionModule, self).__init__()
        
        # Adaptive fusion weights
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
    def forward(self, x, smooth_weights, edge_map, region_map):
        """
        Apply boundary optimization
        Args:
            x: [B, C, H, W] original features
            smooth_weights: [B, C, H, W] smoothing weights
            edge_map: [B, 1, H, W] edge map
            region_map: [B, 2, H, W] region map
        Returns:
            refined_x: [B, C, H, W] optimized features
        """
        # Apply smoothing in boundary regions
        # Apply stronger smoothing in background regions (region_map[:, 1:2])
        background_mask = region_map[:, 1:2]  # [B, 1, H, W] background probability
        
        # Apply smoothing in boundary + background regions
        boundary_background = edge_map * background_mask
        
        # Generate smoothed features
        smoothed_x = x * smooth_weights
        
        # Adaptively fuse original and smoothed features
        # Use more smoothed features in boundary background regions, keep original features in text regions
        fusion_weight = boundary_background.expand_as(x)
        blended_x = x * (1 - fusion_weight) + smoothed_x * fusion_weight
        
        # Final fusion
        fused_features = torch.cat([x, blended_x], dim=1)
        refined_x = self.fusion_conv(fused_features)
        
        return refined_x


class DoubleConv(nn.Module):
    """Two 3×3 convolutions + BN + ReLU"""
    def __init__(self, in_ch, out_ch, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordAttention(nn.Module):
    """Coordinate Attention (Hou et al., 2021)"""
    def __init__(self, in_channels, reduction=32):
        super(CoordAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.size()
        x_h = F.adaptive_avg_pool2d(x, (H, 1))  # [B, C, H, 1]
        x_w = F.adaptive_avg_pool2d(x, (1, W))  # [B, C, 1, W]
        x_w = x_w.permute(0, 1, 3, 2)           # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)        # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))

        y_h, y_w = torch.split(y, [H, W], dim=2)
        attn_h = torch.sigmoid(self.conv_h(y_h))    # [B, C, H, 1]
        attn_w = torch.sigmoid(self.conv_w(y_w))    # [B, C, W, 1]
        attn_w = attn_w.permute(0, 1, 3, 2)         # [B, C, 1, W]

        return x * attn_h * attn_w


class Up(nn.Module):
    """Upsampling block + skip connection + DoubleConv in U-Net decoder"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DocumentAwareRefinementNetwork(nn.Module):
    """
    Document-Aware Refinement Network (DARN)
    U-Net + CoordAttention + DocumentBoundaryAttention version
    Input: [B, 5, H, W]
    Output: [B, 3, H, W]
    
    Document boundary attention module is integrated by default
    """
    def __init__(self, bilinear=True):
        super(DocumentAwareRefinementNetwork, self).__init__()

        # Encoding path
        self.enc1 = DoubleConv(5, 64)       # 5 -> 64
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)     # 64 -> 128
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = DoubleConv(128, 256)  # 128 -> 256

        # Decoding path
        self.up2 = Up(256 + 128, 128, bilinear)  # 256+128 -> 128
        self.ca2 = CoordAttention(128)
        
        # Document boundary attention module - applied after second decoding layer
        self.doc_boundary2 = DocumentBoundaryAttention(128)

        self.up1 = Up(128 + 64, 64, bilinear)   # 128+64 -> 64
        self.ca1 = CoordAttention(64)
        
        # Document boundary attention module - applied after final decoding layer (main boundary optimization)
        self.doc_boundary1 = DocumentBoundaryAttention(64)

        # Output
        self.out_conv = nn.Conv2d(64, 3, kernel_size=1)  # 64 -> 3

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)                     # [B, 64, H, W]
        e2 = self.enc2(self.pool1(e1))        # [B, 128, H/2, W/2]
        b = self.bottleneck(self.pool2(e2))   # [B, 256, H/4, W/4]

        # Decoding path + coordinate attention
        d2 = self.ca2(self.up2(b, e2))        # [B, 128, H/2, W/2]
        
        # Apply document boundary attention - second layer
        d2 = self.doc_boundary2(d2)           # boundary optimization
        
        d1 = self.ca1(self.up1(d2, e1))       # [B, 64, H, W]
        
        # Apply document boundary attention - final layer (main boundary processing)
        d1 = self.doc_boundary1(d1)           # main boundary optimization
        
        return self.out_conv(d1)              # [B, 3, H, W]


# ====================== Test ======================
if __name__ == '__main__':
    x = torch.randn(1, 5, 512, 512)  # Input: [B, 5, H, W]
    y = model(x)
    print(f'Input: {x.shape}')
    print(f'Output: {y.shape}')
