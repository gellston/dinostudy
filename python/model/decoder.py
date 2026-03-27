import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


from utils.helper import LayerNorm2d

class UNetBlock(nn.Module):
    def __init__(self, cin, cout):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=4, stride=2, padding=1, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False), 
            LayerNorm2d(num_channels=cin, eps=1e-6),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), 
            LayerNorm2d(num_channels=cout, eps=1e-6)
        )
    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)



class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        embed_dims=[320, 160, 80, 40]
    ):
        super().__init__()

        self.decoder1 = UNetBlock(cin=embed_dims[0], cout=embed_dims[1])
        self.decoder2 = UNetBlock(cin=embed_dims[1], cout=embed_dims[2])
        self.decoder3 = UNetBlock(cin=embed_dims[2], cout=embed_dims[3])

        self.projection = nn.Conv2d(embed_dims[-1], out_channels, kernel_size=1, stride=1, bias=True)

    def forward(self, x):


        feature1 = x[-1]
        feature2 = x[-2]
        feature3 = x[-3]
        feature4 = x[-4]


        y = self.decoder1(feature1) + feature2
        y = self.decoder2(y) + feature3
        y = self.decoder3(y) + feature4

        y = self.projection(y)

        y = torch.sigmoid(y)
        out = F.interpolate(y, scale_factor=(4.0, 4.0), mode='bilinear', align_corners=False)
        
        return out
