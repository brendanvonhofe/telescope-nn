import torch
import torch.nn as nn

class MatteRefinementLayer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(4, 64, (3,3), padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 64, (3,3), padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(64, 1, (3,3), padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.relu3(out)
        
        out = self.conv4(out)
        out = out + x[:, 3, :, :].unsqueeze(1)
        
        return out