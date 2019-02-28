import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DeepMattingVGG(nn.Module):
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg16(pretrained=True).features

        # (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): ReLU(inplace)
        # (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (3): ReLU(inplace)
        # (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (6): ReLU(inplace)
        # (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (8): ReLU(inplace)
        # (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (11): ReLU(inplace)
        # (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (13): ReLU(inplace)
        # (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (15): ReLU(inplace)
        # (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (18): ReLU(inplace)
        # (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (20): ReLU(inplace)
        # (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (22): ReLU(inplace)
        # (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (25): ReLU(inplace)
        # (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (27): ReLU(inplace)
        # (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (29): ReLU(inplace)
        # (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # Add fourth channel to initial conv for trimap
        pretrained_weights = self.vgg[0].weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1.weight.data.normal_(0, 0.001)
        self.conv1.weight.data[:, :3, :, :] = pretrained_weights

        self.conv2 = self.vgg[2]
        self.conv3 = self.vgg[5]
        self.conv4 = self.vgg[7]
        self.conv5 = self.vgg[10]
        self.conv6 = self.vgg[12]
        self.conv7 = self.vgg[14]
        self.conv8 = self.vgg[17]
        self.conv9 = self.vgg[19]
        self.conv10 = self.vgg[21]
        self.conv11 = self.vgg[24]
        self.conv12 = self.vgg[26]
        self.conv13 = self.vgg[28]

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.conv14 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Naming of layers corresponds to Figure 3 of Deep Image Matting Paper
        self.deconv6 = nn.Conv2d(512, 512, kernel_size=(1, 1))
        
        self.unpool5 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv5 = nn.Conv2d(512, 512, kernel_size=(5, 5), padding=2)

        self.unpool4 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv4 = nn.Conv2d(512, 256, kernel_size=(5, 5), padding=2)

        self.unpool3 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv3 = nn.Conv2d(256, 128, kernel_size=(5, 5), padding=2)

        self.unpool2 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv2 = nn.Conv2d(128, 64, kernel_size=(5, 5), padding=2)

        self.unpool1 = torch.nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding=2)
        
        self.deconv0 = nn.Conv2d(64, 1, kernel_size=(5, 5), padding=2)

    def forward(self, x):
        # Encoding portion
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x, idx1 = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x, idx2 = self.pool2(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x, idx3 = self.pool3(x)

        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x, idx4 = self.pool4(x)

        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x, idx5 = self.pool5(x)

        x = F.relu(self.conv14(x))

        # Decoding portion
        x = F.relu(self.deconv6(x))
        x = F.relu(self.deconv5(self.unpool5(x, idx5)))
        x = F.relu(self.deconv4(self.unpool4(x, idx4)))
        x = F.relu(self.deconv3(self.unpool3(x, idx3)))
        x = F.relu(self.deconv2(self.unpool2(x, idx2)))
        x = F.relu(self.deconv1(self.unpool1(x, idx1)))
        
        x = self.deconv0(x)

        return x