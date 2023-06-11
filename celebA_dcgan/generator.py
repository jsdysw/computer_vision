from torch import nn

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, bias = False):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)
import torch.nn.functional as F
from torch import nn

def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, bias = False):
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
    
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)

class Generator(nn.Module):
    def __init__(self, z_size, conv_dim):
        super().__init__()
        self.conv_dim = conv_dim
        
        self.fc1 = nn.Linear(z_size, conv_dim*4*8)
        self.fc2 = nn.Linear(conv_dim*4*8, conv_dim*4*4*8)

        self.de_conv1 = deconv(conv_dim*8, conv_dim*4)
        self.de_conv2 = deconv(conv_dim*4, conv_dim*2)
        self.de_conv3 = deconv(conv_dim*2, conv_dim)
        self.de_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = x.view(-1, self.conv_dim*8, 4, 4)
        x = F.relu(self.de_conv1(x))
        x = F.relu(self.de_conv2(x))
        x = F.relu(self.de_conv3(x))
        x = self.de_conv4(x)
        x = F.tanh(x)
        return x