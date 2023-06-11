from torch import nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True, bias = False):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
        
    return nn.Sequential(*layers)


class Discriminator(nn.Module):
    def __init__(self, conv_dim):
        super().__init__()

        # complete init function
        self.conv_dim = conv_dim
        
        self.conv1 = conv(3, conv_dim, batch_norm=False)  
        self.conv2 = conv(conv_dim, conv_dim*2)           
        self.conv3 = conv(conv_dim*2, conv_dim*2*2)
        self.conv4 = conv(conv_dim*2*2, conv_dim*2*2*2)
        self.fc1 = nn.Linear(conv_dim*4*4*8, conv_dim*4*8)
        self.fc2 = nn.Linear(conv_dim*4*8, 1)

    def forward(self, x):
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(-1, self.conv_dim*4*4*8)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x