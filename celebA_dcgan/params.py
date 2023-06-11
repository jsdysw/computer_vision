import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

image_size = 64

DATA_DIR = 'img_align_celeba/'
batch_size = 256
lr = 0.0002 
n_epochs = 20
beta1=0.5
beta2=0.999

d_conv_dim = 64
g_conv_dim = 64
z_size = 100