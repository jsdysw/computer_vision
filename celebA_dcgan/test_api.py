from torchvision.utils import save_image
import numpy as np
import torch

from generator import Generator
from params import z_size, g_conv_dim

def main():

    batch_size = 1
    z = np.random.uniform(-1, 1, size=(batch_size, z_size))
    z = torch.from_numpy(z).float()
    
    print('load model ')
    generator = Generator(z_size=z_size, conv_dim=g_conv_dim)
    generator.load_state_dict(torch.load('./snapshot/face-gan.pt'))
   
    print('inference ')
    samples_z = generator(z)
    print('image shape : ', samples_z.shape)
    img = samples_z[0]
    save_image(img, 'face-gan-result.png')


if __name__ == '__main__':
   main()
