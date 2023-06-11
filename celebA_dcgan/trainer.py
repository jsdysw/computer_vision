import os

from torchvision import datasets, transforms, models
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from params import batch_size, DATA_DIR, d_conv_dim, g_conv_dim, z_size, beta1, lr, beta2, n_epochs, device, image_size
from utils import scale_images
from discriminator import Discriminator
from generator import Generator


def main():
    '''
    augmentions 
    1. Resizing the image to 32x32 as the smaller the image the faster we can train 
    2. Cropping from center with 32x32
    3. Changing type to tensor
    we won't be using normalize here as we will have to do that manually later for tanh activation function
    '''
    transform_img = transforms.Compose([
        transforms.Resize(image_size),  
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
     
    ds = datasets.ImageFolder(DATA_DIR, transform_img)
    train_dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers= 4, shuffle=True)
    print('number of train dataset : ', len(train_dataloader.dataset))


    # building discriminator and generator from the classes defined above
    discriminator = Discriminator(d_conv_dim).to(device)
    generator = Generator(z_size=z_size, conv_dim=g_conv_dim).to(device)

    print(discriminator)
    print(generator)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr, (beta1, beta2))
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr, (beta1, beta2)) 

    best_train_loss = None
    for epoch in range(n_epochs):
        discriminator.train()
        generator.train()
        for (real_images, _) in tqdm(train_dataloader):
            # Train the discriminator on real and fake images
            real_images = scale_images(real_images).to(device)

            discriminator_optimizer.zero_grad()
            
            D_real = discriminator(real_images)
            d_real_loss = real_loss(D_real, device)

            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
            fake_images = generator(z)
            
            D_fake = discriminator(fake_images)
            d_fake_loss = fake_loss(D_fake, device)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()
            discriminator_optimizer.step()     

            # 2. Train the generator with an adversarial loss
            generator_optimizer.zero_grad()
            
            z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            z = torch.from_numpy(z).float().to(device)
  
            fake_images = generator(z)
            D_fake = discriminator(fake_images)
            g_loss = real_loss(D_fake, device)
            g_loss.backward()
            generator_optimizer.step()

        print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch+1, n_epochs, d_loss.item(), g_loss.item()))

        # save least loss model
        if not best_train_loss or g_loss.item() < best_train_loss:
            print('model save')
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(generator.state_dict(), './snapshot/face-gan.pt')
            best_val_loss = g_loss.item()


def real_loss(D_out, device):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out, device):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss

if __name__ == '__main__':
   main()