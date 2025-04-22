import os
import os.path
import glob
from PIL import Image

import numpy as np

import torch
from torchvision import transforms
import utils

def load_SingleVideo(data, noisy_path, batch_size=8,image_size=None, stride=64, n_frames=5,
                     aug=0):

    train_dataset = SingleVideo(data, noisy_path, patch_size=image_size, stride=stride, n_frames=n_frames,
                            aug=aug
                                )
    test_dataset = SingleVideo(data, noisy_path, patch_size=None, stride=stride, n_frames=n_frames,
                               aug=0
                              )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False)
    return train_loader, test_loader

class SingleVideo(torch.utils.data.Dataset):
    def __init__(self, data_path, noisy_path, patch_size=None, stride=64, n_frames=5, aug=0):
        super().__init__()
        self.data_path = data_path
        self.noisy_path = noisy_path
        self.size = patch_size
        self.stride = stride
        self.n_frames = n_frames
        self.aug = aug

        self.files = sorted(glob.glob(os.path.join(data_path, "*.png")))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_path, "*.png")))

        self.len = self.bound = len(self.files)

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.reverse = transforms.Compose([transforms.ToPILImage()])

        Img = Image.open(self.files[0])
        Img = np.array(Img)
        H, W, C = Img.shape

        if self.size is not None:
            self.n_H = (int((H-self.size)/self.stride)+1)
            self.n_W = (int((W-self.size)/self.stride)+1)
            self.n_patches = self.n_H * self.n_W
            self.len *= self.n_patches

        self.hflip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.vflip = transforms.Compose([transforms.RandomVerticalFlip(p=1)])

        if aug >= 1: # Horizonatal and Vertical Flips
            self.len *= 4
        if aug >= 2: # Reverse the Video
            self.len *= 2
        if aug >= 3: # Variable Frame Rate
            self.len *= 4

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        hop = 1
        reverse = 0
        flip = 0
        if self.aug >= 3: # Variable Frame Rate
            hop = index % 4 + 1
            index = index // 4
        if self.aug >= 2: # Reverse the Video
            reverse = index % 2
            index = index // 2
        if self.aug >= 1: # Horizonatal and Vertical Flips
            flip = index % 4
            index = index // 4

        if self.size is not None:
            patch = index % self.n_patches
            index = index // self.n_patches

        ends = 0
        x = ((self.n_frames-1) // 2)*hop
        if index < x:
            ends = x - index
        elif self.bound-1-index < x:
            ends = -(x-(self.bound-1-index))

        Img = Image.open(self.files[index])
        Img = np.array(Img)
        noisy_Img = Image.open(self.noisy_files[index])
        noisy_Img = np.array(noisy_Img)

        for i in range(hop, x+1, hop):
            end = max(0, ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index-i+off])
            img = np.array(img)
            noisy_img = Image.open(self.noisy_files[index-i+off])
            noisy_img = np.array(noisy_img)
            if reverse == 0:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)
            else:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)

        for i in range(hop, x+1, hop):
            end = -min(0,ends)
            off = max(0,i-x+end)
            img = Image.open(self.files[index+i-off])
            img = np.array(img)
            noisy_img = Image.open(self.noisy_files[index+i-off])
            noisy_img = np.array(noisy_img)
            if reverse == 0:
                Img = np.concatenate((Img, img), axis=2)
                noisy_Img = np.concatenate((noisy_Img, noisy_img), axis=2)
            else:
                Img = np.concatenate((img, Img), axis=2)
                noisy_Img = np.concatenate((noisy_img, noisy_Img), axis=2)

        if self.size is not None:
            nh = (patch // self.n_W)*self.stride
            nw = (patch % self.n_W)*self.stride
            Img = Img[nh:(nh+self.size), nw:(nw+self.size), :]
            noisy_Img = noisy_Img[nh:(nh+self.size), nw:(nw+self.size), :]
        if flip == 1:
            Img = np.flip(Img, 1)
            noisy_Img = np.flip(noisy_Img, 1)
        elif flip == 2:
            Img = np.flip(Img, 0)
            noisy_Img = np.flip(noisy_Img, 0)
        elif flip == 3:
            Img = np.flip(Img, (1,0))
            noisy_Img = np.flip(noisy_Img, (0,1))
        return self.transform(np.array(Img)).type(torch.FloatTensor), self.transform(np.array(noisy_Img)).type(torch.FloatTensor)