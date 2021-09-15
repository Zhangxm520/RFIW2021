from torch.utils.data import Dataset
from keras.preprocessing import image
import numpy as np
import os
from Track3.utils import np2tensor


class Probe(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):
        self.sample_path=sample_path
        self.sample_list = self.load_sample()
        self.transform=transform

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                sample_list.append(line.split(' '))
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (0,3,1,2))

    def read_dir(self,dir):
        images=[]
        for image in os.listdir(dir):
            image_path=os.path.join(dir,image)
            image=self.read_image(image_path)
            if self.transform is not None:
                image=self.transform(image)
            images.append(np.array(image))
        return self.preprocess(np.array(images))

    def __getitem__(self, item):
        sample = self.sample_list[item]
        index=int(sample[0])
        imgs_array=self.read_dir(sample[1])
        return index,np2tensor(imgs_array)


class Gallery(Dataset):
    def __init__(self,
                 sample_path,
                 transform=None):
        self.sample_path = sample_path
        self.sample_list = self.load_sample()
        self.transform = transform

    def load_sample(self):
        sample_list = []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                sample_list.append(line.split(' '))
        f.close()
        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def read_image(self, path):
        img = image.load_img(path, target_size=(112, 112))
        return img

    def preprocess(self, img):
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, item):
        sample = self.sample_list[item]
        index = np2tensor(np.array(float(sample[0])))
        img = self.read_image(sample[1])
        if self.transform is not None:
            img = self.transform(img)
        img = np2tensor(self.preprocess(np.array(img, dtype=float)))
        return index ,img