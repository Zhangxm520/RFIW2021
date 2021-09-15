from keras.preprocessing import image
import numpy as np
import random

class FIW():
    def __init__(self,
                 sample_path,
                 transform=None):

        self.sample_path=sample_path
        self.transform=transform
        self.sample_list=self.load_sample()
        self.pairs_cur = 0
        self.pairs_len = len(self.sample_list)

    def load_sample(self):
        sample_list= []
        f = open(self.sample_path, "r+", encoding='utf-8')
        while True:
            line = f.readline().replace('\n', '')
            if not line:
                break
            else:
                tmp = line.split(' ')
                sample_list.append(tmp)
        f.close()
        return sample_list


    def __len__(self):
        return len(self.sample_list)


    def read_image(self,path):
        img = image.load_img(path, target_size=(112, 112))
        if self.transform is not None:
            img=self.transform(img)
        img = np.array(img).astype(np.float)
        return img

    def get_batch_pairs_image(self, batch_size):
        if self.pairs_cur + batch_size > self.pairs_len:
            end = self.pairs_len
        else:
            end = self.pairs_cur + batch_size
        now_pairs = self.sample_list[self.pairs_cur:end]
        father_img = np.array([self.read_image(p[0]) for p in now_pairs])
        mother_img = np.array([self.read_image(p[1]) for p in now_pairs])
        child_img  = np.array([self.read_image(p[2]) for p in now_pairs])
        labels =np.array([float(p[-1]) for p in now_pairs])
        self.pairs_cur = end
        if self.pairs_cur == self.pairs_len:
            self.pairs_cur=0
        return father_img,mother_img,child_img,labels


    def get_val_on_batch(self, list_tuples, batch_size=30):
        total_len = len(list_tuples)
        start = 0
        while True:
            if start + batch_size > total_len:
                end = total_len
            else:
                end = start + batch_size
            batch_list = list_tuples[start:end]
            datas = []
            labels = []
            for now in batch_list:
                datas.append([now[0], now[1],now[2]])
                labels.append(float(now[-1]))

            father_img = np.array([self.read_image(x[0]) for x in datas])
            mother_img = np.array([self.read_image(x[1]) for x in datas])
            child_img = np.array([self.read_image(x[2]) for x in datas])
            yield father_img, mother_img,child_img ,labels
            start = end
            if start == total_len:
                yield None, None, None,None


