import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

imgpath='/public/zebanghe2/cycledomain/dataset/reversemix'
transpath='/public/zebanghe2/cycledomain/dataset/transmission'
maskpath='/public/zebanghe2/cycledomain/dataset/reverseref'
gmixpath='/public/zebanghe2/cycledomain/dataset/globalmix'
grefpath='/public/zebanghe2/cycledomain/dataset/globalref'
class ImageDataset(Dataset):
    def __init__(self, transforms_=None, mode="train"):
        self.transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
])
        self.transforms2 = transforms.Compose([transforms.Resize((384, 384)),transforms.Grayscale(),transforms.ToTensor()])
        g=os.walk(imgpath)
        self.name=[]
        for path, dir_list, file_list in g:
           for file_name in file_list:
             self.name.append(file_name.split('.')[0])

    def __getitem__(self, index):

        img = Image.open(imgpath+'/'+self.name[index]+'.jpg')
        trans= Image.open(transpath+'/'+self.name[index]+'.jpg')
        mask= Image.open(maskpath+'/'+self.name[index]+'.jpg')
        gmix = Image.open(gmixpath+'/'+self.name[index]+'.jpg')
        gref= Image.open(grefpath+'/'+self.name[index]+'.jpg')
        w, h = img.size
        #print(img.size)
        #print(mask.shape)
        #img_A = img.crop((0, 0, w / 2, h))
        #img_B = img.crop((w / 2, 0, w, h))

        #if np.random.random() < 0.5:
            #img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            #img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        imgs = self.transform(img)
        trans=self.transform(trans)
        masks = self.transform(mask)
        gmix=self.transform(gmix)
        gref=self.transform(gref)

        total={"img": imgs, "trans":trans, "mask": masks, "gmix": gmix, "gref": gref}
        return total

    def __len__(self):
        return len(self.name)
