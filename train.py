#! /usr/bin/env python

import argparse
import os
import numpy as np
import math
import itertools
from PIL import Image
import sys
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision
from tqdm import tqdm

import random

torch.cuda.empty_cache()

from dataset import ImageDataset
from network import NLayerDiscriminator,unet256,unet512,discritic
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn.functional as F

def tensor2im(input_image, imtype=np.uint8):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225) 
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path,h,w):
    im_grid = im
    im_numpy = tensor2im(im_grid) 
    im_array = Image.fromarray(im_numpy)
    im_array=im_array.resize((h,w))
    im_array.save(path)

trainloader = DataLoader(
    ImageDataset(transforms_=None),
    batch_size=8,
    shuffle=False,drop_last=True
)
writer = SummaryWriter(log_dir="/public/zebanghe2/20200701/patchGAN/log/", comment="patchGAN")
print("data length:",len(trainloader))
#train_iterator = tqdm(trainloader, total=len(trainloader))
gen=unet256()
#gen=unet512()
crit=discritic()
d1=NLayerDiscriminator(3)
d2=NLayerDiscriminator(3)
gen=nn.DataParallel(gen)
crit=nn.DataParallel(crit)
d1=nn.DataParallel(d1)
d2=nn.DataParallel(d2)
criterion_MSE=torch.nn.MSELoss()
criterion_L1=torch.nn.L1Loss()

gen.load_state_dict(torch.load("/public/zebanghe2/20200701/patchGAN/generatorMSE_ 79MSE.pth").state_dict())
crit.load_state_dict(torch.load("/public/zebanghe2/20200701/patchGAN/cricticsMSE_ 79MSE.pth").state_dict())
d1.load_state_dict(torch.load("/public/zebanghe2/20200701/patchGAN/discriminator1MSE_ 79MSE.pth").state_dict())


criterion_MSE.cuda()
criterion_L1.cuda()

optimizer_G1 = torch.optim.Adam(gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_C1 = torch.optim.Adam(crit.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D1 = torch.optim.Adam(d1.parameters(), lr=0.0001, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(d2.parameters(), lr=0.0001, betas=(0.5, 0.999))

def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

eopchnum=200
print("start training")

for epoch in range(81, eopchnum+1):
  print("epoch:",epoch)
  iteration=0
  train_iterator = tqdm(trainloader, total=len(trainloader))
  if epoch>100:
    adjust_learning_rate(optimizer_G1)
    adjust_learning_rate(optimizer_C1)
    adjust_learning_rate(optimizer_D1)
    adjust_learning_rate(optimizer_D2)
  if(epoch%10==9):
    torch.save(gen,"generatorMSE_%3dMSE.pth"%epoch)
    torch.save(crit,'cricticsMSE_%3dMSE.pth'%epoch)
    torch.save(d1,"discriminator1MSE_%3dMSE.pth"%epoch)
    #torch.save(d2,'discriminator2MSE_%3d.pth'%epoch)
  
  for total in train_iterator:
    gen.train()
    gen.cuda()
    crit.train()
    crit.cuda()
    d1.train()
    d1.cuda()
    d2.train()
    d2.cuda()
    
    iteration=iteration+1
    optimizer_G1.zero_grad()
    optimizer_C1.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()
    
    # Model inputs
    real_img = total["img"]
    real_trans = total["trans"]
    real_mask = total["mask"]#actually reflection
    gmix=total["gmix"]
    gref=total["gref"]
    real_img=real_img.cuda()
    real_mask=real_mask.cuda()
    real_trans=real_trans.cuda()
    gmix=gmix.cuda()
    gref=gref.cuda()
    marks=1
    if np.any(np.isnan(real_img.cpu().numpy())):
       print('Input data has NaN!')
    
    
    
    
    if epoch<30:
      out1,out2=gen(gmix)
      if iteration==35:
        fk_B=tensor2im(out1.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSE"+'_trans'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out2.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSE"+'_ref'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out1.cpu()[2,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSEn2"+'_trans'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out2.cpu()[2,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSEh2"+'_ref'+str(epoch)+'.jpg',384,384)
      #cross
      #d11=criterion_L1(out1,real_trans)
      #d12=criterion_L1(out1,real_mask)
      
      #d21=criterion_L1(out2,real_trans)
      #d22=criterion_L1(out2,real_mask)
      
      d11=criterion_MSE(out1,real_trans)
      d12=criterion_MSE(out1,gref)
      
      d21=criterion_MSE(out2,real_trans)
      d22=criterion_MSE(out2,gref)
      
      """
      real_cat=torch.cat([real_trans,real_mask],1)
      if(d11+d22<d21+d12):
        fake_cat=torch.cat([out1,out2],1)
      else:
        fake_cat=torch.cat([out2,out1],1)
      temp_cat=torch.cat([gmix,real_img],1)
      """
      real_dis1=d1(real_trans)
      if(d11+d22<d12+d21):
        fake_dis1=d1(gmix.detach())
      else:
        fake_dis1=d1(gmix.detach())
      
      
      valid=torch.ones(real_dis1.shape).cuda()
      fake=torch.zeros(real_dis1.shape).cuda()
      
      lossd1=criterion_MSE(real_dis1,valid)+criterion_MSE(fake_dis1,fake)+2
      #lossd1=Variable(lossd1,requires_grad=True)
      lossd1.backward()
      #for name, weight in d1.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
            # print(name,weight.grad)
      optimizer_D1.step()
      
      #real_dis2=d2(real_mask)
      #temp_dis2=d2(gmix)
      #fake_dis2=d2(out2.detach())
      
      
      #lossd2=criterion_MSE(real_dis2,valid)+criterion_MSE(fake_dis2,fake)+2
      #lossd2=Variable(lossd2,requires_grad=True)
      #lossd2.backward()
      #optimizer_D2.step()
      if(d11+d22<d12+d21):
        fake_dis1=d1(out1.detach())
      else:
        fake_dis1=d1(out2.detach())
      lossd1=criterion_MSE(fake_dis1,valid)
      lossg=torch.min(d11+d22,d12+d21)+lossd1
      
      print(d11+d22)
      print(d12+d21)
      if(d11+d22>d12+d21):
        marks=2

      #lossg=Variable(lossg,requires_grad=True)
      #print(lossg.grad)
      #train_iterator.set_description("batch:%3d,iteration:%3d,loss_g:%3f"%(epoch+1,iteration,lossg.item()))
      lossg.backward()
      #for name, weight in gen.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
             #print(name,weight.grad.mean)
      optimizer_G1.step()
      #print([x.grad for x in optimizer_G1.param_groups[0]['params']])
      
      train_iterator.set_description("batch:%3d,iteration:%3d,loss_g:%3f"%(epoch+1,iteration,lossg.item()))
    else:
      out1,out2=gen(real_img)
      if iteration==35:
        fk_B=tensor2im(out1.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSE"+'_trans'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out2.cpu()[0,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSE"+'_ref'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out1.cpu()[2,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSEn2"+'_trans'+str(epoch)+'.jpg',384,384)
        fk_B=tensor2im(out2.cpu()[2,:,:,:])
        save_img(fk_B,"/public/zebanghe2/20200701/patchGAN/sample/"+"MSEh2"+'_ref'+str(epoch)+'.jpg',384,384)
      d11=criterion_L1(out1,real_trans)+criterion_MSE(out1,real_trans)
      d12=criterion_L1(out1,real_mask)+criterion_MSE(out1,real_mask)
      
      d21=criterion_L1(out2,real_trans)+criterion_MSE(out2,real_trans)
      d22=criterion_L1(out2,real_mask)+criterion_MSE(out2,real_mask)
      print(d11+d22)
      print(d12+d21)
      if(d11+d22>d12+d21):
        marks=2
      
      if(d11+d22<d12+d21):
        fake_cat=torch.cat([out1,out2],1)
      else:
        fake_cat=torch.cat([out2,out1],1)
      real_cat=torch.cat([real_trans,real_mask],1)
      temp_cat=torch.cat([gmix,real_img],1)
      
      real_cric=crit(real_cat.detach())
      fake_cric=crit(fake_cat.detach())
      temp_cric=crit(temp_cat.detach())
      cricmix=torch.ones(real_cric.shape).cuda()
      cricclean=torch.zeros(real_cric.shape).cuda()
      
      lossc=(criterion_MSE(real_cric,cricclean)+criterion_MSE(fake_cric,cricmix)+criterion_MSE(temp_cric,cricmix))+2
      #lossc=Variable(lossc,requires_grad=True)
      lossc.backward()
      optimizer_C1.step()
      
      real_dis1=d1(real_trans)
      temp_dis1=d1(gmix)
      
      if(d11+d22<d12+d21):
        fake_dis1=d1(out1.detach())
      else:
        fake_dis1=d1(out2.detach())
      
      valid=torch.ones(real_dis1.shape).cuda()
      fake=torch.zeros(real_dis1.shape).cuda()
      
      lossd1=criterion_MSE(real_dis1,valid)+criterion_MSE(fake_dis1,fake)+2
      #lossd1=Variable(lossd1,requires_grad=True)
      lossd1.backward()
      optimizer_D1.step()
      
      
      #cross
      
      
      if(d11+d22<d12+d21):
        fake_cat=torch.cat([out1,out2],1)
      else:
        fake_cat=torch.cat([out2,out1],1)
      
      g_crit=crit(fake_cat.detach())
      cricclean=torch.ones(g_crit.shape).cuda()
      cricmix=torch.zeros(g_crit.shape).cuda()
      lossc1=criterion_MSE(g_crit,cricclean)
      
      
      fake_dis1=d1(out1.detach())
        
      if(d11+d22<d12+d21):
        fake_dis1=d1(out1.detach())
      else:
        fake_dis1=d1(out2.detach())
      
      valid=torch.ones(real_dis1.shape).cuda()
      fake=torch.zeros(real_dis1.shape).cuda()
      
      loss1d=criterion_MSE(fake_dis1,valid)
        
      
      lossg=torch.min(d11+d22,d21+d12)+(loss1d+lossc1)
      #lossg=Variable(lossg,requires_grad=True)
      #train_iterator.set_description("batch:%3d,iteration:%3d,loss_g:%3f"%(epoch+1,iteration,lossg.item()))
      train_iterator.set_description("batch:%3d,iteration:%3d,loss_g:%3f,loss_c:%3f,loss_D2:%3f"%(epoch+1,iteration,lossg.item(),lossc1.item(),loss1d.item()))
      lossg.backward()
      #for name, weight in gen.named_parameters():
           #if weight.requires_grad:	
		          #train_iterator.set_description(name,str(parms.requires_grad),str(parms.grad))
             #print("name",name)
            #print(name,weight.grad)
      optimizer_G1.step()
      
      
