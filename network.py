import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        
            
class cup(nn.Module):
  def __init__(self,in_c,out_c,ks=4,st=2,pd=1):
    super(cup,self).__init__()
    self.conv=nn.ConvTranspose2d(in_c,out_c,kernel_size=ks,stride=st,padding=pd)
    self.relu=nn.LeakyReLU(0.2, True)
    #self.relu=nn.ReLU(inplace=True)
  def forward(self,x):
    return self.relu(self.conv(x))
    
class cdown(nn.Module):
  def __init__(self,in_c,out_c,ks=4,st=2,pd=1):
    super(cdown,self).__init__()
    self.conv=nn.Conv2d(in_c,out_c,kernel_size=ks,stride=st,padding=pd)
    self.relu=nn.LeakyReLU(0.2, True)
    #self.relu=nn.ReLU(inplace=True)
  def forward(self,x):
    return self.relu(self.conv(x))
    
class unet256(nn.Module):
  def __init__(self):
    super(unet256,self).__init__()
    self.d1=cdown(3,64)#128
    self.d2=cdown(64,128)#64
    self.d3=cdown(128,256)#32
    self.d4=cdown(256,512)#16
    #self.down=nn.AdaptiveAvgPool2d((128,128))

    self.b1=cdown(512,512,ks=3,st=1,pd=1)
    self.b2=cdown(512,512,ks=3,st=1,pd=1)
    self.b3=cdown(512,512,ks=3,st=1,pd=1)
    self.b4=cdown(512,512,ks=3,st=1,pd=1)
    self.u2=cup(512,512,ks=3,st=1,pd=1)
    self.u3=cup(512,512,ks=3,st=1,pd=1)
    self.u4=cup(512,512,ks=3,st=1,pd=1)
    self.u5=cup(512,512,ks=3,st=1,pd=1)
    
    self.up=nn.UpsamplingBilinear2d(scale_factor=2)
    #branch1
    #self.u16=cup(512,512,ks=3,st=1,pd=1)#16
    self.u17=cup(512+512,256)#16,16
    self.u18=cup(256+256,128)#32
    self.u19=cup(128+128,64)
    self.u20=nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1)
    self.tanh1=nn.Tanh()

    #branch2
    #self.u26=cup(512,512,ks=3,st=1,pd=1)
    self.u27=cup(512+512,256)
    self.u28=cup(256+256,128)
    self.u29=cup(128+128,64)
    self.u30=nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1)
    self.tanh2=nn.Tanh()
  def forward(self,x):
    d1=self.d1(x)
    d2=self.d2(d1)
    d3=self.d3(d2)
    d4=self.d4(d3)
    
    bd=self.u5(self.u4(self.u3(self.u2(self.b4(self.b3(self.b2(self.b1(d4))))))))
    #bd=self.up(bd)
    #branch1
    #d4=F.interpolate(d4,scale_factor=4)
    #d3=F.interpolate(d3,scale_factor=4)
    #d2=F.interpolate(d2,scale_factor=4)
    #u16=self.u16(bd)
    u17=self.u17(torch.cat([bd,d4],1))
    u18=self.u18(torch.cat([u17,d3],1))
    u19=self.u19(torch.cat([u18,d2],1))
    u20=self.u20(u19)
    out1=self.tanh1(u20)

    #branch2
    #u26=self.u26(bd)
    u27=self.u27(torch.cat([bd,d4],1))
    u28=self.u28(torch.cat([u27,d3],1))
    u29=self.u29(torch.cat([u28,d2],1))
    u30=self.u30(u29)
    out2=self.tanh2(u30)

    return out1,out2


class unet512(nn.Module):
  def __init__(self):
    super(unet512,self).__init__()
    self.d1=cdown(3,64)
    self.d2=cdown(64,128)
    self.d3=cdown(128,256)
    self.d4=cdown(256,512)

    self.b1=cdown(512,512)
    self.b2=cdown(512,512)
    self.b3=cdown(512,512)
    self.b4=cdown(512,512)
    self.b5=cdown(512,512)
    self.u1=cup(512,512)
    self.u2=cup(512,1024)
    self.u3=cup(1024,1024)
    self.u4=cup(1024,1024)
    self.u5=cup(1024,1024)
    self.u6=cup(1024,1024)

    #branch1
    self.u16=cup(1024,512)
    self.u17=cup(512+512,256)
    self.u18=cup(256+256,128)
    self.u19=nn.ConvTranspose2d(128+128,3,kernel_size=3,stride=1,padding=1)
    self.tanh1=nn.Tanh()

    #branch2
    self.u26=cup(1024,512)
    self.u27=cup(512+512,256)
    self.u28=cup(256+256,128)
    self.u29=nn.ConvTranspose2d(128+128,3,kernel_size=3,stride=1,padding=1)
    self.tanh2=nn.Tanh()
  def forward(self,x):
    d1=self.d1(x)
    d2=self.d2(d1)
    d3=self.d3(d2)
    d4=self.d4(d3)

    bd=self.u6(self.u5(self.u4(self.u3(self.u2(self.u1(self.b5(self.b4(self.b3(self.b2(self.b1(d4)))))))))))

    #branch1
    d4=F.interpolate(d4,scale_factor=4)
    d3=F.interpolate(d3,scale_factor=4)
    d2=F.interpolate(d2,scale_factor=4)
    u16=self.u16(bd)
    u17=self.u17(torch.cat([u16,d4],1))
    u18=self.u18(torch.cat([u17,d3],1))
    u19=self.u19(torch.cat([u18,d2],1))
    out1=self.tanh1(u19)

    #branch2
    u26=self.u26(bd)
    u27=self.u27(torch.cat([u26,d4],1))
    u28=self.u28(torch.cat([u27,d3],1))
    u29=self.u29(torch.cat([u28,d2],1))
    out2=self.tanh2(u19)

    return out1,out2

class discritic(nn.Module):
  def __init__(self):
    super(discritic,self).__init__()
    #64*64
    self.conv1=nn.Conv2d(3+3,64,kernel_size=4,stride=2,padding=1)#128
    self.bn1=nn.InstanceNorm2d(64)
    self.relu1=nn.LeakyReLU(0.2, True)
    self.conv2=nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1)#64
    self.bn2=nn.InstanceNorm2d(64)
    self.relu2=nn.LeakyReLU(0.2, True)
    self.conv3=nn.Conv2d(128,256,kernel_size=4,stride=2,padding=1)#32
    self.bn3=nn.InstanceNorm2d(64)
    self.relu3=nn.LeakyReLU(0.2, True)
    self.conv4=nn.Conv2d(256,512,kernel_size=4,stride=2,padding=1)#16
    self.bn4=nn.InstanceNorm2d(64)
    self.relu4=nn.LeakyReLU(0.2, True)
    self.conv5=nn.Conv2d(512,1,kernel_size=4,stride=2,padding=1)
    #self.bn5=nn.InstanceNorm2d(64)
    #self.relu5=nn.Sigmoid()
  def forward(self,x):
    x=F.interpolate(x,size=(64,64))
    d1=self.relu1(self.bn1(self.conv1(x)))
    d2=self.relu2(self.bn2(self.conv2(d1)))
    d3=self.relu3(self.bn3(self.conv3(d2)))
    d4=self.relu4(self.bn4(self.conv4(d3)))
    d5=self.conv5(d4)
    return d5
    #return self.d5(self.bn3(self.d4(self.bn2(self.d3(self.bn1(self.d2(self.d1(x))))))))
    
    