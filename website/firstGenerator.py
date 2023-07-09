import torch

import torch.nn as nn
import torch.nn.functional as F
import random
from collections import OrderedDict
from matplotlib import pyplot as plt
import numpy as np
manualSeed = 80812
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

ngpu =1

class ResBlockG(nn.Module):
  def __init__(self, channel, upscale = False, _kernel_size = 3, addChannel = 0):
    super(ResBlockG, self).__init__()
    self.nots = False 
    
    self.up = nn.Identity()
    if(upscale):
      self.up = nn.PixelShuffle(2) 
      channel = channel//4
    
    
    self.conv1 = nn.utils.parametrizations.spectral_norm(nn.Conv2d(channel,channel + addChannel, kernel_size = _kernel_size,
                           bias = False, padding = 'same'))
    self.conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv2d(channel+ addChannel, channel + addChannel, kernel_size = _kernel_size,
                           bias = False, padding = 'same'))
    self.conv3 = nn.Identity()
    if(addChannel > 0):
      self.conv3 = nn.utils.parametrizations.spectral_norm(nn.Conv2d(channel, channel + addChannel, kernel_size = 1,
                            bias = False))
    self.bn1 = nn.BatchNorm2d(channel) 
    self.bn2 = nn.BatchNorm2d(channel + addChannel)
    

  def forward(self,x):
    x = self.up(x)
    y = self.conv1(F.relu(self.bn1(x)))
    y = self.conv2(F.relu(self.bn2(y)))
    x = self.conv3(x) 
    y += x 
    return y 

#Changes Leakyrelu and removed bn1 #Changed Kernel To 9
class Generator(nn.Module):
  def __init__(self): 
    super(Generator, self).__init__()
    self.linear = nn.Linear(128, 4*4*1024, bias = False) # ADD SPECTRAL NORM HERE

  

    self.copy = nn.Identity()
    self.blockbefore1 = nn.Sequential(*[ResBlockG(64) for i in range(16)])
    self.block1 = ResBlockG(64, addChannel = 512-64)
    self.block2 = ResBlockG(512, True)
    self.block2a = ResBlockG(128,addChannel = 128)
    self.block3 = ResBlockG(256,True)
    self.block3a = ResBlockG(64,addChannel = 64)
    self.block4 = ResBlockG(128, True)
    #self.block5 = ResBlockG(16)
    self.conv2 = nn.utils.parametrizations.spectral_norm(nn.Conv2d(32,3,9,1,4))
    self.bn2 = nn.BatchNorm2d(32)


  def forward(self,x):

    y = self.linear(x)
    y = y.view(-1,64,16,16)
    y_copy = self.copy(y)
    y = self.blockbefore1(y)
    y += y_copy
    y = self.block1(y)
    y = self.block2(y)
    y = self.block2a(y)
    y = self.block3(y)
    y = self.block3a(y)
    y = self.block4(y)
    y = F.relu(self.bn2(y))
    y = self.conv2(y)
    y = torch.tanh(y) 
    return y 


    
generatorFirst = Generator().to(device)


class EMA:
    """
    exponential moving average
    """
    def __init__(self, model, decay=0.9999):
        self.shadow = OrderedDict([
            (k, v.clone().detach())
            for k,v in model.named_parameters()])
        self.decay = decay
        self.num_updates = 0
        
    def update(self, params):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, v in params:
            self.shadow[k] += (1 - decay) * (v.detach() - self.shadow[k])

    def apply(self, model):
        self.original =  OrderedDict([
            (k, v.clone().detach())
            for k,v in model.named_parameters()])
        for k, v in model.named_parameters():
            v.data.copy_(self.shadow[k])

    def restore(self, model):
        for k,v in model.named_parameters():
            v.data.copy_(self.original[k])
        del self.original


emaFirst = EMA(generatorFirst)
checkpoint = torch.load('models\KanonNet_30_rms.pt', map_location = device)

generatorFirst.load_state_dict(checkpoint["netG"])
emaFirst.__dict__ = checkpoint["ema"]
emaFirst.apply(generatorFirst)
generatorFirst.eval()

def generateFirst(inputValues):
  inputValues = torch.from_numpy(inputValues).type(torch.FloatTensor).to(device)
  with torch.no_grad():
    return origGenerator(inputValues).cpu()
