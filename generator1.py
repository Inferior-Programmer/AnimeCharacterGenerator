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
  def __init__(self, channel,h,w, upscale = False, _kernel_size = 3, addChannel = 0):
    super(ResBlockG, self).__init__()
    self.nots = False

    self.up = nn.Identity()
    if(upscale):
      self.up = nn.PixelShuffle(2)
      channel = channel//4


    self.conv1 = nn.Conv2d(channel,channel + addChannel, kernel_size = _kernel_size,
                           bias = True, padding = 'same')
    self.conv2 = nn.Conv2d(channel+ addChannel, channel + addChannel, kernel_size = _kernel_size,
                           bias = True, padding = 'same')
    self.conv3 = nn.Identity()
    if(addChannel > 0):
      self.conv3 = nn.Conv2d(channel, channel + addChannel, kernel_size = 1,
                            bias = True)
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
    self.linear = nn.Linear(128+23, 4*4*1024, bias = True) # ADD SPECTRAL NORM HERE
    #self.linear2 = nn.Linear(23, 4*4*256, bias = False)
    self.copy = nn.Identity()
    self.blockbefore1 = nn.Sequential(*[ResBlockG(64,16,16) for i in range(16)])
    self.block1 = ResBlockG(64,16,16, addChannel = 512-64)
    self.block2 = ResBlockG(512,32,32, True)
    self.block2a = ResBlockG(128,32,32, addChannel = 256-128)
    self.block3 = ResBlockG(256,64,64,True)
    self.block3a = ResBlockG(64,64,64, addChannel = 128-64)
    self.block4 = ResBlockG(128,128,128, True)
    #self.block5 = ResBlockG(16)
    self.conv2 = nn.Conv2d(32,3,9,1,4)
    self.bn2 = nn.BatchNorm2d(32)


  def forward(self,x, labels):

    y = self.linear(torch.cat((x,labels), dim = 1))
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


def __init__weights(m):
  if isinstance(m, (nn.Linear, nn.Conv2d)):
    nn.init.kaiming_normal_(m.weight, a=0., mode="fan_in", nonlinearity="leaky_relu")


def __weights__init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)


origGenerator = Generator().to(device)
origGenerator.apply(__weights__init)

def generatePoints(batch_size, dim):
  hair_selector = np.random.randint(0,13, size = (batch_size))
  hair_indices = np.arange(0,batch_size)
  hair_features = np.zeros((batch_size,13))
  hair_features[hair_indices,hair_selector] = 1
  eye_selector = np.random.randint(0,10, size = (batch_size))
  eye_indices = np.arange(0,batch_size)
  eye_features = np.zeros((batch_size,10))
  eye_features[eye_indices,eye_selector] = 1
  features = np.random.randn(batch_size,dim)
  labels = np.concatenate((hair_features,eye_features), axis = 1)
  #finalFeatures = np.concatenate((features,labels), axis = 1)
  labels = labels
  finalFeatures = features
  return finalFeatures, labels

class EMA:
    """
    exponential moving average
    """
    def __init__(self, model, decay=0.999):
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

emaOrig = EMA(origGenerator)
checkpoint = torch.load('models\KanonNetE_30_rms.pt', map_location = device)
origGenerator.load_state_dict(checkpoint["netG"])
emaOrig.__dict__ = checkpoint["ema"]
emaOrig.apply(origGenerator)
origGenerator.eval()

def generateImage1(inputValues, labels):
  inputValues = torch.from_numpy(inputValues).type(torch.FloatTensor).to(device)
  labels = torch.from_numpy(labels).type(torch.FloatTensor).to(device)
  with torch.no_grad():
    return origGenerator(inputValues, labels).cpu()


if __name__ == '__main__':
  inputValues, labels = generatePoints(1,128)
  images = generateImage1(inputValues, labels).numpy().transpose(0, 2, 3, 1)
  x = np.array(["blonde hair", "brown hair", "black hair", "blue hair", "pink hair", "purple hair", "green hair", "red hair", "silver hair", "white hair", "orange hair", "aqua hair", "grey hair", "blue eyes", "red eyes", "brown eyes", "green eyes", "purple eyes", "yellow eyes", "pink eyes", "aqua eyes", "black eyes", "orange eyes"])
  label_names = x[np.asarray(labels > 0).nonzero()[1]]
  label_names = label_names.reshape(-1,2)
  print(label_names)
  images = (images + 1)/2
  plt.imshow(images[0], interpolation='nearest')
  plt.show()
