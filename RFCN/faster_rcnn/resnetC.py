from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from model.faster_rcnn.faster_rcnn import _fasterRCNN
from faster_rcnn import FasterRCNN
from model.utils.config import cfg

from .torch_deform_conv.layers import ConvOffset2D as ConvOffset2d
import torch.utils.model_zoo as model_zoo
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class deformResNet(nn.Module):
  def __init__(self, resnet, lastlayer, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = resnet.conv1
    self.bn1 = resnet.bn1
    self.relu = resnet.relu
    self.maxpool = resnet.maxpool
    self.layer1 = resnet.layer1
    self.layer2 = resnet.layer2
    self.layer3 = resnet.layer3
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    self.layer4 = lastlayer

    self.avgpool = resnet.avgpool
    self.fc = resnet.fc

    self.finalConv = nn.Conv2d(2048, 1024, 1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    print('before layer 4, size of feature:')
    print(x.size())
    x = self.layer4(x)
    print('after layer 4, size of feature:')
    print(x.size())

    #x = self.avgpool(x)
    #x = x.view(x.size(0), -1)
    #x = self.fc(x)
    x = self.finalConv(x)

    return x


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    #self.finalConv = nn.Conv2d(2048, 1024, 1)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


class lastLayer(nn.Module):
  def __init__(self, oldLayer, lastblock, num_blocks=3):
    self.inplanes = 64
    super(lastLayer, self).__init__()
    self.Block1 = oldLayer[0]
    self.Block2 = oldLayer[1]
    self.Block3 = lastblock

  def forward(self, x):
    x = self.Block1(x)
    x = self.Block2(x)
    x = self.Block3(x)
    return x


class last_block(nn.Module):
  def __init__(self, old_block, len1, len2, len3):
    super(last_block, self).__init__()
    self.deform1 = ConvOffset2d(len1)
    self.conv1 = nn.Sequential(*list(old_block.children()))[0]
    self.bn1 = nn.Sequential(*list(old_block.children()))[1]
    self.deform2 = ConvOffset2d(len2)
    self.conv2 = nn.Sequential(*list(old_block.children()))[2]
    self.bn2 = nn.Sequential(*list(old_block.children()))[3]
    self.deform3 = ConvOffset2d(len3)
    self.conv3 = nn.Sequential(*list(old_block.children()))[4]
    self.bn3 = nn.Sequential(*list(old_block.children()))[5]
    self.relu = nn.Sequential(*list(old_block.children()))[6]

  def forward(self, x):
    x = self.deform1(x)
    x = self.conv1(x)
    x = self.bn1(x)

    x = self.deform2(x)
    x = self.conv2(x)
    x = self.bn2(x)

    x = self.deform3(x)
    x = self.conv3(x)
    x = self.bn3(x)
    x = self.relu(x)
    return x

    
def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  resnet = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    resnet.load_state_dict(model_zoo.load_url(model_urls['resnet101']))

  # Remove last two layers
  resnet1 = nn.Sequential(*list(resnet.children())[:-2])

  # replace last 3 conv with deform conv
  lastlayer = nn.Sequential(*list(resnet1.children()))[-1]
  lastblock = nn.Sequential(*list(lastlayer.children()))[-1]
  new_last_block = last_block(lastblock, 2048, 512, 512)
  new_last_layer = nn.Sequential(*list(lastlayer.children()))[:-1]

  #new_last_layer = nn.Sequential(new_last_layer, new_last_block)
  new_last_layer = lastLayer(new_last_layer, new_last_block)
  # build a new class to hold it
  resnet_final = deformResNet(resnet, new_last_layer)

  return resnet_final

class resnet(FasterRCNN):#_fasterRCNN
  def __init__(self, classes=None, num_layers=101, pretrained=False, class_agnostic=False):
    classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    FasterRCNN.__init__(self, classes, class_agnostic)#_fasterRCNN

  def _init_modules(self):
    resnet = resnet101(pretrained=True)
    '''
    if self.pretrained == True:
      print("Loading pretrained weights from %s" % (self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})
    '''
    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                   resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.finalConv = deformResNet.finalConv                                        # ************ MODIFIED **********************

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base[0].parameters():
      p.requires_grad = False
    for p in self.RCNN_base[1].parameters():
      p.requires_grad = False

    assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
    if cfg.RESNET.FIXED_BLOCKS >= 3:
      for p in self.RCNN_base[6].parameters():
        p.requires_grad = False
    if cfg.RESNET.FIXED_BLOCKS >= 2:
      for p in self.RCNN_base[5].parameters():
        p.requires_grad = False
    if cfg.RESNET.FIXED_BLOCKS >= 1:
      for p in self.RCNN_base[4].parameters():
        p.requires_grad = False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters():
          p.requires_grad = False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

#    # check auto grad init
#    print(resnet_final)
#    for param in model.parameters():
#      print('gradient setting of each layer')
#      print(param.requires_grad)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base.eval()
      self.RCNN_base[5].train()
      self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
