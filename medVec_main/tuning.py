from torch.optim import AdamW, Adam, SGD
import torch
from torch import nn
from transformers.optimization import Adafactor


class sftTune:

  '''

  Descriptor here

  '''
  def __init__(self,model,learn_rate):
    self.model = model
    self.learn_rate = learn_rate

  def adamW(self):
    optimizer = AdamW(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def adam(self):
    optimizer = Adam(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def adafactor(self):
    optimizer = Adafactor(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def sgdM(self,momentum):
    optimizer = SGD(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def xc_entropy(self):
    loss = torch.nn.CrossEntropyLoss()
    return loss

  def mse(self):
    loss = torch.nn.MSELoss()
    return loss

  def l1loss(self):
    loss = torch.nn.L1Loss()
    return loss


