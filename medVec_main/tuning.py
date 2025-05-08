from torch.optim import AdamW, Adam, SGD
import torch
from torch import nn
from transformers.optimization import Adafactor


class sftTune:

  '''

  A class for configuring optimizers and loss functions for PyTorch models. 

  
  '''
  def __init__(self,model,learn_rate):
    '''
    
    Initializes the tuning functionality with the specified model and learning rate.

    '''
    self.model = model
    self.learn_rate = learn_rate

  def adamW(self):
    '''
    
    Creates an AdamW optimizer. 

    Returns: The AdamW optimizer object.
    
    '''
    optimizer = AdamW(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def adam(self):
    '''
    
    Creates an Adam optimizer. 

    Returns: The Adam optimizer object.
    
    '''
    optimizer = Adam(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def adafactor(self):
    '''
    
    Creates an Adafactor optimizer. 

    Returns: The Adafactor optimizer object.
    
    '''
    optimizer = Adafactor(self.model.parameters(), lr=self.learn_rate)
    return optimizer

  def sgdM(self,momentum):
    '''
    
    Creates a Stochastic Gradient Descent optimizer. 


    Parameters:
    - momentum: The momentum level to use with SGD.


    Returns: The SGD optimizer object.
    
    '''
    optimizer = SGD(self.model.parameters(), lr=self.learn_rate, momentum=momentum)
    return optimizer

  def xc_entropy(self):
    '''
    
    Creates a Cross-Entropy loss function.

    Returns: The Cross-Entropy Loss function object.

    '''
    loss = torch.nn.CrossEntropyLoss()
    return loss

  def mse(self):
    '''
    
    Creates a Mean Square Error (MSE) loss function.

    Returns: The Mean Square Error loss function object.

    '''
    loss = torch.nn.MSELoss()
    return loss

  def l1loss(self):
    '''
    
    Creates a Mean Absolute Error (MAE) loss function.

    Returns: The Mean Square Absolute loss function object.

    '''
    loss = torch.nn.L1Loss()
    return loss


