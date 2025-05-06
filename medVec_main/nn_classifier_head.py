
from torch import nn

class cBertMCChead(nn.Module):
    '''

    A classification head to sit on top of BERT model.
    This head consists of multiple connected layers with ReLU activations and dropout, terminating in an output layer corresponding to the number of target classes.

    '''
    #Standard initialization function for PyTorch based neural networks
    def __init__(self,
                  input_dim=768, #BERT embedding base
                  hidden_dim=256, #Size of hidden layer
                  dropout=0.3, #Dropout prob
                  num_layers=2, #Number of hidden layers to construct
                  num_classes=4): #Number of classes to target
        super(cBertMCChead, self).__init__()

        #Initializes a list to store the layers of the sequence
        layers = []

        #Constructs the first layer -- BERT input to hidden dimension
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        #Constructs an additional layer of the same format if the number of layers > 1
        for i in range(num_layers - 1):
          layers.append(nn.Linear(hidden_dim, hidden_dim))
          layers.append(nn.ReLU())
          layers.append(nn.Dropout(dropout))

        #Constructs a final output layer -- hidden dimension to the number of classes (i.e. raw logits and no softmax)
        layers.append(nn.Linear(hidden_dim, num_classes))

        #Wraps all layers into a sequential container
        self.fc = nn.Sequential(*layers)

    def forward(self,x):
      #Defines the forward pass and returns the raw logits -- input X goes through the sequential layers
      return self.fc(x)
    
