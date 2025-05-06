

class cBERT_train:

  '''

  A class to handle training of the BERT classifier head.
  Contains a method `train_loop` which iterates through the data loader, runs the forward and backward passes, and updates model parameters.

  '''
  def __init__(self):
    pass


  def train_loop(self, dataloader, model, loss_fn, optimizer,device='cpu'):
    '''

    Runs one training epoch over the provided dataloader.

    Parameters:
    - dataloader: A PyTorch DataLoader producing batches of (inputs, labels)
    - model: The classification head to train
    - loss_fn: Loss function
    - optimizer: Optimizer
    - device: Device to train on ('cpu','gpu,'cuda')

    '''

    #Total number of examples
    size = len(dataloader.dataset)

    #Sets the model to training mode
    model.train()

    #Performs a check for whether a valid device is set
    if device not in ['cpu','gpu','cuda']:
      raise ValueError('Please select a valid device!')

    #Moves the model to the specified device
    model.to(device)

    #Iterates through the batches
    for batch, (X_batch, y_batch) in enumerate(dataloader):

        #Moves batch data to the same device as the model
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        #Forward Pass -- model predicts output logits
        pred = model(X_batch)

        #Computes loss between predictions and true labels
        loss = loss_fn(pred, y_batch)

        #Backpropagation -- backward pass and optimization
        #---Old gradients are cleared, new gradients are computed, and model weights are updated
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Standard PyTorch print of batch progress
        if batch % 100 == 0:
          loss_value = loss.item()
          print(f"loss: {loss_value:>7f}  [{batch * len(X_batch):>5d}/{size:>5d}]")








