
import torch
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score


class Bert_test:

  '''

  A class for evaluating the BERT classifier head.
  Contains a method `test_loop` which evaluates model performance and computes metrics like accuracy, precision, recall, and F1-score.

  '''

  def __init__(self):
     pass

  def test_loop(self, dataloader, model, loss_fn, device='cpu'):
    '''

    Runs evaluation on a given dataset and returns performance metrics.

    Parameters:
    - dataloader: A PyTorch DataLoader providing test data
    - model: The trained model to evaluate
    - loss_fn: Loss function
    - device: Device to evaluate on ('cpu','gpu,'cuda')

    Returns:
    - test_loss: Average loss across all test batches
    - accuracy: Overall classification accuracy
    - precision: Weighted precision score
    - recall: Weighted recall score
    - f1: Weighted F1 score

    '''
    #Sets the model to evaluation mode
    model.eval()

    #Total number of test samples and batches
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    #Tracks the total loss and correct predictions
    test_loss, correct = 0, 0

    #Performs a check for whether a valid device is set
    if device not in ['cpu','gpu','cuda']:
      raise ValueError('Please select a valid device!')

    #Moves the model to the specified device
    model.to(device)

    #Stores all predicted and true labels
    all_preds = []
    all_labels = []


    with torch.no_grad():
        #Iterates through the batches
        for X_batch, y_batch in dataloader:

          #Moves batch data to the same device as the model
          X_batch, y_batch = X_batch.to(device), y_batch.to(device)

          #Forward Pass -- model predicts output logits
          pred = model(X_batch)

          #Accumulates the total loss across batches
          test_loss += loss_fn(pred, y_batch).item()

          #Gets the predicted class labels by selecting the index with highest logit
          preds = pred.argmax(1)

          #Counts the number of correct predictions
          correct += (preds == y_batch).type(torch.float).sum().item()

          #Stores predictions and labels for computing precision, recall, and F1
          #Coverts back to CPU if on a GPU or other device
          all_preds.extend(preds.cpu().numpy())
          all_labels.extend(y_batch.cpu().numpy())


    #Averages the loss over all batches in the test set
    test_loss /= num_batches
    #Computes the overall accuracy
    accuracy = correct / size

    #Computes the precision, recall, and F1
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    #Standard PyTorch print of accuracy
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")


    return test_loss, accuracy, precision, recall, f1
  
