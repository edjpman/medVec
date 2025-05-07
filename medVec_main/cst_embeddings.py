
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np


class cBERTbase:


    '''

    A class to product BERT based embeddings utilizing the `ClinicalBERT` pretrained model as the base.

    Methods:
    - mv_tokenizer: Tokenizes and formats code, initializes BERT model, returns the output embeddings.
    - bertInst: method to acutally create the sentence level embedding. This method is used internally to the tokenizer.
    - ttd_splits: Splits data into train/test/dev sets.

    '''

    def __init__(self, inputs, model_name="medicalai/ClinicalBERT"):
        '''

        Initializes cBERTbase by loading the ClinicalBERT model and tokenizer.

        '''
        self.inputs = inputs
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def mv_tokenizer(self,train_texts,dev_texts,test_texts,train_labels,dev_labels,test_labels,repr='pooled'):
      '''

      Tokenizes the provided text data, feeds it through the `ClinicalBERT` model, and returns the resulting embeddings along with corresponding labels for each data split.

      Parameters:
        - train_texts: List of training text samples.
        - dev_texts: List of tuning text samples.
        - test_texts: List of test text samples.
        - train_labels: Corresponding labels for training data.
        - dev_labels: Corresponding labels for tuning data.
        - test_labels: Corresponding labels for test data.
        - repr: Type of sentence embedding representation to return. Either 'pooled' (mean of last hidden state) or 'cls' (CLS token embedding).

        Returns:
        - embd_splits: Dictionary containing embeddings and labels for train, dev, and test splits.

        
      '''

      #Check if data is in a list format if not it converts to a Python list object
      if not isinstance(train_texts, list):
        train_texts = train_texts.tolist()
      if not isinstance(dev_texts, list):
          dev_texts = dev_texts.tolist()
      if not isinstance(test_texts, list):
          test_texts = test_texts.tolist()
      if not isinstance(train_labels, list):
          train_labels = train_labels.tolist()
      if not isinstance(dev_labels, list):
          dev_labels = dev_labels.tolist()
      if not isinstance(test_labels, list):
          test_labels = test_labels.tolist()

      #Creates dictionary structure for datasplits
      data_splits = {
        'train': {'texts': train_texts, 'labels': train_labels},
        'dev':   {'texts': dev_texts,   'labels': dev_labels},
        'test':  {'texts': test_texts,  'labels': test_labels}
      }

      #Initializes an empty dictionary to store the embeddings and labels within
      embd_splits = {}


      #Loops through each of the split groups
      #---First tokenizing the text
      #---Instantiates BERT and initializes them as the model inputs
      #---Creates the sentence embeddings and adds them to the empty dictionary by split group for raw use or classification tasks
      for split in ['train','dev','test']:
        texts = data_splits[split]['texts']
        labels = data_splits[split]['labels']

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        self.inputs = inputs
        embeddings = self.bertInst(repr=repr)

        embd_splits[split] = {
            'embeddings': embeddings,
            'labels': torch.tensor(labels, dtype=torch.long)
        }

      return embd_splits


    def bertInst(self,repr='pooled'):
        '''

        Generates sentencel embeddings using the `ClinicalBERT` model.

        Parameters:
        - repr: Type of sentence embedding representation to return. Either 'pooled' (mean of last hidden state) or 'cls' (CLS token embedding).

        Returns:
        - Sentence embeddings: PyTorch tensor of embeddings based on selected representation.

        
        '''

        #Assigns the IDs and Attention mask features of the tokenized inputs to separate objects
        input_ids = self.inputs['input_ids']
        attention_mask = self.inputs['attention_mask']

        #Passes the inputs through the specified BERT model assigning the last hidden state to an object
        with torch.no_grad():
          outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
          last_hidden_state = outputs.last_hidden_state

        #Conditional logic to specify the desired representation of the sentence embedding
        if repr == 'cls':
          cls_embedding = last_hidden_state[:, 0, :]
          return cls_embedding
        elif repr == 'pooled':
          sentence_embedding = torch.mean(last_hidden_state, dim=1)
          return sentence_embedding
        else:
          return ValueError('Expected "cls" or "pooled"')



    def ttd_splits(self,dataset,x_col,y_col,train_prop=0.7,dev_prop=0.2,test_prop=0.1):

      '''
      
      Splits the dataset into training, tuning, and testing sets.

        Parameters:
        - dataset: The dataset to be split.
        - x_col: Column name of the text input.
        - y_col: Column name of the label.
        - train_prop: Proportion of data to use for training.
        - dev_prop: Proportion of data to use for tuning.
        - test_prop: Proportion of data to use for testing.

        Returns:
        - Dataset splits: X_train, X_dev, X_test, y_train, y_dev, y_test; those with the X prefix refer to the `x_col` argument and y with the `y_col` argument.

      '''

      #Raises an error if the sum of the three is not equal to 1 (i.e. 100% of data)
      if not np.isclose(train_prop + dev_prop + test_prop, 1.0):
        raise ValueError('The sum of the proportions must sum to 1!')


      #Automatically balances the proportions for the incorporation of a dev set
      modified_dev_prop = dev_prop/(1-test_prop)

      #This splits it into the true test set (i.e. 10% default)
      X_train, X_test, y_train, y_test = train_test_split(dataset[x_col], dataset[y_col], test_size=test_prop, random_state=42)
      #This splits it into the true train size (i.e. 70% default) and dev size (i.e. 20% default)
      X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=modified_dev_prop, random_state=42)

      return X_train, X_dev, X_test, y_train, y_dev, y_test



class BERTdataset(Dataset):

  '''

  Wraps the embeddings into a `PyTorch` compatible dataset for efficient loading and batching during training.

  Takes inputs such as:

  embeddings = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
  ]
  labels = [0, 1]

  dataset[1] --- ([0.4, 0.5, 0.6], 1)


  If a batch of 2 is set throguh DataLoader() method:

  ([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3] ], [1, 0])

  '''


  def __init__(self, embeddings, labels):
     self.embeddings = embeddings
     self.labels = labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.embeddings[idx], self.labels[idx]

