
import re
import spacy
from datasets import load_dataset
import pandas as pd

class mvPreproc:

    '''

    A class for preprocessing medical definition and patient text datasets for BERT modeling and classification texts.

    Methods:
    - ds_load: Loads datasets in from Hugging Face.
    - clean_text: Regex text cleaning tool.
    - noun_phrases: Extracts noun phrases via a spaCy model.
    - feat_map: Maps categorical features to new labels.


    '''

    #Initialized spacy model for noun parsing
    def __init__(self):
      '''

      Initializes the English spaCy language model for noun phrase extraction.

      '''
      self.nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def ds_load():
      '''

      Loads in the two datasets from `Hugging Face`.

      Parameters: NONE

      Returns: patient dataset, medical definition dataset, both as pandas dataframes.


      '''
      #Datasets from Hugging Face
      patient_ds = load_dataset('gretelai/symptom_to_diagnosis')
      med_ds = load_dataset('celikmus/mayo_clinic_symptoms_and_diseases_v1')
      patient_ds = pd.DataFrame(patient_ds['train'])
      patient_ds = patient_ds.rename(columns={'output_text':'label','input_text':'text'})
      med_ds = pd.DataFrame(med_ds['train'])
      return patient_ds, med_ds

    @staticmethod
    def clean_text(text):
      '''

      Cleans text data by removing URLs, citation marks, and HTML tagging.

      Parameters:
      - text: The raw text to be cleaned.

      Returns: 
      - text: Cleaned version of the text.


      '''
      #URL removal
      text = re.sub(r'http[s]?://[^\s\)\]]+', '', text)
      #Citation removal
      text = re.sub(r'\[\d+\]', '', text)
      text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
      #HTML removal
      text = re.sub('<.*?>', '', text)
      return text

    def noun_phrases(self,text):
      '''

      Extracts the noun phrases from text data. This is specific to English only.

      Parameters:
      - text: The raw text to parse nouns from.

      Returns: 
      - text: Noun phrases from input text separaed by a |.


      '''
      #Leverages the initialized spacy model
      doc = self.nlp(text)
      return ' | '.join(chunk.text for chunk in doc.noun_chunks)


    @staticmethod
    def feat_map(label_map,dataset,col):
      '''

      Method maps new labels for categories within a feature of a dataset.

      Example of expected mapping
      label_map = {'cat': 0, 'dog': 1, 'rabbit': 2}

      Parameters:
      - label_map: A dictionary of the old-new mapped labels. 
      - dataset: The pandas DataFrame to which the mapping is applied to.
      - col: The name of the column to apply the mapping to.

      Returns: The dataset with the transformed column.


      '''
      dataset[col] = dataset[col].map(label_map)
      return dataset


