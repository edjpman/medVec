# medVec
A BERT-Powered Vector Diagnosis Recommender

This library allows prospective medical professionals to upskill using AI in a clinical setting. The library offers easy to use functionality to explore the contextual relationships between a patients details of what they face to what exists in the medical literature. While this library utilizes a ClinicalBERT base model trained on a vast clinical corpus, this tool is intendend for educational and exploratory purposes only. Use in a real clinical setting to recommend diagnosis should always the at the discresion of a licensed medical professional.

*** ***

## Installing medVec


To run a demo notebook of the working sentence classification model please clone the repo, build the Docker image, and simply run it!


```bash

git clone https://github.com/edjpman/medVec.git
cd medVec
docker build -t medvec-demo .
docker run -p 8888:8888 medvec-demo

```

*** ***

## Demo Overview

Checkout this page [here](https://edjpman.github.io/medVec/)!




