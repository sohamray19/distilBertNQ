# distilBertNQ

Question Answering performed on Google's Natural Questions dataset using Huggingface's 
DistilBert Model

## Getting Started
 * Clone this repository to get started. 
 * Make sure you have python 3 installed.
 If not installed already, follow the instructions on  [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) 
### Prerequisites
You will need:  
* Huggingface's transformers library
* Tensorflow 2.0 (with gpu)
* gsutils
* venv
```
pip install -r requirements.txt
```
### Getting the data
To get the preprocessed training set from Google's bert-joint-baseline, you will need to run
```
gsutil cp -R gs://bert-nq/bert-joint-baseline/nq-train.tfrecords-00000-of-00001 .
```
```
gsutil cp -R gs://bert-nq/bert-joint-baseline/vocab-nq.txt .
```
To get the dev (validation) data from the task, you have to run 
```
gsutil -m cp -R gs://natural_questions/v1.0/dev .
```
To generate the validation data record, you will need to run
```
python -m generate_validation.py --logtostderr
```

## Training
To run training, run this command
```
python bertNQ_training.py
```

## Evaluation
For evaluation, first you will need to download the evaluation script from [here](https://ai.google.com/research/NaturalQuestions/download)
If the model is fully trained, this should give a score as such: 
```
{"long-best-threshold-f1": 0.00043383947939262476, "long-best-threshold-precision": 0.5, "long-best-threshold-recall": 0.00021701388888888888, "long-best-threshold": -2.664327621459961e-05, "long-recall-at-precision>=0.5": 0.00021701388888888888, "long-precision-at-precision>=0.5": 0.5, "long-recall-at-precision>=0.75": 0, "long-precision-at-precision>=0.75": 0, "long-recall-at-precision>=0.9": 0, "long-precision-at-precision>=0.9": 0, "short-best-threshold-f1": 0.0, "short-best-threshold-precision": 0.0, "short-best-threshold-recall": 0.0, "short-best-threshold": 0.0, "short-recall-at-precision>=0.5": 0, "short-precision-at-precision>=0.5": 0, 
"short-recall-at-precision>=0.75": 0, "short-precision-at-precision>=0.75": 0, "short-recall-at-precision>=0.9": 0, "short-precision-at-precision>=0.9": 0}
```
## Model Description and Architectural Decisions

## Potential Improvements to Performance with more time

## Potential Improvements to Performance with better hardware 

## Feedback

## Charity Organisation of choice