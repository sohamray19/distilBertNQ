# distilBertNQ

Question Answering performed on Google's Natural Questions dataset using Huggingface's 
DistilBert Transformer

## Getting Started
 * Clone this repository to get started. 
 * Make sure you have python 3 installed.
 If not installed already, follow the instructions on  [Anaconda](https://docs.anaconda.com/anaconda/install/mac-os/) website
### Prerequisites
You will need:  
* Huggingface's transformers library
* Tensorflow 2.0 (with gpu)
* gsutils
* tensorflow addons
```
pip install -r requirements.txt
```
### Getting the data
To get the preprocessed training set, and vocabulary from Google's bert-joint-baseline, run these commands
```
gsutil cp -R gs://bert-nq/bert-joint-baseline/nq-train.tfrecords-00000-of-00001 .
```
```
gsutil cp -R gs://bert-nq/bert-joint-baseline/vocab-nq.txt .
```
To get the dev (validation) data from Google's bert-joint-baseline, run this command
```
gsutil -m cp -R gs://natural_questions/v1.0/dev .
```
To generate the validation data record, run this command
```
python -m generate_validation.py --logtostderr \
--vocab_file=/Users/aashnabanerjee/Documents/Cortx/inference/code/bert-joint-baseline/vocab-nq.txt \
--output_dir=/Users/aashnabanerjee/Documents/Cortx/inference/code/ \
--predict_file=/Users/aashnabanerjee/Documents/Cortx/inference/code/bert-joint-baseline/dev/nq-dev-??.jsonl.gz
```
```
python -m generate_validation.py --logtostderr \
--vocab_file=/path/to/vocab-nq.txt \
--output_dir=/path/to/output/directory/ \
--predict_file=path/to/validation/file/pattern
```

## Training
To train the model, run this command
```
pip install tensorflow==2.2.0
```
then this
```
python run_distilbert.py --training_mode=True --train_file=/path/to/train-file \
--use_chkpt=False --checkpoint_path=/path/to/store/checkpoint/ --epochs=2 --batch_size=2\
--init_learning_rate=3e-5 --init_weight_decay_rate=0.01
```
For example, 
```
python run_distilbert.py --training_mode=True --train_file=./train.tf_record \
--use_chkpt=False --checkpoint_path=./checkpoints/ --epochs=2 --batch_size=2\
--init_learning_rate=3e-5 --init_weight_decay_rate=0.01
```

## Validation
To generate the validation predictions in a predictions.json file, run this  command
```
python run_distilbert.py --training_mode=False --val_file=/path/to/val-record \
--use_chkpt=True --checkpoint_path=/path/to/stored/checkpoint/ --pred_file=path/to/dev/json/file \
--json-output-path=/path/to/store/predictions.json --batch_size=2
```
For example, 
```
python run_distilbert.py --training_mode=False \
--val_file=./dev.tf_record \
--use_chkpt=False --checkpoint_path=./checkpoints/ \
--batch_size=2 --pred_file=./nq-dev-sample.jsonl.gz \
--json_output_path=./
```
Note: Pred-file is usually just the val file before it was converted into a tf_record

## Evaluation
For evaluation, first you will need to download the evaluation script from [here](https://ai.google.com/research/NaturalQuestions/download).
Run it with this command:
```
!python -m nq_eval --gold_path=/path/to/validation-jsonl.gz --predictions_path=/path/to/predictions.json --logtostderr
```

If the model is fully trained, this should give a score similar to: 
```json
{"long-best-threshold-f1": 0.6041666666666666, "long-best-threshold-precision": 0.651685393258427,
 "long-best-threshold-recall": 0.5631067961165048, "long-best-threshold": 7.45175576210022,
 "long-recall-at-precision>=0.5": 0.6504854368932039, "long-precision-at-precision>=0.5": 0.5153846153846153,
 "long-recall-at-precision>=0.75": 0.30097087378640774, "long-precision-at-precision>=0.75": 0.7560975609756098,
 "long-recall-at-precision>=0.9": 0.02912621359223301, "long-precision-at-precision>=0.9": 1.0, 
 "short-best-threshold-f1": 0.4722222222222222, "short-best-threshold-precision": 0.4927536231884058, 
 "short-best-threshold-recall": 0.4533333333333333, "short-best-threshold": 8.397958993911743, 
 "short-recall-at-precision>=0.5": 0.44, "short-precision-at-precision>=0.5": 0.5, 
 "short-recall-at-precision>=0.75": 0.16, "short-precision-at-precision>=0.75": 0.75, 
 "short-recall-at-precision>=0.9": 0.04, "short-precision-at-precision>=0.9": 1.0}
```
## Model Description and Architectural Decisions
* Literature Review: Before I get into model description, here are the papers I referenced to make my architectural decisions: 
    * [LAMB](https://arxiv.org/abs/1904.00962.pdf)
    * [BERT-Baseline](https://arxiv.org/pdf/1901.08634.pdf)
    * [BERT](https://arxiv.org/abs/1810.04805)
    * [DistilBert](https://arxiv.org/abs/1910.01108.pdf)
    * [Fine-tuning](https://arxiv.org/abs/2002.06305)
    * [HuggingFace-Transformers](https://arxiv.org/pdf/1910.03771.pdf)
* **Approach**: After the literature review, my first step was to set up the skeleton of the pipeline. I wrote and revised the code
to train the model, generate the validation dataset, and then generate the predictions.json file for evaluation. For generating predictions, as well 
as generating the validation record file, I modified the code presented in the Bert-Baseline repo. 
Once this setup was established, I went deeper into the model, optimizers and loss functions to find the optimal set up. Based on the Fine-tuning paper mentioned above, 
I started multiple processes, on small parts of the dataset, and continued with only those that showed promising results.
Then, I did an ablation study by starting with the best suggested optimizer using weight decay and a scheduler 
and then removed them one by one to observe the impact it made. This helped me find the optimal optimizer configuration.
I also experimented with different loss functions and have presented my results in the excel sheet referenced below in the hyperparameter tuning section.
* **Data**: I have used the preprocessed training dataset provided in the bert-joint-baseline model. I have also written a script to generate
the validation set in a similar fashion. Not only does the smaller size make the data easier to deal with, but also 
it evens out the number of instances with NULL values, as well as adds the special tokens for where the answers are most often found.
This increases accuracy. For more information, please refer to the bert-baseline paper cited above.
I have used the tiny-dev dataset provided by Google as the validation dataset and am referring to the dev set by Google as the test set.
* **Model**: For this task, I have used the distilBert transformers model by HuggingFace. As per folk-lore, 
BERT-based models has been providing the best results. We can observe this if we look at the leaderboard as well. 
While I would have loved to use (or atleast experiment with) a larger BERT model, due to resource constraints, 
I have used the DISTILBERT model made by huggingface that is shown to have similar results, but is much lighter 
and faster. I started with the 'bert-fine-tuned-on-squad' model, but switched to distilbert version for faster, yet
similar results. For more information, please refer to the DistilBert paper cited above.
* **Optimizer**: For the optimizer, I tried 3 options, Adam, AdamW with custom scheduler, and finally LAMB. LAMB has
been shown to provide optimal results with BERT, and my model seemed to perform the best with it as well. 
For more information, please refer to the LAMB paper sited above. 
* **Hyperparameter Tuning**: Given the resource constraint, I had to be selective with my hyperparameter tuning. 
I am attaching an excel file [here](https://docs.google.com/spreadsheets/d/1zbUUo1AZ3lSKmf6uOprpUxFrqU7zJGqbrNLiNW-782o/edit?usp=sharing) 
which represents my process of hyperparameter tuning, with results. With more time (or with better hardware), I would love to run 
Bayesian Optimisation or random search on the parameters to improve performance.
## Potential Improvements to Performance with time
* I would love to go deeper into Deep learning techniques, including gradient and batch accumulation, and
other optimizations functions. I wold try to tweak these methods to perform best with our dataset.
* I would try out different architectures on top of distilbert and other transformer models, especially RoBERTa and BERT large, since with this much
data I would suspect larger models would give better results.
* I would like to perform an ablation study on multiple hyperparameter settings
and optimizer configurations to observe their effect on f1-score. 
* I would try visualising accuracy, loss, error rates and convergence for all combinations to look for patterns and actionable
insights.
* I would try to minimize re-allocations in my code to speed up operation
* I would learn and experiment with distributed training techniques.
## Potential Improvements to Performance with hardware 
* Gradient Accumulation. I tried doing it with my current code but couldn't get around the resource exhaustion(OOM) error. Additionally, it makes more sense to use gradient accumulation with multiple GPU's.
* With better hardware, I would have more time, allowing me to try out different combinations of hyperparameters
and transformers models.
* I would love to try different pre-processing techniques, especially methods that could make better use of the abundant data
* I would use Random Search/Bayesian Optimization (or something more suitable if that exists) for hyperparameter tuning on the model
* I would use techniques like cross validation to check for overfitting
* I would try an ensemble-based approach

## Feedback
As someone attempting a question answering task for the first time, I enjoyed learning and working on this challenging task. Ample time and resources were provided, and the 
charity donation was a cherry on top!

Potentially, I would consider providing lesser time and a starter code, so that the candidate can focus
more on the architectural decisions, and preprocessing change, if any, without spending too much time on reading and understanding
basic (albeit necessary) preprocessing and prediction generation code.

Overall, having never worked on data as large as this before, I loved the assignment, 
and I believe that I learned a lot from it!


## Charity Organisation of choice