###### Prerequisite: Please download [Git lfs](https://git-lfs.github.com/) 
 
# [Same Side Classification](https://sameside.webis.de) 

Identifying (classifying) the stance of an argument towards a particular topic is a fundamental task in computational argumentation. The stance of an argument as considered here is a two-valued function: it can either be ''pro'' a topic (= yes, I agree), or ''con'' a topic (= no, I do not agree).

With the new task » same side (stance) classification« we address a simpler variant of this problem: Given two arguments regarding a certain topic, the task is to decide whether or not the two arguments have the same stance. 
  
## Task: Same Side Classification

Given two arguments on the same topic, decide whether they have the same or opposite stance towards the topic. 


## Settings
We have two experimental settings:
 - Within: Train on a set of topics and evaluate on the same set of topics.
 - Cross: Train on one topic and evaluate on another topic.
We choose the 2 topics with highest number of arguments: *abortion* and *gay marriage*.


## Data
### Data source:
idebate.org, debatepedia.org, debatewise.org, debate.org

The data folder contains the training and testing data, for *cross* and *within* topics. You can split the *training* data as you like in order to train your model. After that, the model will be evaluated using the test data.

## Baseline
We trained a model using lemma 3 grams for argument1 and argument2 on the training set and then we evaluated the model using the test set. The results are the following:

 - Within Topics:
   - 	Accuracy: 54%
   - 	Macro-F1:  0.39
   - 	Micro-F1: 0.54

 - Cross Topics:
   - 	Accuracy: 58%
   - 	Macro-F1:  0.39
   - 	Micro-F1: 0.58
   
For more details, visit https://sameside.webis.de


## Approaches

### lightGBM 
We used a lightGBM a gradient boosting framework from Microsoft that uses tree based learning algorithms as a classifier model. We experimented with a number of representations based on CountVectorizer and TIDFVectorizer with 1, 2, 3, 1-2, 1-3-grams. As lightGBM returns a probability confidence score, we also experimented with different thresholds for a classifier to assign a binary label. We found out the following combinations of representations and thresholds worked best: TfidfVectorizer, 1-grams, threshold=0.520 for within-topics und TfidfVectorizer, 1-grams, threshold=0.501 for cross-topics.

### rule-based 
We considered the same-side classification as a sentiment analysis task. Given two arguments, both with negative as well as positive sentiments would be on the same side. We used a list of positive and negative words from Minqing Hu and Bing Liu, Mining and summarizing customer reviews and build rule-based classifier to assign each argument in a pair a positive or negative label. We used negation before the word to swap the polarity. If the argument does not contain any of the sentiment words, we used a random label assignment.

(Partly) Implementation of the system described in "Stance Classification of Context-Dependent Claims" from Bar-Haim et al.

Takes three arguments when run:
1. Path to the claim dataset
2. Mode (0: training the target identifier, 1: testing, 2: testing (& caching))
3. Path to a word2Vec Model (ie. Model trained on Google News: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)