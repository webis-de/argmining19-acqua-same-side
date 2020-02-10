## Approaches

### lightGBM 
We used a lightGBM a gradient boosting framework from Microsoft that uses tree based learning algorithms as a classifier model. We experimented with a number of representations based on CountVectorizer and TIDFVectorizer with 1, 2, 3, 1-2, 1-3-grams. As lightGBM returns a probability confidence score, we also experimented with different thresholds for a classifier to assign a binary label. We found out the following combinations of representations and thresholds worked best: TfidfVectorizer, 1-grams, threshold=0.520 for within-topics und TfidfVectorizer, 1-grams, threshold=0.501 for cross-topics.

Takes the following arguments:

| Option  | Description | Default |
| ------------------- | ------------- | ------------- |
| --mode  | Use training set (previously splitted to train, dev and test sets) for the training process (train_validation) or choose the test set (train_test)  | train_validation |
| --dataset  | Use within or cross dataset  | within |
| -- lemmatized | After the first experiment the lemmatized data can be used | false |
| --vect | Vectorizer used for feature extraction: if True - CountVectorizer, else - TfidfVectorizer | true |
| --ngram_range_1 | First value of ngram range used for feature extraction (choose between 3 and 1) | 3 |
| --ngram_range_2 | Second value of ngram range used for feature extraction (choose between 1, 2 and 3) | 3 |
| --activation_th | Activation Threshold of output | 0.5 |

If the training set need to be splitted first, you can change the path in the "read_fin_data" function to the path to the traning set and then use the train_test mode option. 

Example: main.py --mode train_validation --lemmatized --ngram_range_1 1 --activation_th 0.555

This set of option would perform an experiment on the training set, using already lemmatized data, CountVectorizer, 1-3-grams and acivation threshold of 0.555


### rule-based 
We considered the same-side classification as a sentiment analysis task. Given two arguments, both with negative as well as positive sentiments would be on the same side. We used a list of positive and negative words from Minqing Hu and Bing Liu, Mining and summarizing customer reviews and build rule-based classifier to assign each argument in a pair a positive or negative label. We used negation before the word to swap the polarity. If the argument does not contain any of the sentiment words, we used a random label assignment.

### IBM
(Partly) Implementation of the system described in "Stance Classification of Context-Dependent Claims" from Bar-Haim et al.

Takes three arguments when run:
1. Path to the claim dataset
2. Mode (0: training the target identifier, 1: testing, 2: testing (& caching))
3. Path to a word2Vec Model (ie. Model trained on Google News: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)
