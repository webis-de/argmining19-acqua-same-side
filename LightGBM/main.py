#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import lightgbm as lgb
import pandas as pd
import csv
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

from ipynb.fs.full.lemmatize import get_wordnet_pos
from ipynb.fs.full.lemmatize import lemmatize_stemming
from ipynb.fs.full.lemmatize import preprocess
from ipynb.fs.full.lemmatize import get_lemma

#split data in X and y
def cut_df(df):
    X = df[['argument1', 'argument2', 'topic']]
    y = df[['is_same_side']]
    return X, y

#split data in train and dev sets
def get_train_dev_sets(df):
    X = df[['argument1', 'argument2', 'topic']]
    y = df[['is_same_side']]

    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    return X_train, X_dev, y_train, y_dev

#extraction ngrams function; 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def extract_ngrams(X_train, X_test, X_dev, CV, nr1, nr2, col, idx='id'):
    print(col)
    if CV == True:
        vectorizer = CountVectorizer(min_df=6, max_df=0.7, ngram_range=(nr1, nr2), max_features=5000 )
    else:
        vectorizer = TfidfVectorizer(min_df=6, max_df=0.7, ngram_range=(nr1, nr2), max_features=5000 )
    
    vectorizer.fit(X_train[col].values.astype('U'))
    features = vectorizer.transform(X_train[col].values.astype('U'))
    features_test = vectorizer.transform(X_test[col].values.astype('U'))
    features_dev = vectorizer.transform(X_dev[col].values.astype('U'))

    train_df =pd.DataFrame(
        features.todense(),
        columns=vectorizer.get_feature_names()
    )
    train_df = train_df.add_prefix(col)

    
    aid_df = X_train[[idx]]

    train_df = train_df.merge(aid_df, left_index =True, right_index=True, suffixes=(False, False), how='inner')
    train_df.set_index(idx, inplace=True)    
    
    test_df =pd.DataFrame(
        features_test.todense(),
        columns=vectorizer.get_feature_names()
    )
    test_df = test_df.add_prefix(col)

    
    aid_test_df = X_test[[idx]]

    test_df = test_df.merge(aid_test_df, left_index =True, right_index=True, suffixes=(False, False), how='inner')
    test_df.set_index(idx, inplace=True)
    
    dev_df =pd.DataFrame(
        features_dev.todense(),
        columns=vectorizer.get_feature_names()
    )
    dev_df = dev_df.add_prefix(col)

    
    aid_dev_df = X_dev[[idx]]

    dev_df = dev_df.merge(aid_dev_df, left_index =True, right_index=True, suffixes=(False, False), how='inner')
    dev_df.set_index(idx, inplace=True)
    return train_df, test_df, dev_df

def extract_n_grams_features(X_train, X_test, X_dev, CV, nr1, nr2,  columns,idx='id'): 

    X_train = X_train.reset_index()
    result_train_df =  X_train[[idx]]
    result_train_df.set_index(idx, inplace=True)
    
    X_test = X_test.reset_index()
    result_test_df =  X_test[[idx]]
    result_test_df.set_index(idx, inplace=True)
    
    X_dev = X_dev.reset_index()
    result_dev_df =  X_dev[[idx]]
    result_dev_df.set_index(idx, inplace=True)
    
    for col in columns:
        result_train_df_, result_test_df_, result_dev_df_ = extract_ngrams(X_train, X_test, X_dev, CV, nr1, nr2, col)
        result_train_df = result_train_df.join(result_train_df_)
        result_test_df = result_test_df.join(result_test_df_)
        result_dev_df = result_dev_df.join(result_dev_df_)
    return result_train_df, result_test_df_, result_dev_df

#read the data for experiments, lemmatize if needed
def read_exp_data (dataset, lemmatized):
    if dataset == 'cross':
        data_path = 'data/cross-topic/{}.csv'
    elif dataset == 'within':
        data_path = 'data/within-topic/{}.csv'
    if lemmatized == False:
        train_df = pd.read_csv(data_path.format('train_rand'),quotechar='"',quoting=csv.QUOTE_ALL,encoding='utf-8',escapechar='\\',doublequote=False, index_col='id')
        test_df = pd.read_csv(data_path.format('test_rand'),header = None, names = ['id','argument1','argument1_id','argument2','argument2_id','debate_id','is_same_side','topic'], quotechar='"',quoting=csv.QUOTE_ALL,encoding='utf-8',escapechar='\\',doublequote=False, index_col=0)
        dev_df = pd.read_csv(data_path.format('dev_rand'),header = None, names = ['id','argument1','argument1_id','argument2','argument2_id','debate_id','is_same_side','topic'],quotechar='"',quoting=csv.QUOTE_ALL,encoding='utf-8',escapechar='\\',doublequote=False, index_col=0)

        print(train_df[:3])
        print('/n')
        print(test_df[:3])
        # 1. Getting train, test and dev data
        X_train, y_train = cut_df(train_df)
        X_test, y_test = cut_df(test_df)
        X_dev, y_dev = cut_df(dev_df)
        print('1')

        # 2. Lemmatizing argument1 and argument2
        X_train = X_train.apply(get_lemma, axis=1)
        X_test = X_test.apply(get_lemma, axis=1)
        X_dev = X_dev.apply(get_lemma, axis=1)
        print('2')

            # 3. save lemmatized arguments
        train = X_train
        train['is_same_side'] = y_train
        test = X_test
        test['is_same_side'] = y_test
        dev = X_dev
        dev['is_same_side'] = y_dev

        train.to_csv(data_path.format('lem/train'))
        test.to_csv(data_path.format('lem/test'))
        dev.to_csv(data_path.format('lem/dev'))
                
    else:
        X_train = pd.read_csv(data_path.format('lem/train'),encoding='utf-8', index_col='id')
        X_test =  pd.read_csv(data_path.format('lem/test'),encoding='utf-8', index_col='id')
        X_dev =  pd.read_csv(data_path.format('lem/dev'),encoding='utf-8', index_col='id')

        y_train = pd.DataFrame(columns = ['is_same_side']) 
        y_train['is_same_side'] = X_train['is_same_side']
        y_test = pd.DataFrame(columns = ['is_same_side']) 
        y_test['is_same_side'] = X_test['is_same_side']
        y_dev = pd.DataFrame(columns = ['is_same_side']) 
        y_dev['is_same_side'] = X_dev['is_same_side']

        X_train = X_train.drop(columns='is_same_side')
        X_test = X_test.drop(columns='is_same_side')
        X_dev = X_dev.drop(columns='is_same_side')
            
    return X_train, X_dev, X_test, y_train, y_dev, y_test

#read the final data, lemmatize if needed
def read_fin_data (dataset, lemmatized):
    if dataset == 'cross':
        data_path = 'data/cross-topic/{}.csv'
    elif dataset == 'within':
        data_path = 'data/within-topic/{}.csv'
            
    traindev_df = pd.read_csv(data_path.format('final/training'),quotechar='"',quoting=csv.QUOTE_ALL,encoding='utf-8',escapechar='\\',doublequote=False,index_col='id')
    test_df =  pd.read_csv(data_path.format('final/test'), index_col='id')
            
    if lemmatized == False:
        # 1. Getting train and dev data
        X_train, X_dev, y_train, y_dev = get_train_dev_sets(traindev_df)
        X_test = test_df
        print('1')

        # 2. Lemmatizing argument1 and argument2
        X_train = X_train.apply(get_lemma, axis=1)
        X_dev = X_dev.apply(get_lemma, axis=1)
        X_test = X_test.apply(get_lemma, axis=1)

        print('2')
                
        # 3. save lemmatized arguments
        train = X_train
        train['is_same_side'] = y_train
        test = X_test
        #test['is_same_side'] = y_test
        dev = X_dev
        dev['is_same_side'] = y_dev

        train.to_csv(data_path.format('lem/train_f'))
        test.to_csv(data_path.format('lem/test_f'))
        dev.to_csv(data_path.format('lem/dev_f'))
                
    else:
        X_train = pd.read_csv(data_path.format('lem/train_f'),encoding='utf-8', index_col='id')
        X_test =  pd.read_csv(data_path.format('lem/test_f'),encoding='utf-8', index_col='id')
        X_dev =  pd.read_csv(data_path.format('lem/dev_f'),encoding='utf-8', index_col='id')
                
        y_train = pd.DataFrame(columns = ['is_same_side']) 
        y_train['is_same_side'] = X_train['is_same_side']
        #y_test = pd.DataFrame(columns = ['is_same_side']) 
        #y_test['is_same_side'] = X_test['is_same_side']
        y_dev = pd.DataFrame(columns = ['is_same_side']) 
        y_dev['is_same_side'] = X_dev['is_same_side']

        X_train = X_train.drop(columns='is_same_side')
        #X_test = X_test.drop(columns='is_same_side')
        X_dev = X_dev.drop(columns='is_same_side')
            
    return X_train, X_dev, X_test, y_train, y_dev

#preparing data for Light GBM
def preprocess_exp_data (X_train, X_train_, X_dev, X_dev_, X_test, X_test_, y_train, y_dev, y_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=True, with_mean=False)

    scaler.fit(X_train_)
    X_train = scaler.transform(X_train_)

    scaler.fit(X_test_)
    X_test = scaler.transform(X_test_)

    scaler.fit(X_dev_)
    X_dev = scaler.transform(X_dev_)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_dev_ = y_dev['is_same_side'].tolist()
    y_test_ = y_test['is_same_side'].tolist()
    y_train_ = y_train['is_same_side'].tolist()

    le.fit(y_dev_)
    y_dev = le.transform(y_dev_)

    le.fit(y_test_)
    y_test = le.transform(y_test_)

    le.fit(y_train_)
    y_train = le.transform(y_train_)

    y_dev = pd.Series(y_dev)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

def preprocess_fin_data (X_train, X_train_, X_dev, X_dev_, X_test, X_test_, y_train, y_dev):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(copy=True, with_mean=False)

    scaler.fit(X_train_)
    X_train = scaler.transform(X_train_)

    scaler.fit(X_test_)
    X_test = scaler.transform(X_test_)

    scaler.fit(X_dev_)
    X_dev = scaler.transform(X_dev_)

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    y_dev_ = y_dev['is_same_side'].tolist()
    y_train_ = y_train['is_same_side'].tolist()

    le.fit(y_dev_)
    y_dev = le.transform(y_dev_)

    le.fit(y_train_)
    y_train = le.transform(y_train_)

    y_dev = pd.Series(y_dev)
    y_train = pd.Series(y_train)

    return X_train, X_dev, X_test, y_train, y_dev

def main():
    """
    Parses input parameters for networks
    """
    import argparse
    global args
    parser = argparse.ArgumentParser(description="LightGBM")
    parser.add_argument('--mode', type=str, default='train_validation', choices=['train_validation', 'train_test'], help="Mode of the system.")
    parser.add_argument('--dataset', type=str, default='within', choices=['within', 'cross'], help="Within or cross dataset.")
    parser.add_argument('--lemmatized', action='store_true', default=False, help = "Lemmatized or not lemmatized data used.")
    parser.add_argument('--vect', action='store_false', default=True, help="Vectorizer used for feature extraction: if True - CountVectorizer, else - TfidfVectorizer.")
    parser.add_argument('--ngram_range_1', type=int, default=3, choices=[3, 1, 1], help="ngram range used for feature extraction - first value.")
    parser.add_argument('--ngram_range_2', type=int, default=3, choices=[3, 1, 2], help="ngram range used for feature extraction - second value.")
    parser.add_argument('--activation_th', type=float, default=0.5, help = "Activation Threshold of output")

    args = parser.parse_args()
    import json
    params = vars(args)
    print(json.dumps(params, indent = 2))
    run()
    
    
def run():
    """
    Execution pipeline for each mode
    """
    mode = args.mode
    dataset = args.dataset
    lemmatized = args.lemmatized
    vect = args.vect
    nr1 = args.ngram_range_1
    nr2 = args.ngram_range_2
    th = args.activation_th
    
    #read, lemmatize, extract ngrams features, prepare data for LightGBM
    if mode == 'train_validation':
        X_train, X_dev, X_test, y_train, y_dev, y_test = read_exp_data(dataset, lemmatized)
        X_train_, X_test_, X_dev_ = extract_n_grams_features(X_train, X_test, X_dev, vect, nr1, nr2, columns=['argument1_lemmas', 'argument2_lemmas'])
        X_train, X_dev, X_test, y_train, y_dev, y_test = preprocess_exp_data (X_train, X_train_, X_dev, X_dev_, X_test, X_test_, y_train, y_dev, y_test)
    elif mode == 'train_test':
        X_train, X_dev, X_test, y_train, y_dev = read_fin_data(dataset, lemmatized)
        X_train_, X_test_, X_dev_ = extract_n_grams_features(X_train, X_test, X_dev, vect, nr1, nr2, columns=['argument1_lemmas', 'argument2_lemmas'])
        X_train, X_dev, X_test, y_train, y_dev = preprocess_fin_data (X_train, X_train_, X_dev, X_dev_, X_test, X_test_, y_train, y_dev)

        
    # create dataset for lightgbm; train on train, using dev as validation set, then save results for test
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_dev = lgb.Dataset(X_dev, y_dev, reference = lgb_train)

    num_test, num_feature = X_train.shape
    
    
    import json
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, f1_score

    def lgb_f1_score(y_hat, data):
        y_true = data.get_label()
        y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
        return 'f1', f1_score(y_true, y_hat), True

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'application': 'binary',
    #    'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    # generate feature names
    feature_name = ['feature_' + str(col) for col in range(num_feature)]

    print('Starting training...')
    # train
    evals_result = {}
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=10,
                    valid_sets=lgb_dev,
                    early_stopping_rounds=5,
                    #feval=lgb_f1_score,
                    feature_name=feature_name,
                    evals_result=evals_result)

    print('Saving model...')
    # save model to file
    gbm.save_model('model.txt')

    print('Dumping model to JSON...')
    # dump model to JSON (and save to file)
    model_json = gbm.dump_model()

    with open('model.json', 'w+') as f:
        json.dump(model_json, f, indent=4)

    # feature names
    #print('Feature names:', gbm.feature_name())

    # feature importances
    #print('Feature importances:', list(gbm.feature_importance()))

    print('Loading model to predict...')
    # load model to predict
    bst = lgb.Booster(model_file='model.txt')
    # can only predict with the best iteration (or the saving iteration)
    y_pred = bst.predict(X_test, num_iteration=gbm.best_iteration)
    # eval with loaded model
    #print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

    #print('Starting predicting...')
    # predict
    #y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    # eval


    #lgb.plot_metric(evals_result, metric='f1')
    if mode == 'train_validation':
        print('Do you want to check classification report and adjust threshold if needed? (y/n)')
        activate_th = input()
        if activate_th == 'y':
            adjust_th(th, y_pred, y_test)

def adjust_th (th, y_pred, y_test):
    from sklearn.metrics import classification_report, confusion_matrix , accuracy_score, f1_score
    #adjust threshold
    threshold = th
    predictions = []
    for i_pred in y_pred.tolist():
        if i_pred >= threshold: predictions.append(1)
        else: predictions.append(0)

    print(classification_report(y_test.tolist(), predictions))
    print('Do you want to adjust threshold? (y/n)')
    activate_th = input()
    if activate_th == 'y':
        print('Type a new threshold value')
        th = float(input())
        adjust_th(th, y_pred, y_test)
    
    
if __name__ == '__main__':
    main()

