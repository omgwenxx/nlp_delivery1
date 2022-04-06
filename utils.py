# import libraries and set randome seed for reproducibility
import pandas as pd
import numpy as np
import sklearn
import scipy
from sklearn import *
import os
import pickle  # to save model
import nltk
# nltk.download('punkt') # download if not exist
# nltk.download('stopwords') # download if not exist
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

RANDOM_SEED = 123  # taken from task description
import pickle


def to_string(x):
    return str(x)


def convert_questions(df):
    """
    converts columns of quora dataframe to strings
    :param df: quora dataframe, expected to have question1 and question2 columns
    :return: none, using
    """
    for col in df.columns:
        if col == "question1":
            df.loc[:, (col)] = df.loc[:, (col)].apply(to_string).astype("string")  # to avoid warning
        if col == "question2":
            df.loc[:, (col)] = df.loc[:, (col)].apply(to_string).astype("string")  # to avoid warning


def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    # assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(df["question1"])
    X_q2 = count_vectorizer.transform(df["question2"])
    X_q1q2 = scipy.sparse.hstack((X_q1, X_q2))
    ############### End exercise ###################

    return X_q1q2


def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    # assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted = cast_list_as_strings(list(df["question1"]))
    q2_casted = cast_list_as_strings(list(df["question2"]))

    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)
    X_q1q2 = scipy.sparse.hstack((X_q1, X_q2))
    ############### End exercise ###################

    return X_q1q2


def diff(li1, li2):
    """
    Gets two lists and returns their difference, intended for finding the difference between word arrays
    :param li1: first list
    :param li2: second list
    :return: list with different values between lists
    """
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))


def save_model(model_name, model):
    """
    Saves sklearn model to disk in order to load in later
    :param model_name: string with prefix of the file
    :param model: variable holding classifier model
    :return: None
    """
    filename = model_name + '.sav'
    pickle.dump(model, open(filename, 'wb'))


def simple_preprocess(train_df, test_df):
    """
    Function in order to use in reproduce_results.ipynb. Takes as input
    Dataframes with quora train and test set. Uses CountVectorizer for
    preprocessing and returns finalized dataframes that
    can be used for training/evaluation.
    :param train_df: Dataframe with quora_train_data.csv data
    :param test_df: Dataframe with quora_test_data.csv data
    :return: returns pre
    """

    # extract questions (documents) and cast to strings
    q1_train = cast_list_as_strings(list(train_df["question1"]))
    q2_train = cast_list_as_strings(list(train_df["question2"]))
    all_questions = q1_train + q2_train

    # fit on train set
    count_vectorizer_v1 = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1, 1))
    count_vectorizer_v1.fit(all_questions)

    X_tr_q1q2 = get_features_from_df(train_df, count_vectorizer_v1)
    X_te_q1q2 = get_features_from_df(test_df, count_vectorizer_v1)

    return X_tr_q1q2, X_te_q1q2


def preprocess(sentence):
    """
    Function to preprocess sentences in question1 and question2 column. First words are lower cased,
    then punctuation and stop words (using nltk library) are removed. Then we stem the words
    to remove pre- or postfixes and return array of tokens.
    :param sentence: sentence representing questions1 or question2 feature from dataframe
    :return: array of processed sentence word tokens
    """
    text = sentence.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    #print(text)
    text = text.lower()  # lower case
    #print(text)
    text_tokens = word_tokenize(text)  # tokenizing words
    #print(text_tokens)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]  # remove stop words

    ps = PorterStemmer()
    tokens_stem = [ps.stem(word) for word in tokens_without_sw]
    return " ".join(tokens_stem)

stemmer = PorterStemmer()
def stemmed_words(tokens):
    """
    To be used in tokenize method to preprocess sentences.
    :param tokens: tokenized sentence
    :return: stemmed tokens
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    """
    To be used in CounterVectorizer object of sklearn in parameter tokenizer.
    :param text: sentence as string
    :return: stemmed tokens
    """
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    tokens = [i for i in tokens if i not in stopwords.words()]
    stems = stemmed_words(tokens)
    return stems