#!/usr/bin/env python

import sklearn.cross_validation
import sklearn.datasets
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import pylab as pl
import numpy as np

import sys ,os
from optparse import OptionParser
from pprint import pprint

DEFAULT_DATASET = "/data/20newsgroup_small"
# DEFAULT_DATASET = "/data/20newsgroup"
# DEFAULT_DATASET = "/data/wikipedia"

def load_data(directory):
    """
Load dataset from informed directory.

Load text files with categories as subfolder names.
Individual samples are assumed to be files stored a two levels folder structure such as the following:
container_folder/
    category_1_folder/
        file_1.txt 
        file_2.txt 
        ... 
        file_42.txt
    category_2_folder/
        file_43.txt 
        file_44.txt 
        ...
    """
    dataset = sklearn.datasets.load_files(directory)
    
    print "Loaded %d documents" % len(dataset.data)
    print "Loaded %d categories" % len(dataset.target_names)
    print "Categories ",dataset.target_names
    print

    return dataset


def plot_confusion_matrix(cm):
    # Print confusion matrix from last result
    pl.matshow(cm)
    pl.title('Confusion matrix')
    pl.colorbar()
    pl.show()

def print_metrics(scores):    
    """
    Compute and print evaluation metrics.
    Accuracy, Precision, Recall and F1.
    """
    print
    print "--------- Final results for Cross Validation ---------"
    print "Avg Accuracy",(np.sum(scores["accuracy"])/cmd_options.n_iterations)
    print "Avg Precision",(np.sum(scores["precision"])/cmd_options.n_iterations)
    print "Avg Recall",(np.sum(scores["recall"])/cmd_options.n_iterations)
    print "Avg F1",(np.sum(scores["f1"])/cmd_options.n_iterations)
    print

def perform_classification(dataset):
    """
Perform classification with cross validation.
    """

    # Perform Cross Validation
    # The ShuffleSplit iterator will generate a user defined number of 
    # independent train / test dataset splits. 
    # Samples are first shuffled and then splitted into a pair of train and test sets.

    ss = sklearn.cross_validation.ShuffleSplit(len(dataset.data), 
        n_iter=cmd_options.n_iterations, test_size=cmd_options.test_size,random_state=1234)
    
    scores = {"accuracy":[],"precision":[],"recall":[],"f1":[]}

    print 
    print "Number of documents for training",(1 -cmd_options.test_size )* len(dataset.data)
    print "Number of documents for testing",cmd_options.test_size * len(dataset.data)
    print 

    fold = 0
    for train_index, test_index in ss:
        fold +=1 
        print "Fold", fold, "--------------------------------"
        
        # get the text and label for the training data
        train_text = [dataset.data[x] for x in train_index]
        train_target = [dataset.target[x] for x in train_index]

        # label and text for the test data
        test_text = [dataset.data[x] for x in test_index]
        test_target = [dataset.target[x] for x in test_index]

        print "Performing classification with method ", cmd_options.classifier.upper()
        print


        if(cmd_options.classifier == "svm"):
            # Best score: 1.000
            # Best parameters set:
            #     clf_svm__C: 2.0
            #     tfidf__use_idf: False
            #     vect__max_df: 1.0
            #     vect__max_features: 5000
            #     vect__ngram_range: (1, 5)

            vectorizer = CountVectorizer( max_df=1.0, 
                                        max_features=5000, 
                                      ngram_range=(1, 5),
                                      charset_error='ignore',
                                      # stop_words='english',
                                      lowercase=True,
                                      strip_accents="unicode"
                                      )

            text_clf = Pipeline([('vect', vectorizer),
                                ('tfidf', TfidfTransformer(use_idf=False)),
                                ('clf', SVC(C=2.0,kernel='linear')), ])


        if(cmd_options.classifier == "linear_model"):
            # Best score: 0.997
            # Best parameters set:
            #     clf_linear_model__alpha: 1e-05
            #     clf_linear_model__penalty: 'l2'
            #     tfidf__use_idf: False
            #     vect__max_df: 0.75
            #     vect__max_features: 5000
            #     vect__ngram_range: (1, 5)


            vectorizer = CountVectorizer( max_df=0.75, 
                                        max_features=5000, 
                                      ngram_range=(1, 5),
                                      charset_error='ignore',
                                      # stop_words='english',
                                      lowercase=True,
                                      strip_accents="unicode"
                                      )

            text_clf = Pipeline([('vect', vectorizer),
                                ('tfidf', TfidfTransformer(use_idf=False)),
                                ('clf', SGDClassifier(alpha=0.00001,penalty="l2")), ])




        if (cmd_options.classifier == "naive_bayes"):

            # Best score: 1.000
            # Best parameters set:
            #     tfidf__use_idf: False
            #     vect__max_df: 0.5
            #     vect__max_features: None
            #     vect__ngram_range: (1, 5)

                vectorizer = CountVectorizer( max_df=0.5, 
                                        max_features=None, 
                                      ngram_range=(1, 5),
                                      charset_error='ignore',
                                      # stop_words='english',
                                      lowercase=True,
                                      strip_accents="unicode"
                                      )

                text_clf = Pipeline([('vect', vectorizer),
                                    ('tfidf', TfidfTransformer(use_idf=False)),
                                    ('clf', MultinomialNB()), ])
        #train
        _ = text_clf.fit(train_text, train_target)

        #predict
        predicted = text_clf.predict(test_text)

        #compute metrics
        accuracy = metrics.accuracy_score(test_target,predicted)
        precision = metrics.precision_score(test_target,predicted)
        recall = metrics.recall_score(test_target,predicted)
        f1 = metrics.f1_score(test_target,predicted)

        scores["accuracy"].append(accuracy)
        scores["precision"].append(precision)
        scores["recall"].append(recall)
        scores["f1"].append(f1)

        print "**** Classification report ****"
        print metrics.classification_report(test_target, predicted,
                                            target_names=dataset.target_names)

        # Compute confusion matrix
        print "**** Confusion matrix ****"
        cm = confusion_matrix(test_target, predicted)
        print cm 
        print

    print_metrics(scores)

    # if run without CV plot Confusion Matrix
    if(cmd_options.n_iterations == 1):
        plot_confusion_matrix(cm)

def main():
    dataset_dir = ""
    if not (cmd_options.directory):
        cmd_parser.print_help()
        print
        print "Loading default dataset. 20newgroup small"

        current_file = os.path.realpath(__file__)
        basedir = os.path.abspath(os.path.join(current_file, os.pardir))
        dataset_dir = basedir + DEFAULT_DATASET

    else:
        print
        print "Loading dataset from ",cmd_options.directory
        dataset_dir = cmd_options.directory

    print
    print "Parameters{"
    for option in cmd_options.__dict__.keys():
        print "     ", option,":",cmd_options.__dict__[option]
    print "}"


    # load dataset
    dataset = load_data(dataset_dir)
    # compute classes
    perform_classification(dataset)

if __name__ == "__main__":
    # Command Line Arguments Parser
    desc="""Automatic Supervised Text Classification laboratory.
This program aims to present an introdution to supervised text classficiation.
It performs experiences using SVM, Naive Bayes and Linear Model being dataset agnostic.
    """

    cmd_parser = OptionParser(version="%prog 0.1",description=desc)
    
    cmd_parser.add_option("-D", "--dataset", type="string", action="store", dest="directory", help="Base directory with dataset files")    
    cmd_parser.add_option("-C", "--classifier", default="svm", dest="classifier",
                  help="Classifier method: svm, naive_bayes, or linear_model [default: %default]")
    cmd_parser.add_option("-N", "--cv_n_iterations", default=1, dest="n_iterations", type="int",
                  help="Number of interations for Cross Validation")
    cmd_parser.add_option("-S", "--test_size", default=0.2, dest="test_size", type="float",
                  help="Test size. float from 0 to 1. [default: %default]")
    

    (cmd_options, cmd_args) = cmd_parser.parse_args()

    allowed_classifiers = ["linear_model","svm","naive_bayes"]

    if(cmd_options.classifier):
        if not any(cmd_options.classifier in s for s in allowed_classifiers):
            print
            print "Sorry.",cmd_options.classifier, " is not an allowed classifier method"
            cmd_parser.print_help()
            sys.exit(1)

    main()