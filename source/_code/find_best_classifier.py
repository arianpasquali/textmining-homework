#!/usr/bin/env python

from pprint import pprint
from time import time
import logging

import sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

import sys ,os
from optparse import OptionParser
from pprint import pprint

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


def build_pipeline(classifier):
    """
    Define a pipeline combining a text feature extractor with a simple classifier
    """

    vecttorizer = CountVectorizer(charset_error='ignore',
                                # stop_words='english',
                                strip_accents="unicode")

    if(classifier == "svm"):
        pipeline = Pipeline([
            ('vect', vecttorizer),
            ('tfidf', TfidfTransformer()),
            ("clf_svm", SVC(kernel='linear'))
        ])
        
    if(classifier == "linear_model"):
        pipeline = Pipeline([
            ('vect', vecttorizer),
            ('tfidf', TfidfTransformer()),
            ("clf_linear_model", SGDClassifier())
        ])

    if(classifier == "naive_bayes"):
        pipeline = Pipeline([
            ('vect', vecttorizer),
            ('tfidf', TfidfTransformer()),
            ("clf_naive_bayes", MultinomialNB())
        ])

    return pipeline

def build_parameters(classifier):
    """
    Define parameters to be evaluated
    """

    parameters = {
        # uncommenting more parameters will give better exploring power but will
        # increase processing time in a combinatorial way
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__lowercase': (True,False),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 2),(1,5)),  # unigrams and bigrams, or unirams to 5grams
        'tfidf__use_idf': (True, False),
    }

    if(classifier == "svm"):
      parameters['clf_svm__C'] = (1.0, 2.0)  

    if(classifier == "linear_model"):
        parameters['clf_linear_model__alpha'] = (0.00001, 0.000001)
        parameters['clf_linear_model__penalty'] = ('l2', 'elasticnet')
    
    return parameters

# get data
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

def perform_optimization(data):
    """
    Find the best parameters for both the feature extraction and the classifier
    """
    print "%d documents" % len(data.filenames)
    print "%d categories" % len(data.target_names)
    print

    pipeline = build_pipeline(cmd_options.classifier)
    parameters = build_parameters(cmd_options.classifier)

    grid_search = GridSearchCV(pipeline, 
                               parameters, 
                               n_jobs=-1, verbose=1)

    print "Performing grid search..."
    print "pipeline:", [name for name, _ in pipeline.steps]
    print "parameters:"
    pprint(parameters)
    t0 = time()
    grid_search.fit(data.data, data.target)
    print "done in %0.3fs" % (time() - t0)
    print

    print "Best score: %0.3f" % grid_search.best_score_
    print "Best parameters set:"
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t%s: %r" % (param_name, best_parameters[param_name])

if __name__ == "__main__":
    # Command Line Arguments Parser
    desc="""Automatic Supervised Text Classification laboratory. This program aims to find the best parameters for both the feature extraction and the classifier.
    """

    cmd_parser = OptionParser(version="%prog 0.1",description=desc)
    
    cmd_parser.add_option("-D", "--dataset", type="string", action="store", dest="directory", help="Base directory with dataset files")    
    cmd_parser.add_option("-C", "--classifier", default="svm", dest="classifier",
                  help="Classifier method: svm, naive_bayes, or linear_model [default: %default]")

    (cmd_options, cmd_args) = cmd_parser.parse_args()

    if not (cmd_options.directory):
        cmd_parser.print_help()
        sys.exit(1)

    allowed_classifiers = ["linear_model","svm","naive_bayes"]

    if(cmd_options.classifier):
        if not any(cmd_options.classifier in s for s in allowed_classifiers):
            print
            print "Sorry.",cmd_options.classifier, " is not an allowed classifier method"
            cmd_parser.print_help()
            sys.exit(1)

    dataset = load_data(cmd_options.directory)
    perform_optimization(dataset)
