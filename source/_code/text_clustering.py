#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans,Ward
import sklearn.datasets
import numpy as np
from optparse import OptionParser
import sys,os
from time import time
import logging
import pylab as pl

import mpl_toolkits.mplot3d.axes3d as p3

# from pytagcloud import create_tag_image, make_tags
# from pytagcloud.lang.counter import get_tag_counts

# from wordcloud import make_wordcloud

# print wordcloud

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

DEFAULT_DATASET = "/data/20newsgroup_small"
# DEFAULT_DATASET = "/data/20newsgroup"
# DEFAULT_DATASET = "/data/wikipedia"

def reduce_dems(data):
    """
    Reduces dimensions using PCA.  
    """
    rpca=RandomizedPCA(n_components=2)
    return rpca.fit_transform(data)

def plot_wordclouds(dataset,clusters,features):
    """
    Compute and plot wordcloud for each cluster
    """

    doc_limit_max = 5
    for cluster_id in range(clusters.n_clusters):
        doc_limit = 0
        idx = 0
        wordcloud_text = ""
        print "cluster ", cluster_id
        for cluster_label in clusters.labels_:
            if(doc_limit <= doc_limit_max):
                print idx,cluster_label
                if(cluster_id == cluster_label):
                    print dataset.data[idx]
                    wordcloud_text = wordcloud_text + dataset.data[idx]
                    doc_limit+=1
                idx+=1

        cv = CountVectorizer(min_df=1, charset_error="ignore",
                         stop_words="english", max_features=200)

        counts = cv.fit_transform([wordcloud_text]).toarray().ravel()
        words = np.array(cv.get_feature_names())
        # throw away some words, normalize
        words = words[counts > 1]
        counts = counts[counts > 1]
        output_filename = str(cluster_id)+ "_wordcloud.png"
        make_wordcloud(words, counts, output_filename)

        # tags = make_tags(get_tag_counts(wordcloud_text), maxsize=80)
        # print tags
        # create_tag_image(tags, str(cluster_id) + '_wordcloud.png', size=(900, 700))

    # # get the text and label for the training data
    # train_text = [dataset.data[x] for x in train_index]
    # train_target = [dataset.target[x] for x in train_index]

    # # label and text for the test data
    # test_text = [dataset.data[x] for x in test_index]

    # # print features.data
    # print dataset.keys()
    # # print dataset.target_names
    # print dataset.target
    # print dataset.data.keys()

def plot(kmeans,reduced_data):
    """
    Plot results
    """

    kmeans.fit(reduced_data)
    h = 0.1
    x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
    y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # pl.figure(1)
    # pl.clf()

    # pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=10)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_

    print centroids
    # pl.scatter(centroids[:, 0], centroids[:, 1],
    #            marker='x', s=20, linewidths=3,
    #            color='r', zorder=10)

    # print TfidfVectorizer.inverse_transform()

    # pl.title('K-means clustering on selected 20_newsgroup')
    # pl.xlim(x_min, x_max)
    # pl.ylim(y_min, y_max)
    # pl.xticks(())
    # pl.yticks(())
    # pl.show()

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

def extract_features(dataset):
    """
    Extract features from text.
    It uses term frequency / inverted document frequency method to vectorize the text.
    """
    vectorizer = TfidfVectorizer( max_df=cmd_options.max_df, 
                                  max_features=cmd_options.max_features,
                                  smooth_idf=True,
                                  charset_error='ignore',
                                  stop_words='english',
                                  lowercase=True,
                                  strip_accents="unicode",
                                  use_idf=True)

    X = vectorizer.fit_transform(dataset.data)
    # print Y
    # print X.__dict__.keys()
    # print X.data
    # print X.indices
    # print X.format
    # print X.maxprint

    print "n_samples: %d, n_features: %d" % X.shape
    print

    return X

def print_metrics(dataset,km,features):    
    """
    Compute and print evaluation metrics.
    Homogeneity, completeness, v-measure, adjusted rand-index and silhouette coefficient.
    """

    labels = dataset.target
    sample_size = len(dataset.data)
    print 
    print "*************** Clustering evaluation metrics ***************"
    print " Homogeneity: %0.6f" % metrics.homogeneity_score(labels, km.labels_)
    print " Completeness: %0.6f" % metrics.completeness_score(labels, km.labels_)
    print " V-measure: %0.6f" % metrics.v_measure_score(labels, km.labels_)
    print " Adjusted Rand-Index: %.6f" % metrics.adjusted_rand_score(labels, km.labels_)
    print " Silhouette Coefficient: %0.6f" % metrics.silhouette_score(features, km.labels_, 
                                                sample_size=sample_size)
    print 

def compute_clusters(dataset,features_vector):
    """
    Apply clustering method
    """

    labels = dataset.target
    true_k = np.unique(labels).shape[0]
    
    # Run clustering method
    print "Performing clustering with method ", cmd_options.clust_method.upper()
    print

    if(cmd_options.clust_method == "hclust"):
        result = features_vector.toarray()
        ward = Ward(n_clusters=true_k)
        ward.fit(result) 

        return ward

    if(cmd_options.clust_method == "kmeans"):
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000, verbose=1)
        km.fit(features_vector)

        return km
        
    ###############################################################################
    # labels = ward.labels_
    # Plot result
    # fig = pl.figure()
    # ax = p3.Axes3D(fig)
    # ax.view_init(7, -80)
    # for l in np.unique(labels):
    #     ax.plot3D(X[labels == l, 0], X[labels == l, 1], X[labels == l, 2],
    #               'o', color=pl.cm.jet(np.float(l) / np.max(labels + 1)))
    # pl.title('Without connectivity constraints')
    # pl.show()
    
def perform_clustering(dataset):
    """
Process the whole pipeline. 
    - 1. Extract features
    - 2. Apply clustering method
    - 3. Present evaluation metrics
    """

    # extract features from documents
    features = extract_features(dataset)

    # run clustering
    clusters = compute_clusters(dataset,features)

    # print metrics
    print_metrics(dataset,clusters,features)

    # if(cmd_options.wordcloud):
    #     plot_wordclouds(dataset,clusters,features)

    # reduce dimensions to plot result
    # reduced = reduce_dems(features)
    # plot(cluster,reduced)

def main():
    dataset_dir = ""
    if not (cmd_options.directory):
        cmd_parser.print_help()
        print 
        print "You didn't define your dataset."
        print "Loading default dataset: [20newgroup_small]"

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
    perform_clustering(dataset)

if __name__ == "__main__":
    desc="""Text Clustering laboratory.
This program aims to present an introdution to text clustering.
It performs experiences using kmeans and hierarchical clustering being dataset agnostic.
    """

    cmd_parser = OptionParser(version="%prog 0.1",description=desc)

    cmd_parser.print_version()

    cmd_parser.add_option("-D", "--dataset", type="string", action="store", dest="directory", help="Base directory with dataset files")    
    cmd_parser.add_option("-C", "--clust_method", default="kmeans", dest="clust_method",
                  help="Clustering method: hclust, kmeans [default: %default]")

    cmd_parser.add_option("-F", "--max_features", default=5, dest="max_features", type="int",
                  help="optional, None by default. If not None, consider the top max_features ordered by term frequency across the corpus")

    cmd_parser.add_option("-M", "--max_df", default=0.4, dest="max_df", type="float",
                  help="float in range [0.0, 1.0] or int, optional, 1.0 by default. When building the vocabulary ignore terms that have a term frequency strictly higher than the given threshold (corpus specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.")

    cmd_parser.add_option("-W", "--wordcloud", default=0.4, dest="wordcloud", type="float",
                  help="Compute wordcloud")
    (cmd_options, cmd_args) = cmd_parser.parse_args()

    # print cmd_options

    allowed_clust_methods = ["hclust","kmeans"]

    if(cmd_options.clust_method):
        if not any(cmd_options.clust_method in s for s in allowed_clust_methods):
            print
            print "Sorry.",cmd_options.clust_method, " is not a an allowed clustering method"
            cmd_parser.print_help()
            sys.exit(1)

    main()