Text Mining in Action
========================

:author: Arian Pasquali <dm@arianpasquali.com>
:revision: 1.0
:date: 23/05/2013
   
Introduction
------------
This is a personal project and serves as laboratory to put in practice what I have been studing about text mining. It exploits mainly topics like text clustering, text classification and some evaluation analysis.

Datasets
--------
Scripts provided here tries to be dataset agnostic. 
It means you can provide your own. All you need is to inform the directory with a particular structure.
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

The datasets used to demonstrate this project are available `here <https://www.dropbox.com/s/ax497mv9ioetpbd/data.zip>`_ :

Dataset - `https://www.dropbox.com/s/ax497mv9ioetpbd/data.zip <https://www.dropbox.com/s/ax497mv9ioetpbd/data.zip>`_


Text Classification
--------------------


Applying methods for supervised learning.

Classification algorithms works quite different from clustering. They essencially try to build an statistical model based on samples used for training and try to classify new samples according to that model. For that mean there are many different kinds of algorithms. 

The following script aims to provide simple demonstration for supervised text classificiation in Python performing
experiences with three different algorithms: Support Vector Machines, Naive Bayes and Linear Model. All being dataset agnostic.
More info about the code :doc:`here </source_doc>`. 

Script usage:
``text_classification.py --help`` gives useful output::

	Options:
	  --version             show program's version number and exit
	  -h, --help            show this help message and exit
	  -D DIRECTORY, --dataset=DIRECTORY
	                        Base directory with dataset files
	  -C CLASSIFIER, --classifier=CLASSIFIER
	                        Classifier method: svm, naive_bayes, or linear_model
	                        [default: svm]
	  -N N_ITERATIONS, --cv_n_iterations=N_ITERATIONS
	                        Number of interations for Cross Validation
	  -S TEST_SIZE, --test_size=TEST_SIZE
	                        Test size. float from 0 to 1. [default: 0.2]

Classification results and analysis
####################################
For text classification I performed tests with the following three datasets.

.. toctree::
	:maxdepth: 1

	20newsgroup_small	
	20newsgroup
	wikipedia	


Finding best classifier with model optimization
################################################

Each dataset has its own caracteristics and asks for different approaches. There are so many ways to extract text features and setup a classifier that is hard to test all possibilities for each dataset and find the best mix.

In order to find the best possible classifier I provide an script to find the optimal parameters for the classifier and the feature extraction phase (e.g. C, kernel and gamma for Support Vector Classifier, alpha for Lasso, etc.). I tried to applied these parameters at the `text_classification.py` script.
More info about the code :doc:`here </source_doc>`. 

Script usage:

``find_best_classifier.py --help``.::

	Options:
		--version             show program's version number and exit
		-h, --help            show this help message and exit
		-D DIRECTORY, --dataset=DIRECTORY
		                    Base directory with dataset files
		-C CLASSIFIER, --classifier=CLASSIFIER
		                    Classifier method: svm, naive_bayes, or linear_model
		                    [default: svm]



Text Clustering
--------------------

The whole point of clustering methods is grouping documents according some notion of similarity. When we don't know how many groups or classes are in the dataset we can use clustering methods to find out.

I provide few scripts to deal with it. At `text_clustering.py` I provide an script to simple demonstrate how text clustering can be done using Python. It performs experiences being dataset agnostic. You have the option to choose which algorithm you want, kmeans or hierarchical clustering .

In our case we have labels for each document, but for the clustering task we assume this data is missing and try to group them according to some similarity between them.

More info about the code :doc:`here </source_doc>`.

Script usage:

``text_clustering.py --help``::

	Options:
	--version             show program's version number and exit
	-h, --help            show this help message and exit
	-D DIRECTORY, --dataset=DIRECTORY
	                    Base directory with dataset files
	-C CLUST_METHOD, --clust_method=CLUST_METHOD
	                    Clustering method: hclust, kmeans [default: kmeans]
	-F MAX_FEATURES, --max_features=MAX_FEATURES
	                    optional, None by default. If not None, consider the
	                    top max_features ordered by term frequency across the
	                    corpus
	-M MAX_DF, --max_df=MAX_DF
	                    float in range [0.0, 1.0] or int, optional, 1.0 by
	                    default. When building the vocabulary ignore terms
	                    that have a term frequency strictly higher than the
	                    given threshold (corpus specific stop words). If
	                    float, the parameter represents a proportion of
	                    documents, integer absolute counts. This parameter is
	                    ignored if vocabulary is not None.

.. Using R Scripts
.. ---------------

I provide some scripts to perform clustering analysis in R too. The first tries to find the most appropriate number of clusters based on  silhouette coefficient.
This is done running the algorithm KMeans with different values for `K` and comparing results for the silhouette coefficient. 


You must inform as arguments the dataset path, if you want TF-IDF (Term-Frequency - Inverted Document Frequency) weighting and the max number of `K` values to test. 
The output is a plot of the silhouette coefficient for each tested `k`::

	R -f find_k_clusters.r [dataset_path] [use idf] [max number of clusters]

An example of use with `20newsgroup_small` dataset::

	R -f find_k_clusters.r ../../data/20newsgroup_small/ FALSE 6

There is also another script to plot worldclouds for each found cluster. Wordclouds are good to summarize a group of documents. It is great to visually highlight the most representative words for each cluster. 
This script in R perform clustering and as output plots wordclouds for each cluster.

You must inform as arguments the dataset path, if you want TF-IDF (Term-Frequency - Inverted Document Frequency) weighting, the number of `K` clusters, minimum frequency for the term to be eligible to appear at the worldcloud and max words to appear at the wordcloud.::


	R -f build_worldclouds.r [dataset_path] [use idf] [number of clusters]  [min freq] [max words]

An example of use with `20newsgroup_small` dataset::

	R -f build_worldclouds.r ../../data/20newsgroup_small/ FALSE 3 30 80


Clustering results and analysis
####################################
I made a few tests and analysis for the following two datasets.

.. toctree::
	:maxdepth: 1

	20newsgroup_small_clust
	wikipedia_clust



Additional Notes
------------------

Evaluation metrics
******************
Details for the evaluation metrics applied in this project.
	
.. toctree::
	:maxdepth: 1
	
	evaluation_metrics

Dependencies
************
The python code tries to apply as most as possible the standard Python 2.7.2 library. Only the following external libraries are necessary to install:

	- Scikit-Learn - http://scikit-learn.org/
	- PyLab - http://www.scipy.org/PyLab

References
******************
- Scikit-learn: Working with text data http://scikit-learn.github.io/scikit-learn-tutorial/working_with_text_data.html