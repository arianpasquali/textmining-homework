Evaluation
==================

Evaluation for classification methods
---------------------------------------------

Typical evaluation metrics come for classification tasks: Accuracy, Precision, Recall and F-measure.

	TP - True positives
	FP - False positives
	TN - True negatives
	FN - False negatives

Accuracy
##########

Percentage of correct answers.

	Accuracy = (TP + TN) / (TP + FP + FN + TN)

Precision
##########

Proportion of the predictions of the model that are correct

	P = TP=(TP + FP)

Recall
##########
Proportion of the real events that are captured by the model

	R = TP=(TP + FN)

F1
##########
The traditional F-measure or balanced F-score (F1 score) is the harmonic mean of precision and recall.

	F1 = 2 * (precision * recall) / (precision + recall)


K-Fold Cross Validation 
************************

The data is partitioned into k equally (or nearly equally) sized segments (folds).
k iterations of training and testing are performed.
In each iteration fold k is used for testing and the remaining k  1 folds for training.
This means, there are k different classifers learned.
The test results of all folds are combined to form a total result (usually averaged).
A typical value for k is 10.


Evaluation for clustering methods
-----------------------------------------

Homogeneity
############

In order to satisfy our homogeneity criteria, a
clustering must assign only those datapoints that are
members of a single class to a single cluster. That is,
the class distribution within each cluster should be
skewed to a single class, that is, zero entropy.

Completeness
############

Completeness is symmetrical to homogeneity. In
order to satisfy the completeness criteria, a clustering must assign all of those datapoints that are members of a single class to a single cluster.

V-Measure
##########

V-measure is an entropy-based measure which explicitly measures how successfully the criteria of homogeneity and completeness have been satisﬁed. Vmeasure is computed as the harmonic mean of distinct homogeneity and completeness scores, , just as precision and recall are commonly combined into F-measure (Van Rijsbergen, 1979). As F-measure scores can be weighted, V-measure can be weighted
to favor the contributions of homogeneity or completeness.

Silhouette Coefficient
######################
This popular method combines both cohesion and separation ideas. 
The value of the silhouette can vary between -1 and 1. Negative value is undesirable. 
We want positive values, as close to 1 as possible. As most close to 0 it means there are overlaps between clusters. 

------------------------------------------------------------

Reference
----------
- Cluster Analysis: Basic Concepts and Algorithms  (http://www-users.cs.umn.edu/~kumar/dmbook/ch8.pdf)
- V-Measure: A conditional entropy-based external cluster evaluation measure (http://acl.ldc.upenn.edu/D/D07/D07-1043.pdf)
- Machine Learning evalutation (http://www.ims.uni-stuttgart.de/institut/mitarbeiter/kesslewd/lehre/sentimentanalysis12s/ml_evaluation.pdf)
- Performance measures (http://www.cs.cornell.edu/courses/cs578/2003fa/performance_measures.pdf)
