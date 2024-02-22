# Supervised learning
## Labeled Data
- Column Header -> Attributes
- Column Data -> Features
- Row -> Observation

## Supervised learning techniques
- Classification -> predicting a descrete class label or category
- Regression -> predicting continuous values
	
## Summary	
- More evaluation methods
- Controlled environment
	

# Unsupervised learning
## Unlabeled data
- unlabeled
- dificult algorithms
- no knowledge about the data
	
## Unsupervised learning techniques
- Dimension reduction
- Density estimation
- Market basket analysis
- Clustering
    - Find patterns and groupings from unlabeled data
	- Discovering structure
	- Summarization
	- Anomaly detection

## Summary	
	Clustering
	Less evaluation methods
	Less controlled environment
	
# Regression
	estimate a condinuous value
		sales forecasting, satisfaction analysis, price estimation
	
	x Independent/explanatory variable
	y Dependent variable
	y = f(x)
	
## Simple regression
	1 independant variable
	simple linear regression
	simple non linear regression
	
## Multiple regression
	> 1 independant variable
	multiple linear regression
	multiple non linear regression
	
## Regression algorithms
	Ordinal regression
	Poisson regression
	Fast fores quantile regression
	Linear, Polynomial, Lasso, Stepwise, Ridge regression
	Bayesian linear regression
	Neural network regression
	Decision fores regression
	Boosted decision tree regression
	KNN (K-neares neighbors)

## Simple linear regression
$$
\widehat{y} = \varTheta_0+\varTheta_1x_1
$$

	response variable
	intercept (Theta 0)
	coefficient od slope/gradiant (Theta 1)
	single predictor

	residual error
		-> Abweichung der regression line vom korrekten Wert
	alle Fehler: MSE Mean squared error
	
	Objective: 
		minimize MSE -> find best parameters for Theta 0 and Theta 1
		
		1) mathematical approach über Durchschnittswerte für x/y
		2) optimization approach

	Pros
	- Very fast
	- no parameter tuning
	- easy to understand, highly interpretable

## Model evaluation
	Approaches
		Train and test on the same dataset
			- High "training accuracy"
				not necessarily good
				result ov over-fitting -> overtrained, non-generalized model
			- Low "out-of-sample accuracy"
				important to have high "out-of-sample accuracy"
		train/test split
			mutually exclusive
			+ more accurate evaluation on out-of-sample accuracy
			- highly dependant on which dataset for train and test
		K-fold cross-validation
			z.B. 4 Folds
				distinct train data
				train/test split -> avg(accuracy der folds)
			
	Metrics
	
		Accuracy
			MAE Mean absolute error
			MSE Mean squared error
			RMSE Root mean squared error
			RAE Relative absolute error
			RSE Relative squared error
$$R^2 = 1 - RSE$$

## Multiple linear regression
$$
\widehat{y} = \varTheta_0+\varTheta_1x_1+\varTheta_2x_2+\dots+\varTheta_nx_n \\
$$


$$
\widehat{y} = \varTheta^TX \\
$$

Weight vector:

$$
\varTheta^T= [\varTheta_0, \varTheta_1, \varTheta_2, \dots]   \\
$$

Feature set:

$$
X = \begin{bmatrix}
   1  \\
   x_1 \\
   x_2 \\
   \dots
\end{bmatrix}
$$

	When to use
		Independent variables effectiveness on prediction
			relative impact of variables on prediction
		Predicting impacts of changes
	
	
	Expose errors with goal:	
	minmize MSE Mean squared error
	
	find parameters 
		Ordinary leas squares
			linear algebra
			takes a long time for large datasets
		optimization arlgorithm
			gradient descent
			propper aproach for very large datasets
	
			
	Concerns
		how to determin when to use simple or multiple linear regression
		independent variables need to be numerical
		how many independent variables should be used? To many may result in an overfit model.
		should the independent variable be categorical or continuous?
			-> dependent must be continuous
			
		what are the linear releationshps between de dependent variable and the independent variables?
		
# K-Nearest neighbors
			
	Classification
		supervised learning approach
		categorize some unknown items into a discrete set of categories or classes
		target attribute is a categorical variable
		multi-class classification
		
	Classification algorithms
		decision trees (ID3, C4.5, C5.0)
		Naive Bayes
		Linear discriminant analysis
		k-neares neighbor
		logistic regression
		neural networks
		support vector machines (SVM)
		
	KNN
		classification algorithm
		classifies cases on their similarity to other cases
		near cases said to be "neighbors"
		similar cases with same class labels are near each other
		1st nearest vs. nth nearest
		
		Algorithm
			pick a value vor K
			calculate the distance of unknown case from all cases
			select the K-observations in the training data that are "nearest" to the unknown datapoint
			predict the respose useing the most popular response value from the K-nearest neighbors
			non popular responses -> noise/anomaly
			
			Similarity/Distance
				multi dimensional vectors for n features in dataset
				
			Choosing K
			low value -> komplex model -> overfitting (non general model)
			high value -> overly generalized model
			calculate accuracy with part of the data, by increasing K -> late use k with the best accuracy
				
			Compute continuous targets using KNN
				KNN can be used for regression
				
	Evaluation metrics for classification
		Jaccard index, Jaccard similarity coeficient (grösse der gemeinsamen Menge von actual/predicted labels)
		F1-score, Confusion matrix Best: 1.0
			bei binary categories -> Precision = TP/(TP + FP), Recall = TP/(TP + FN)
		Logisticy/Log loss, lower is better accuracy
		
# Decision trees
	map all possible decision paths in the form of a tree
	split dataset into distinct nodes
	internal node -> test
	branch -> result of a test
	leaf node -> classification -> goal: 100% pure
	
	Decision tree learning algorithm
		choose an attribute from the dataset
		calculate the significance of the attribute in splitting of data
		split data based on the value of the best attribute
		repeat for the rest of the attributes
		
	Building decision trees
	recursive partitioning -> reducint impurity
	which attribute/feature is the best
		best predictivness/significance
		less impurity
		lower entropy (amount of randomness/uncertainty), 
			entropy = 0 -> 100% pure
			entropy = 1 -> 50%/50%
	pure node -> 100% same category
	information gain -> information that increases the level of certainity after splitting
		IG = (Entropy before split) - (weighted entropy after split)
	
	=> the best attribute is the one with the highest information gain after splitting
	
# Regression trees
    The basic idea behind regression trees is to split the data into groups based on features, like in classification, 
    and return a prediction that is the average across the data we have already seen. 

##Criterion 

    The way the trees are built are similar to classification, but instead of using the entropy criterion. 
    In Classification Trees, we choose features that increase the information gain. In Regression Trees, we choose features that minimize the error. 

    A popular one is the Mean Absolute Error

    Using the criterion and checking every single feature in the dataset to see which one produces the minimal error.

    Conditions that are commonly used to stop growing regression trees: 
        Tree depth
        Number of remaining samples on a branch
        Number of samples on each branch if another decision is made 


# Logistic regression
    like linear but with categorical target
    classification algorithm for categorical variables
    independent variables should be continous, if categorical they need transformation
    dependent variable is binary/categorical

    When to use
        target is categorical spec. binary: 0/1, yes/no, true/false
        if you need probabilistic results: score between 0 and 1
        when you need a linear decision boundary: points on one side of the boundery -> class 1, points on the other side -> class 2
        if you need to understand the impact of a feature

## Logistic vs. linear regression
    Linear
        progression line
        Step function: threshold < line >= 
        nicht geeignet für classification problems

    Sigmoid
        logistc function

    Cost function
        Summe aller Fehler
        change the weight -> reduce the cost
        minimize cost with optimization approach
            Gradient descent 

        if y = 1        
$$
Cost(\widehat{y},y) = -log(\widehat{y})
$$

        if y = 0

$$
Cost(\widehat{y},y) = -log(1 - \widehat{y})
$$

    Gradient descent
        derivative f a cost function
        change parameter values to find minimum of the cost
        Error ball plot
        Learning rate: size of the steps towards the optimal weights

    Training algorithm
        initialize the parameters randomly
        feed the cost function with traning set and calculate the error
        calculate the gradient of cost function
        update weights with new values
        repeat until cost is small enough


      
# Support vector machine SVM
Supervised algorithm that classifies cases by finding a separator

Most datasets are not linearly separable. Solution:

1. Mapping data to a high-dimensional feature space
2. Finding a separator

Separator is a hyperplane (Fläche)

Applications
- Image recognition
- text category assignment
- detecting spam
- sentiment analysis
- gene expression classifcation
- regression, outlier detection, clustering

## Data transformation
Kernelling:
- Linear
- Plynomial
- RBF Radial basis function
- Sigmoid

Optimize hyperplane
- largest separation/margin between the two classes
- point closest to the margin are "support vectors"
- only support vectors matter for achieving the goal, other training samples can be ignored

--> optimization algorithm to maximize the margin, can be soved by gradient descent

Advantages
- accurate in high-dimensional spaces
- memory efficient

Disatvantages
- prone to over-fitting
- no probability estimation
- small datasets



# Multiclass Prediction

Multi-class Classification:
- SoftMax Regression
- One-vs-All 
- One-vs-One 
            
Convert logistic regression to Multi-class classification using multinomial logistic regression or SoftMax regression (generalization of logistic regression). 
SoftMax: not for Support Vector Machines (SVM)
One vs. All (One-vs-Rest) and One vs One: can convert most two-class classifiers to a multi-class classifier
    
## SoftMax Regression
- similar to logistic regression
- training procedure is almost identical to logistic regression using cross-entropy
- prediction is different

1. Use SoftMax function to generate a probability vector
2. prediction using the argmaxargmax function -> class

Geometric interpretation: classes are between hyperplanes

        
## One vs. All (One-vs-Rest)
- number of class labels present in the dataset is equal to the number of generated classifiers
- if we have K classes, we use K two-class classifier models

1. create an artificial class (dummy class)
2. for each classifier, split the data into two classes
    - take the class samples you would like to classify
    - the rest of the samples will be labelled as a dummy class
3. repeat for each class

Issues
- may get multiple classes
- all the outputs may equal ”dummy

## One-vs-One
- split up the data into each class
- train a two-class classifier on each pair of classes

for 3 classes 0,1,2
- train one classifier on the samples that are class 0 and class 1
- train a second classifier on samples that are of class 0 and class 2
- train a final classifier on samples of class 1 and class 2

Classification
- all classifiers predict class
- majority vote: class with the most predictions




# Clustering
- segmentation
- unsupervised
- based on similarity (profiles)

Clustering vs. classification
- classification algorithm: predict categorical class labels, labeled dataset, supervised
- clustering algorithm: unlabeled, unsupervised

Applications
- buying patterns of customers
- recommending new books to customers
- fraud detection
- identify clusters of customers (loyal vs. churn)
- insurance risk
- auto categorize content
- characterize patient behavior
- group genes with similar expression patterns

Used for
- exploratory data analysis
- summary generation, reducing the scale
- outlier detection
- finding duplicates
- pre-processing step

Algorithms
- partition based
    - relatively efficient
    - k-Means, k-Median, Fuzzy c-Means
    - medium/large datasets
- hierarchical
    - produce trees of clusters
    - agglomerative, devisive
    - very intuitive
    - small datasets
- density based
    - produces arbitrary shaped clusters
    - good for spacial clusters
    - good if there is nois in the dataset
    - DBSCAN


## k-means clustering
- unsupervised
- based on similarity
- partitioning
- divides data into non-overlapping subsets (clusters) without any cluster-internal structure or labels
- medium/large datasets
- rel. efficient
- produces sphere-like clusters
- needs number of clusters (k)

Determing similarity or dissimilarity
- distance of samples to each other is used to shape the clusters
- algorithm tries to minimize the intra-cluster and maximize the inter-cluster distances

1-dimensional similarity/distance: 1 feature
2-dimensional similarity/distance: 2 feature -> 2 dimensional space

Distance measurement:
- euclidean distance
- cosine similarity
- average distance

depends on 
- data domain knowledge
- data types of features

Procedure
1. initialize k (represents number of clusters)
    - choose center point for each cluster (centroids), same feature size as dataset
        - randomly k observations of the dataset
        - create k random points (choice)
2. distance calculation
    - distance matrix (distance to centroids for each datapoint)
3. assign each point to the closest centroid
    - find nearest centroid for each datapoint
    - is not resulting in good clusters becaus centroids are chosen randomly
    - high error: Sum of squares error SSE = sum of the squared differences between each point and its centroid
4. compute the new centroids for each cluster
    - centroids move according to their cluster members (mean of all points in the cluster)
5. repeat for new centroids until the centroids no longer move


Issues
- iterative heuristic algorithm
- guarantees that it converges to a result
- but no guarantee that it converge to the global optimum
- result depends on the initial clusters
- not possibly the best outcome

--> common to run the process multiple times with different randomized centroids 

Accuracy
- external approach
    - compare the clusters with the ground truth, if available
    - since unsupervised usually not available
- internal approach
    - average the distance between data points within a cluster

Elbow method for choosing k
- density of the cluster
- increasing k will always decrease the error 
- elbow point: point where the shape of decrease sharply shifts
- right k for clustering




































