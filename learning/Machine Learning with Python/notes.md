# Supervised learning
## Labeled Data
	Column Header -> Attributes
	Column Data -> Features
	Row -> Observation

## Supervised learning techniques
	Classification -> predicting a descrete class label or category
	Regression -> predicting continuous values
	
## Summary	
	More evaluation methods
	Controlled environment
	

# Unsupervised learning
## Unlabeled data
	unlabeled
	dificult algorithms
	no knowledge about the data
	
## Unsupervised learning techniques
	Dimension reduction
	Density estimation
	Market basket analysis
	Clustering
		Find patterns and groupings from unlabeled data
		Discovering structure
		Summarization
		Anomaly detection

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

        if y = 1        
$$
Cost(\widehat{y},y) = -log(\widehat{y})
$$

        if y = 0

$$
Cost(\widehat{y},y) = -log(1 - \widehat{y})
$$

        

    
            

    
    


        














































