> :bulb: Notes on "Peer-graded Assignment: Build a Regression Model in Keras"


# Instructions

# Assignment Topic:

In this project, you will build a regression model using the Keras library to model the same data about concrete compressive strength that we used in labs 3.

# Concrete Data:

For your convenience, the data can be found here again: https://cocl.us/concrete_data

To recap, the predictors in the data of concrete strength include:

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate

# How to submit

You will need to submit your code for each part in a Jupyter Notebook. Since each part builds on the previous one, you can submit the same notebook four times for grading. 

Please make sure that you:

- use Markdown to clearly label your code for each part
- properly comment your code so that your peer who is grading your work is able to understand your code easily
- include your comments and discussion of the difference in the mean of the mean squared errors among the different parts


# Introduction
Project title: Concrete compressive strength regression model

# Objectives
## A. Build a baseline model
Use the Keras library to build a neural network with the following:
- One hidden layer of 10 nodes, and a ReLU activation function
- Use the adam optimizer and the mean squared error  as the loss function.

1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_split helper function from Scikit-learn.
2. Train the model on the training data using 50 epochs.
3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.
4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.
5. Report the mean and the standard deviation of the mean squared errors.

## B. Normalize the data
Repeat Part A but use a normalized version of the data. Recall that one way to normalize the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

How does the mean of the mean squared errors compare to that from Step A?

## C. Increate the number of epochs
Repeat Part B but use 100 epochs this time for training.
How does the mean of the mean squared errors compare to that from Step B?

## D. Increase the number of hidden layers
Repeat part B but use a neural network with the following instead:
- Three hidden layers, each of 10 nodes and ReLU activation function.

How does the mean of the mean squared errors compare to that from Step B?


# Table of Contents
<div class="alert alert-block alert-info" style="margin-top: 20px">

<font size = 3>
    
1. <a href="#item31">Download and Clean Dataset</a>  
2. <a href="#item32">Import Keras</a>  
3. <a href="#item33">Build a Neural Network</a>  
4. <a href="#item34">Train and Test the Network</a>  

</font>
</div>


Concrete compressive strength regression model


































