![Machine Learning with Python](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Deep_Learning_and_Neural_Networks.jpg "Machine Learning with Python")

> :bulb: Notes on "Deep Learning & Neural Networks with Keras"

# Keras resources
[Keras Activation Functions](https://keras.io/activations)
[Keras Models](https://keras.io/models/about-keras-models/#about-keras-models)
[Keras Optimizers](https://keras.io/optimizers)
[Keras Metrics](https://keras.io/metrics)



# Deep learning
Applications
- color Restoration from b/w images (Convolutional neuronal networks)
- speech reenactment, synching lip movements audio to video, video to video
- automatic handwriting generation (recurrent neural networks)
- automatic machine translation
- automaticly adding sounds to silent movies
- object classification and detection in images
- self driving cars


# Neural networks - making predictions
## Neurons and neural networks
Neuron
    - soma: main body
    - dendrites: network of arms sticking out of the body
    - axon: long arm that sticks out of the soma
    - synapses: whiskers at the end of the axon (terminal buttons)

Learning in the brain occurs by repeatedly activating certain neural connections over others, and this reinforces those connections. This makes them more likely to produce a desired outcome given a specified input.  Once the desired outcome
occurs, the neural connections causing that outcome become strengthened.

Artificial neuron
- soma
- dendrites
- axon (can branch of to connect to many other neurons)

## Artificial neural networks
Layers
    - input layer
    - hidden layers
    - output layer

Main topics
- forward propagagion
- backpropagation
- activation functions (Non-linear transformations like the sigmoid function)


Forward propagation
- process through which data passes through layers of neurons in a neural network from the input layer all the way to the output layer.
- Every connection has a specific weight by which the flow of data is regulated. 
- x: input, w: weight
- b: bias (constant)
- a (output) = z (linear combination of the inputs/weight/bias)

$$ a = z $$

$$ z_1 = x_1w_1+b_1 $$

$$ a_1 = f(z_1) $$

$$ z_2 = w_2a_1+b_2 $$

$$ a_2 = f(z_2) $$

Activation functions
- map the weighted sum to a nonlinear space (sigmoid function)
- + unendlich: 1, - unendlich: 0
- activation functions decide whether a neuron should be activated or not (received info is relevant/ignored by neuron)


Neural network without activation function -> linear regression model
Activation function performs non-linear transformation to the input enabling the neural network of learning and performing more complex tasks, such as image classifications and language translations.

Example

    ### inputs, weights, bias
    x1 = 0.5
    x2 = -0.35
    w1 = 0.55
    w2 = 0.45
    b1 = 0.15
        
    ### weighted sum of the inputs
    z = x1*w1+x2*w2+b1
    
    ### sigmoid activation function
    a = 1.0 / (1.0 + np.exp(-z))
    print('The output of the network is {}'.format(np.around(a, decimals=4)))

## Training a neural network

### Gradient descent
Cost function

$$ Z = wX $$

find the best value for w resulting in the minimum value for the cost or loss function

Gradient descent algorithm
- random initial w0 value
- learning rate (size of the step towards optimal w
- next w1
- number of iterations -> epochs

### Backpropagation
1.  calculate E, the error between the ground truth T and the estimated output
2.  propagate the error back into the network and update each weight and bias as per the following equations:


$$ E = { 1 \over 2}(T - a_2)^2 $$

$$ w_i \to w_i - \eta{\partial E \over \partial w_i} $$

$$ w_2 \to w_2 - (-(T-a_2))(a_2(1-a_2))(a_1) $$

$$ b_2 \to b_2 - (-(T-a_2))(a_2(1-a_2))1 $$

$$ w_1 \to w_1 - (-(T-a_2))(a_2(1-a_2))(w_2)(a_1(1-a_1)x_1 $$

$$ b_1 \to b_1 - (-(T-a_2))(a_2(1-a_2))(w_2)(a_1(1-a_1)1 $$


Complete training algorithm:
1. Initialize the weights and biases
2. Iteratively repeat the following steps
    1. calculate network output using forward propagation
    2. calculate the error beween the ground truth and estimated/predicted output
    3. update weights and biases through backpropagation
    4. repeat the above three steps until number of iteration/epochs is reached od the error is beween threshold

### Vanishing gradient
Problem with the sigmoid activation function that prevented neural networks from booming sooner.

We are using the sigmoid function as the activation function. 
All the intermediate values in the network are between 0 and 1.
So when we do backpropagation, we are multiplying factors that are less than 1 by each other.
With this the gradients get smaller and smaller.
This means that the neurons in the earlier layers learn very slowly.

Do not use sigmoid or similar for activation, since the are prone to the vanishing gradient proglem.

### Activation functions
- Binary step function
- Linear function
- * Sigmoid function
- * tanh (Hyperbolic tangent function)
- * ReLU (rectified linear unit)
- Leaky ReLU
- * Softmax function


Sigmoid function
- widely used as activation functions in the hidden layers of a neural network
- function is pretty flat beyond the +3 and -3 region -> gradients become very small
- vanishing gradient problem
- values only range from 0 to 1. All positive values. Function is not symmetric around the origin.

Hyperbolic tangent function (tanh)
-  just a scaled version of the sigmoid function
-  symmetric over the origin
-  ranges from -1 to +1
- vanishing gradient problem in very deep neural networks

ReLU function
- most widely used today
- does not activate all neurons at the same time
- if input is negative it will be convertet to 0 and neuron is not activated
- only few neurons are activated -> sparse and efficient network
- only used in the hidden layers
- no vanishing gradient problem

Softmax function
- sigmoid function
- for classification problems
- ideally used in the output layer of the classifier


Conclusion
- when building a model, begin with ReLU
- switch to other functions later
- avoid sigmoid or tanh functions


# Deep learning libraries
by popularity:
- TensorFlow
- Keras
- PyTorch
- Theano -> no longer supported

Keras vs. PyTorch vs. TensorFlow
- TensorFlow (Google) most popular
    - developed/used at Google
    - more control
    - dificult to use

- PyTorch is a cousin of the Lua-based Torch framework
    - fast
    - academic research
    - deep learning requiring optimizing custom expressions
    - supported/used at Facebook
    - more control
    - dificult to use
- Keras
    - high level API
    - easy to use
    - syntactic simlicity
    - runs on top of a low-level library such as TensorFlow
    - supported by Google
    
## Regression with Keras
- features of dataset -> nodes in input layer
- n hidden layers
- output layer with dependent variable as node
- all neurons in a layer are connected to all neurons in the next layer (dense network)

predictors and target
- predictors: independent features
- target: dependent column

Keras Code for a regression model:

    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    n_cols = data.shape[1] ### number of columns

    ### add dense layers
    model.add(Dense(5, activation='relu', input_shape=(n_cols,))) ### 1. hidden layer
    model.add(Dense(5, activation='relu')) ### 2. hidden layer
    model.add(Dense(1) ### output layer

    ### train model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(predictors, target)

    ### test model
    predictions = model.predict(test_data)
    

## Classification with Keras










