![Machine Learning with Python](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Deep_Learning_and_Neural_Networks.jpg "Machine Learning with Python")

> :bulb: Notes on "Deep Learning & Neural Networks with Keras"


# Deep learning
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

$$
z_1 = x_1w_1+b_1 //
a_1 = f(z_1) //
z_2 = w_2a_1+b_2 //
a_2 = f(z_2) //
$$

Activation functions
- map the weighted sum to a nonlinear space (sigmoid function)
- + unendlich: 1, - unendlich: 0
- activation functions decide whether a neuron should be activated or not (received info is relevant/ignored by neuron)


Neural network without activation function -> linear regression model
Activation function performs non-linear transformation to the input enabling the neural network of learning and performing more complex tasks, such as image classifications and language translations.
