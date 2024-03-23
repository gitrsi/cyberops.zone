![Deep Neural Networks with PyTorch](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Neuronal_networks.jpg "Deep Neural Networks with PyTorch")

> :bulb: Notes on "Deep Neural Networks with PyTorch"



# Overview of Tensors
- In Pytorch, neural networks are composed of Pytorch tensors
- A Pytorch tensor is a data structure that is a generalization for numbers and dimensional arrays in Python
- For processing the neural network will apply a series of tensor operations on the input it receives
- easily convertable to numpy arrays and vice versa
- easy to integrate with "GPU"

Convert data to tensors
- each row in the db is a tensor
- tensor is a vector
- input the tensor into the neural network

Convert images to tensors
- usually represented as 2D/3D arrays
- each tensor is a matrix


## Tensors 1D
A 0-d tensor is just a number, 1-D tensor is an array of numbers


    import torch
    import numpy
    import pandas

    a = torch.tensor([7,4,3,2,6], dtype=torch.int32) # explicit data type
    b = torch.FloatTensor([1,2,3,4]) # explicit type
    a=a.type(torch.FloatTensor) # convert type

    a[0]=7 
    a[0] # -> tensor(7)
    a[0].item() # -> 7

    a.dtype # type of data
    a.type  # type of tensor
    a.size()    # 5
    a.ndimension()  # 1

    # convert 1D to 2D tensor
    a_col=a.view(5,1)
    # a_col=a.view(-1,1) # -1, if we don't know the number of elements

    # convert tensor to numpy array
    numpy_array = np.array([0,1,2,3,4])
    torch_tensor=torch.from_numpy(numpy_array)
    back_to_numpy=torch_tensor.numpy() # reference pointers are used

    # convert to pandas series
    pandas_series=pd.Series([0.1,2,0.3,10.1])
    pandas_to_torch=torch.from_numpy(pandas_series.values)

    # convert to list
    this_tensor=torch.tensor([1,2,3])
    torch_to_list=this.tensor.tolist()

    # slice tensor
    c=torch.tensor([0,1,2,3,4])
    d=c[0:2]
    d # -> tensor([0, 1])

    # assign values
    c[0:1]=torch.tensor([7,8])

    # vector addition and subtraction
    u=torch.tensor([1.0,0.0])
    v=torch.tensor([0.0,1.0])
    z=u+v
    z # -> tensor([1,1])

    # adding constant to tensor -> broadcast
    z=u+1

    # vector multiplication
    y=torch.tensor([1.0,2.0])
    z=2*y
    z # -> tensor([2,4])

    # product of two tensors
    u=torch.tensor([1,2])
    v=torch.tensor([3,2])
    z=u*v
    z # -> tensor([3,4])

    # Dot product # represents how similar the two vectors are
    u=torch.tensor([1,2])
    v=torch.tensor([3,1])
    result=torch.dot(u,v) # 1*3 + 2*1
    result # -> 5

    # apply functions to tensors
    a=torch.tensor([1,-1,1,-1])
    mean_a=a.mean() # (1 -1+1-1)/4
    max_a=b.max()
    #
    x=torch.tensor([0,np.pi/2,np.pi])
    y=torch.sin(x) # [sin(0), sin(pi/2), sin(pi)
    y # tensor([0,1,0])
    #
    a=torch(linspace(-2,2,steps=5)
    a # torch([-2,-1,0,1,2])

    # plotting mathematical functions
    import matplotlic.pyplot as plt
    x=torch.linspace(0,2*np.pi,100)
    y=torch.sin(x)
    %matplotlib inline
    plt.plot(x.numpy(),y.numpy())

## Tensors 2D
- dimension -> rank, axis
- 2D tensor is a container holding numerical values of the same type
- like rows and columns in a database
- essentially a matrix
- row corresponds to a sample
- column corresponds to a feature/attribute


    import torch
    a = [[11,12,13],[21,22,23],[31,32,33]] # list of rows
    A = torch.tensor(a) # matrix
    A.ndimension() # -> 2
    A[1][0] # -> 21

    # addition
    X=torch.tensor([[1,0],[0,1]])
    Y=torch.tensor([[2,1],[1,2]])
    Z=X+Y
    Z # -> tensor([[3,11],[1,3]])

    # multiplication
    Y=torch.tensor([[2,1],[1,2]])
    Z=2*Y
    Z # -> tensor([[4,2],[2,4]])

    # product of two tensors (Element-wise Product/Hadamard Product)
    X=torch.tensor([[1,0],[0,1]])
    Y=torch.tensor([[2,1],[1,2]])
    Z=X*Y
    Z # -> tensor([[2,0],[0,2]])

    # matrix multiplication
    A=torch.tensor([[0,1,1],[1,0,1]])
    B=torch.tensor([[1,1],[1,1],[-1,1]])
    C=torch.mm(A,B) # dot product of A rows with B columns
    C # -> tensor([[0,2],[0,2])
    

    # slicing
    A[0,0:2]
    A # -> tensor([11,12])

    
## Derivatives in PyTorch/Differentiation in PyTorch
- Behind the scenes, pytorch calculates derivatives by creating a backwards graph

### Derivatives
$$
y(x)=x^2
$$
  
$$
{dy(x) \over dx}=2x^1
$$

    import torch
    x=torch.tensor(2,requires_grad=True) # required for applying functions and derivatives to x
    y=x**2
    y.backward() # -> 2x
    x.grad # 4


$$
z(x)=x^2+2x+1
$$

$$
{dz(x) \over dx}=2x+2
$$

    import torch
    x=torch.tensor(2,requires_grad=True) # required for applying functions and derivatives to x
    y=x**2 + 2*x + 1
    y.backward() # -> 2x + 2
    x.grad # 6


Calculate the derivative with multiple values

    import torch 
    import matplotlib.pylab as plt

    x = torch.linspace(-10, 10, 10, requires_grad = True)
    Y = x ** 2
    y = torch.sum(x ** 2)

    y.backward()

    plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
    plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
    plt.xlabel('x')
    plt.legend()
    plt.show()


Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative

    import torch 
    import matplotlib.pylab as plt

    x = torch.linspace(-10, 10, 1000, requires_grad = True)
    Y = torch.relu(x)
    y = Y.sum()
    y.backward()
    plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
    plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
    plt.xlabel('x')
    plt.legend()
    plt.show()


### Partial derivatives
$$
f(u,v)=uv+u^2
$$

$$
{\partial f(u,v) \over \partial u}=v+2u
$$

$$
{\partial f(u,v) \over \partial v}=u
$$

    u=torch.tensor(1,requires_grad=True)
    v=torch.tensor(2,requires_grad=True)
    f=u*v+u**2
    f.backward() # -> v + 2u
    u.grad # tensor(4)
    
    
## Simple Dataset   

    import torch
    from torch.utils.data import Dataset
    torch.manual_seed(1)

    # Define class for dataset
    class toy_set(Dataset):
        
        # Constructor with defult values 
        def __init__(self, length = 100, transform = None):
            self.len = length
            self.x = 2 * torch.ones(length, 2)
            self.y = torch.ones(length, 1)
            self.transform = transform
         
        # Getter
        def __getitem__(self, index):
            sample = self.x[index], self.y[index]
            if self.transform:
                sample = self.transform(sample)     
            return sample
        
        # Get Length
        def __len__(self):
            return self.len

    # Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.
    our_dataset = toy_set()
    print("Our toy_set object: ", our_dataset)
    print("Value on index 0 of our toy_set object: ", our_dataset[0])
    print("Our toy_set length: ", len(our_dataset))

    # Use loop to print out first 3 elements in dataset
    for i in range(3):
        x, y=our_dataset[i]
        print("index: ", i, '; x:', x, '; y:', y)

    # loop on the dataset object
    for x,y in our_dataset:
        print(' x:', x, 'y:', y)

## Transforms

    # Create tranform class add_mult
    class add_mult(object):
        
        # Constructor
        def __init__(self, addx = 1, muly = 2):
            self.addx = addx
            self.muly = muly
        
        # Executor
        def __call__(self, sample):
            x = sample[0]
            y = sample[1]
            x = x + self.addx
            y = y * self.muly
            sample = x, y
            return sample

    # Create an add_mult transform object, and an toy_set object
    a_m = add_mult()
    data_set = toy_set()

    # Use loop to print out first 10 elements in dataset
    for i in range(10):
        x, y = data_set[i]
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
        x_, y_ = a_m(data_set[i])
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

    # Create a new data_set object with add_mult object as transform
    cust_data_set = toy_set(transform = a_m)

    # Use loop to print out first 10 elements in dataset
    for i in range(10):
        x, y = data_set[i]
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
        x_, y_ = cust_data_set[i]
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

## Compose transforms

    # Run the command below when you do not have torchvision installed
    # !mamba install -y torchvision
    from torchvision import transforms

    # Create tranform class mult
    class mult(object):
        
        # Constructor
        def __init__(self, mult = 100):
            self.mult = mult
            
        # Executor
        def __call__(self, sample):
            x = sample[0]
            y = sample[1]
            x = x * self.mult
            y = y * self.mult
            sample = x, y
            return sample

    # Combine the add_mult() and mult()
    data_transform = transforms.Compose([add_mult(), mult()])
    print("The combination of transforms (Compose): ", data_transform)

    data_transform(data_set[0])

    x,y=data_set[0]
    x_,y_=data_transform(data_set[0])
    print( 'Original x: ', x, 'Original y: ', y)

    print( 'Transformed x_:', x_, 'Transformed y_:', y_)

    # Create a new toy_set object with compose object as transform
    compose_data_set = toy_set(transform = data_transform)

    # Use loop to print out first 3 elements in dataset

    for i in range(3):
        x, y = data_set[i]
        print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
        x_, y_ = cust_data_set[i]
        print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
        x_co, y_co = compose_data_set[i]
        print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)

## Data set for images
- Torch vision datasets
- Torch vision transforms


# Linear regression
- Linear regression involves learning the linear relationship between x and y values in a dataset
- Simple linear regression refers to cases where x is only one dimension
- Datasets for linear regression can include predicting housing prices, stock prices, or fuel economy of cars
- Noise is added to the points on the line to account for errors, and it is assumed to be Gaussian
- The goal of linear regression is to find the best line that represents the points, which can be determined by minimizing a cost function. The cost function used in linear regression is the mean squared error.

## Prediction
- method to predict a continuous value
- predictor: independent variable x
- target: dependent variable y
- b: bias
- w: slope or weight

$$
y=b+wx
$$

Steps
- training step
- prediction step/forward step


    w=torch.tensor(2.0, requires_grad=True)
    b=torch.tensor(-1.0, requires_grad=True)

    def forward(x):
        y=w*x+b
        return y

    x=torch.tensor([1.0])
    yhat=forward(x)
    yhat # tensor([1.0])

## Training
Dataset as mathematical notation
- ordered pairs
- each pair is a data point within a cartesian plain
- corresponding x,y coordinates
- datasets organized as tensors

Loss
- Loss is a measure of how well a model's predictions match the actual values
- The goal is to minimize the loss by finding the best values for the model's parameters
- The loss function is shaped like a concave bowl in the parameter space
- Different values of the parameters result in different lines and different loss values


## Gradient descent
- Gradient descent is a method used to find the minimum of a function
- It can be applied to functions with multiple dimensions, but this video focuses on the example of one dimension
- Gradient descent involves iteratively updating a parameter by adding a value proportional to the negative of the derivative
- The learning rate, represented by the parameter eta, determines how much the parameter should be updated
- Choosing the right learning rate is important. If it's too big, we may miss the minimum, and if it's too small, it may take a long time to reach the minimum.
- There are several ways to stop the process of gradient descent, such as running it for a set number of iterations or stopping when the loss starts increasing.

## Cost
- The cost function is used to determine the value of parameters that minimize the loss value for multiple data points
- The cost function is a function of the slope and the bias, which control the relationship between input and output
- The slope controls the relationship between x and y
- The bias controls the horizontal offset
- Gradient descent is a method used to update the parameter values based on the derivative of the cost function
- The batch size refers to the number of samples used to calculate the loss and update the parameters
- All the samples in the training set are called a batch, and when we use all the samples, it is called batch gradient descent


# PyTorch Slope
- performing gradient descent in PyTorch
- create a PyTorch tensor and set the option requires_grad to true
- mapping X values to a line with a slope of -3 and adding random noise to the data points
- define the forward function and the criterion function or cost function
- PyTorch calculates the derivative with respect to the parameter w using the backward method
- average loss or cost decreases for each iteration




























