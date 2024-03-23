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
- A 0-d tensor is just a number, 1-D tensor is an array of numbers

Usage:

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

Usage:

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
- loss.backward() calculates or accumulate gradients of loss
- average loss or cost decreases for each iteration

Create the Model and Cost Function (Total Loss)

    # Create forward function for prediction
    def forward(x):
        return w * x

    # Create the MSE function for evaluate the result.
    def criterion(yhat, y):
        return torch.mean((yhat - y) ** 2)

    # Create Learning Rate and an empty list to record the loss for each iteration
    lr = 0.1
    LOSS = []

    w = torch.tensor(-10.0, requires_grad = True)


Train the Model

    # Define a function for train the model
    def train_model(iter):
        for epoch in range (iter):
            
            # make the prediction as we learned in the last lab
            Yhat = forward(X)
            
            # calculate the iteration
            loss = criterion(Yhat,Y)
            
            # plot the diagram for us to have a better idea
            gradient_plot(Yhat, w, loss.item(), epoch)
            
            # store the loss into list
            LOSS.append(loss.item())
            
            # backward pass: compute gradient of the loss with respect to all the learnable parameters
            loss.backward()
            
            # updata parameters
            w.data = w.data - lr * w.grad.data
            
            # zero the gradients before running the backward pass
            w.grad.data.zero_()

    # Give 4 iterations for training the model here.
    train_model(4)

    # Plot the loss for each iteration
    plt.plot(LOSS)
    plt.tight_layout()
    plt.xlabel("Epoch/Iterations")
    plt.ylabel("Cost")

## PyTorch Linear Regression Training Slope and Bias
### Slope and bias
Relationship between the slope and bias in linear regression
- Slope: The slope represents the rate of change of the output variable with respect to the input variable. It determines the steepness or the angle of the line that represents the linear relationship between the variables. A positive slope indicates a positive relationship, where an increase in the input variable leads to an increase in the output variable. A negative slope indicates a negative relationship, where an increase in the input variable leads to a decrease in the output variable
- Bias: The bias term represents the y-intercept of the line in the linear regression equation. It determines the starting point or the offset of the line. The bias allows the line to fit the data points even if they do not pass through the origin (0,0). It accounts for the constant term in the linear relationship between the variables

Together, the slope and bias determine the position and orientation of the line in the linear regression model, allowing it to best fit the given data points and capture the relationship between the variables

### Cost
The cost surface, also known as the loss surface or the error surface, is a visual representation of the cost function in linear regression. It helps us understand the relationship between the slope and bias by providing insights into how different values of these parameters affect the overall cost or error of the model. Here's how the cost surface helps us:

- Visualization: The cost surface is typically plotted as a three-dimensional surface, where the slope is represented on one axis, the bias on another axis, and the cost on the vertical axis. This visualization allows us to see the shape and contours of the cost function in relation to the slope and bias.

- Contour lines: Contour lines are lines on the cost surface that connect points with the same cost value. By examining these contour lines, we can observe how the cost changes as we vary the slope and bias. The contour lines can reveal patterns, such as areas of high or low cost, and the direction of steepest descent.

- Optimal parameter values: The cost surface helps us identify the optimal values for the slope and bias that minimize the cost function. By locating the lowest point on the cost surface, we can determine the combination of slope and bias that results in the best fit for the given data.

- Gradient descent: The cost surface provides insights into the direction of steepest descent, which is the direction that leads to the minimum cost. The gradient, which is the vector of partial derivatives of the cost function with respect to the slope and bias, points in the direction of steepest descent. By following the gradient, we can iteratively update the slope and bias values to minimize the cost and improve the model's performance.

In summary, the cost surface helps us visualize and understand how the slope and bias impact the cost function in linear regression. It guides us in finding the optimal parameter values and enables us to iteratively improve the model through techniques like gradient descent.

### Contour lines
The cost function can be visualized as a surface, where one axis represents the slope and the other axis represents the bias, while the height represents the cost.

Contour lines are a useful tool for understanding the cost surface. They are lines on the surface where the points have equal values of the cost function. By cutting the surface at a specific value, we can interpret it as a plane, and the points where the surface intersects with the plane correspond to the contour lines.

Here's how contour lines correspond to specific values of the cost function:

- Cutting the surface at a specific value: Imagine cutting the cost surface at a certain height, let's say 200. This cut can be interpreted as a plane. The contour line corresponds to the points on the surface that have a cost value of 200.

- Different contour lines: By cutting the surface at different values, we get different contour lines. For example, cutting the surface at a value of 400 will result in a different contour line compared to cutting it at 600.

- Changing parameter values: The contour lines show how different parameter values affect the cost. By overlaying points on the graph, we can see how the cost changes for different parameter values. Each iteration of the optimization algorithm aims to find parameter values that minimize the cost.

In summary, contour lines on the cost surface represent points with equal values of the cost function. By cutting the surface at different heights, we can visualize the contour lines and understand how different parameter values affect the cost. The goal of optimization algorithms like gradient descent is to find the parameter values that minimize the cost function, bringing the contour lines closer to the minimum.


### Gradient descent
Gradient descent is an optimization algorithm that helps in finding the minimum of the cost function in linear regression. Here's how gradient descent works and how it helps:

- Derivative and Gradient: The derivative of a function represents its rate of change at a particular point. In the context of linear regression, the cost function is a multivariate function of the slope and bias. The gradient is a vector that consists of the partial derivatives of the cost function with respect to the slope and bias. The gradient points in the direction of the steepest ascent.

- Iterative Update: Gradient descent iteratively updates the values of the slope and bias by taking steps proportional to the negative gradient. The negative gradient points in the direction of the steepest descent, which helps in moving towards the minimum of the cost function.

- Learning Rate: The learning rate is a hyperparameter that determines the size of the steps taken during each iteration of gradient descent. It controls the speed at which the algorithm converges to the minimum. A larger learning rate may result in faster convergence but can overshoot the minimum, while a smaller learning rate may take longer to converge but can provide more accurate results.

- Local and Global Minima: The cost function in linear regression can have multiple local minima, where the cost is relatively low but not the absolute minimum. Gradient descent helps in finding a local minimum by iteratively updating the slope and bias values. However, it does not guarantee finding the global minimum, which is the absolute lowest point of the cost function.

- Convergence: Gradient descent continues to update the slope and bias values until a stopping criterion is met. This criterion can be a maximum number of iterations, a threshold for the change in the cost function, or other conditions. Once the algorithm converges, the slope and bias values correspond to the minimum of the cost function, providing the best-fit line for the given data.

In summary, gradient descent helps in finding the minimum of the cost function in linear regression by iteratively updating the slope and bias values in the direction of the steepest descent. It allows the algorithm to converge to a local minimum, providing the optimal parameter values for the linear regression model.

### Learning rate
The learning rate is a hyperparameter that needs to be carefully chosen, as it directly affects the convergence of the algorithm.

Here's how the learning rate affects the convergence of the algorithm:

- Large learning rate: If the learning rate is set too high, the algorithm may overshoot the minimum of the cost function. This can lead to oscillations or even divergence, where the algorithm fails to converge. The updates to the parameters are too large, causing the algorithm to miss the optimal solution.

- Small learning rate: On the other hand, if the learning rate is set too low, the algorithm may converge very slowly. The updates to the parameters are too small, and it takes a large number of iterations to reach the minimum of the cost function. This can result in longer training times and inefficiency.

- Appropriate learning rate: The ideal learning rate is one that allows the algorithm to converge efficiently. It should be neither too large nor too small. With an appropriate learning rate, the algorithm can make steady progress towards the minimum of the cost function, converging in a reasonable number of iterations.

It's important to note that the choice of learning rate depends on the specific problem and dataset. There is no one-size-fits-all learning rate, and it often requires experimentation and tuning to find the optimal value. Techniques like learning rate schedules or adaptive learning rate methods can also be used to dynamically adjust the learning rate during training.

In summary, the learning rate controls the step size of parameter updates in gradient descent. Choosing an appropriate learning rate is crucial for the algorithm to converge efficiently and find the optimal solution.

## Stochastic Gradient Descent
Stochastic Gradient Descent (SGD) and Batch Gradient Descent (BGD) differ in terms of sample selection and parameter update

Sample Selection:
- SGD: In SGD, one sample or a small random batch of samples is selected at each iteration. This random selection introduces randomness into the optimization process, as different samples are used for parameter updates in each iteration.
- BGD: In BGD, the entire training dataset is used for each iteration. All samples are considered simultaneously to compute the gradients and update the parameters. BGD does not introduce randomness in sample selection.

Parameter Updates:
- SGD: In SGD, the parameters are updated after each sample or batch. The gradients are computed based on the difference between the predicted output and the actual output for that specific sample or batch. The parameters are then updated using these gradients, aiming to minimize the loss for that particular sample or batch.
- BGD: In BGD, the parameters are updated after processing the entire training dataset. The gradients are computed based on the average difference between the predicted output and the actual output over the entire dataset. The parameters are updated using these average gradients, aiming to minimize the overall loss for the entire dataset.

Key Differences:
- SGD is computationally more efficient than BGD because it processes fewer samples or batches in each iteration.
- SGD converges faster initially but may oscillate around the optimal solution due to the randomness introduced by sample selection.
- BGD provides a smoother convergence as it considers the entire dataset, but it can be computationally expensive for large datasets.
- SGD is more prone to noise and may struggle to find the global minimum, while BGD is less affected by noise but may take longer to converge.

In practice, a compromise between SGD and BGD is often used, known as Mini-Batch Gradient Descent. It processes a small batch of samples in each iteration, striking a balance between computational efficiency and convergence stability.

Advantages of Stochastic Gradient Descent (SGD):
- Efficiency: SGD is computationally more efficient than BGD because it updates the parameters for each sample or small batch of samples, rather than the entire dataset. This makes it faster, especially for large datasets.
- Convergence: SGD can converge faster than BGD, especially when the loss landscape is noisy or has many local minima. The frequent updates to the parameters allow SGD to escape shallow local minima and find better solutions.
- Generalization: SGD can generalize better to unseen data because it updates the parameters based on individual samples or small batches. This randomness helps prevent overfitting and improves the model's ability to generalize to new data.

Disadvantages of Stochastic Gradient Descent (SGD):
- Noisy updates: The frequent updates in SGD introduce noise into the optimization process, causing the loss to fluctuate more compared to BGD. This can make it harder to determine convergence and select the optimal learning rate.
- Slower convergence: Although SGD can converge faster in terms of iterations, it may require more iterations to reach convergence compared to BGD. This is because the updates based on individual samples or small batches can be noisier and less accurate.
- Learning rate selection: SGD is more sensitive to the learning rate hyperparameter. Choosing an appropriate learning rate is crucial for SGD to converge effectively. If the learning rate is too high, SGD may overshoot the optimal solution, while a too low learning rate can slow down convergence.

In summary, SGD offers efficiency, faster convergence in terms of iterations, and better generalization. However, it introduces noise, may require more iterations to converge, and requires careful selection of the learning rate. The choice between SGD and BGD depends on the specific problem, dataset size, and computational resources available.

### Epoch
In the context of Stochastic Gradient Descent (SGD), an epoch refers to a complete pass through the entire training dataset. During each epoch, the model updates its parameters based on the gradients computed from individual samples or small batches of samples.

Here's how epochs and SGD are related:
- Dataset: Consider a training dataset with a certain number of samples.
- Iterations: In each epoch, the model goes through multiple iterations. Each iteration involves processing one or a few samples (depending on the batch size) and updating the model's parameters.
- Gradient Calculation: For each sample or batch, the gradients are calculated based on the difference between the predicted output and the actual output.
- Parameter Update: The model's parameters (weights and biases) are updated using the gradients, aiming to minimize the loss function.
- Epoch Completion: After going through all the samples or batches in the dataset, one epoch is completed.
- Multiple Epochs: To improve the model's performance, SGD typically involves running multiple epochs. This allows the model to learn from the entire dataset multiple times, refining the parameter updates and reducing the overall loss.

By performing multiple epochs, the model has the opportunity to see different samples and adjust its parameters accordingly. This iterative process helps the model converge towards an optimal set of parameters that minimize the loss function and improve the model's performance.

It's important to note that the number of epochs is a hyperparameter that needs to be carefully chosen. Too few epochs may result in underfitting, where the model fails to capture the underlying patterns in the data. On the other hand, too many epochs may lead to overfitting, where the model becomes too specialized to the training data and performs poorly on unseen data.

### Cost
Stochastic Gradient Descent (SGD) approximates the cost function by minimizing the loss for each sample in an iterative manner. Here's how it works:
- At each iteration of SGD, a single sample or a small random batch of samples is selected from the training dataset.
- The parameters of the model are initialized, and the predicted output is computed based on the current parameter values.
- The loss or error between the predicted output and the actual output for the selected sample(s) is calculated.
- The gradients of the parameters with respect to the loss are computed. These gradients indicate the direction and magnitude of the steepest ascent or descent in the parameter space.
- The parameters are updated by taking a small step in the opposite direction of the gradients, aiming to minimize the loss for the selected sample(s).
- The process is repeated for the next sample(s) in the dataset, and the parameters are updated accordingly.
- This iterative process continues for multiple epochs, where each epoch represents a complete pass through the entire training dataset.
- The final parameter values obtained after multiple iterations are expected to minimize the average or total loss over the entire dataset.
- Value of the approximate cost will fluctuate rapidly with each iteration

By minimizing the loss for each sample, SGD approximates the cost function by iteratively updating the parameters based on the gradients calculated for each sample. This approach allows SGD to make frequent updates to the parameters, which can be beneficial in certain scenarios. However, it also introduces randomness and fluctuations in the optimization process, as different samples are used for parameter updates in each iteration.

### Learning rate
The learning rate is a hyperparameter in Stochastic Gradient Descent (SGD) that determines the step size at each iteration when updating the model parameters. It plays a crucial role in the convergence of SGD. Here's how the learning rate affects convergence:
- Learning Rate Too High: If the learning rate is set too high, SGD may overshoot the optimal solution and fail to converge. In this case, the updates to the parameters are too large, causing them to oscillate or diverge. The loss may increase instead of decreasing, and the model may fail to find the global or local minima.
- Learning Rate Too Low: On the other hand, if the learning rate is set too low, SGD may converge very slowly. The updates to the parameters are too small, and it takes a large number of iterations to reach the optimal solution. This can result in slow training and longer computation time.
- Appropriate Learning Rate: The learning rate needs to be carefully chosen to ensure convergence. An appropriate learning rate allows SGD to make steady progress towards the optimal solution without overshooting or converging too slowly. It strikes a balance between fast convergence and stability.
- Learning Rate Scheduling: In practice, it is common to use learning rate scheduling techniques to adaptively adjust the learning rate during training. For example, learning rate decay or learning rate annealing can be used to gradually reduce the learning rate over time. This can help SGD converge more effectively by taking larger steps initially and smaller steps as it gets closer to the optimal solution.

In summary, the learning rate is a critical hyperparameter in SGD that affects the convergence of the algorithm. It needs to be carefully chosen to ensure stable and efficient convergence. Setting the learning rate too high can lead to overshooting, while setting it too low can result in slow convergence. Adaptive learning rate scheduling techniques can be used to improve convergence performance.


## Mini-Batch Gradient Descent
Mini-Batch Gradient Descent (MBGD) is a variation of Gradient Descent that allows you to process larger datasets that don't fit into memory. 

### Advantages
Advantages of using Mini-Batch Gradient Descent for processing larger datasets:
- Memory Efficiency: Mini-Batch Gradient Descent allows you to process larger datasets that cannot fit into memory. It achieves this by splitting the dataset into smaller samples or batches.
- Faster Computation: By using smaller batches, Mini-Batch Gradient Descent can perform computations more quickly compared to using the entire dataset. This is because it processes a subset of the data in each iteration, reducing the overall computational load.
- Better Generalization: Mini-Batch Gradient Descent can lead to better generalization of the model. It introduces some randomness in the training process by using different batches in each iteration, which helps the model avoid getting stuck in local minima and find better solutions.
- Noise Reduction: The mini-batches in Mini-Batch Gradient Descent provide a form of regularization by introducing noise into the training process. This can help prevent overfitting and improve the model's ability to generalize to unseen data.

Overall, Mini-Batch Gradient Descent offers a balance between the efficiency of Stochastic Gradient Descent (using one sample at a time) and the stability of Batch Gradient Descent (using the entire dataset). It is a popular choice for training deep neural networks on large datasets.

### Comparison between Mini-Batch Gradient Descent, Stochastic Gradient Descent, and Batch Gradient Descent
Mini-Batch Gradient Descent, Stochastic Gradient Descent, and Batch Gradient Descent are three variations of gradient descent optimization algorithms. Here's how they differ:
- Batch Gradient Descent (BGD): In BGD, the entire training dataset is used to compute the gradient of the cost function in each iteration. It calculates the average gradient over all the training examples and updates the model parameters accordingly. BGD is computationally expensive, especially for large datasets, as it requires processing the entire dataset in each iteration.
- Stochastic Gradient Descent (SGD): In SGD, only one training example is used to compute the gradient and update the model parameters in each iteration. It randomly selects a single sample from the dataset and performs the update. SGD is computationally efficient but can be noisy and may take longer to converge due to the high variance in the gradient estimates.
- Mini-Batch Gradient Descent (MBGD): MBGD is a compromise between BGD and SGD. It processes a small batch of training examples (typically between 10 to 1000) to compute the gradient and update the model parameters. MBGD strikes a balance between the computational efficiency of SGD and the stability of BGD. It reduces the noise compared to SGD and can converge faster than BGD.

Here's a summary of the differences:
- BGD uses the entire dataset in each iteration, while SGD uses only one sample, and MBGD uses a small batch of samples.
- BGD is computationally expensive, SGD is computationally efficient, and MBGD provides a trade-off between the two.
- BGD provides a more stable gradient estimate, while SGD and MBGD introduce some randomness.
- SGD and MBGD can handle larger datasets that do not fit into memory, while BGD may require memory limitations.

Each variant has its advantages and disadvantages, and the choice depends on factors such as dataset size, computational resources, and the trade-off between accuracy and efficiency.

### Mini-Batch Gradient Descent vs. Stochastic Gradient Descent
Mini-Batch Gradient Descent (MBGD) offers several advantages over Stochastic Gradient Descent (SGD):
- Computational Efficiency: MBGD strikes a balance between the efficiency of SGD and the stability of Batch Gradient Descent (BGD). It processes a small batch of training examples at a time, which allows for parallelization and efficient vectorized operations. This makes it faster than BGD, which processes the entire dataset, and more stable than SGD, which updates parameters after each individual sample.
- Convergence Speed: MBGD often converges faster than SGD. By using a small batch of samples, MBGD provides a more accurate estimate of the true gradient compared to SGD's noisy estimates. This allows for more consistent updates to the model parameters, leading to faster convergence towards the optimal solution.
- Generalization: MBGD can generalize better than SGD. Since MBGD uses a mini-batch of samples, it provides a more representative estimate of the overall dataset compared to SGD's single-sample updates. This can lead to better generalization and improved performance on unseen data.
- Stability: MBGD reduces the variance in parameter updates compared to SGD. SGD's updates can be highly erratic due to the high variance in gradient estimates from individual samples. In contrast, MBGD averages the gradients over a mini-batch, resulting in smoother updates and more stable convergence.
- Memory Efficiency: MBGD allows for processing larger datasets that may not fit into memory. By splitting the dataset into smaller batches, MBGD can process them sequentially, reducing memory requirements compared to BGD, which needs to load the entire dataset at once.

Overall, MBGD combines the benefits of both BGD and SGD, providing a computationally efficient and stable optimization algorithm that converges faster than SGD while still being able to handle larger datasets.

### Batch size
The impact of different batch sizes on the convergence rate of Mini-Batch Gradient Descent (MBGD) can be summarized as follows:
- Smaller Batch Size: When using a smaller batch size, such as 1 or 2, the convergence rate of MBGD tends to be faster. This is because each iteration updates the model parameters based on a smaller number of samples, leading to more frequent updates. However, the updates can be noisy and less accurate due to the high variance in gradient estimates from individual samples.
- Larger Batch Size: With a larger batch size, such as 10 or 100, the convergence rate of MBGD tends to be slower compared to smaller batch sizes. This is because each iteration updates the model parameters based on a larger number of samples, resulting in less frequent updates. However, the updates are more accurate as they are based on a more representative estimate of the overall dataset.
- Trade-off: The choice of batch size involves a trade-off between convergence speed and accuracy. Smaller batch sizes converge faster but may have more erratic updates, while larger batch sizes converge slower but provide more stable updates. The optimal batch size depends on the specific problem, dataset size, and computational resources available.
- Convergence Rate Visualization: The video lecture explains that different batch sizes change how long it takes for the cost (or average loss) to stop decreasing, which is known as the convergence rate. The lecture provides a plot showing the cost with different batch sizes, demonstrating the impact on convergence.

It's important to note that the convergence rate is not the only factor to consider when choosing a batch size. Other factors, such as memory requirements and computational efficiency, should also be taken into account.

## Optimization in PyTorch
### Optimizer
The optimizer plays a crucial role in the training process of deep neural networks. Its purpose is to update the model's parameters in order to minimize the loss function and improve the model's performance. Here's a breakdown of the purpose and role of the optimizer:
- Minimizing Loss: The primary goal of the optimizer is to minimize the loss function. The loss function measures the discrepancy between the predicted output of the model and the actual target output. By iteratively adjusting the model's parameters, the optimizer aims to find the optimal set of parameter values that minimize the loss.
- Gradient Descent: Most optimizers, including the popular Stochastic Gradient Descent (SGD), utilize the concept of gradient descent. They calculate the gradients of the loss function with respect to the model's parameters. These gradients indicate the direction and magnitude of the steepest descent in the loss landscape. The optimizer then updates the parameters in the opposite direction of the gradients to move towards the minimum of the loss function.
- Learning Rate: The learning rate is a crucial parameter of the optimizer. It determines the step size or the magnitude of the parameter updates. A higher learning rate can lead to faster convergence but may risk overshooting the optimal solution. On the other hand, a lower learning rate may result in slower convergence but can provide more precise parameter updates. Finding an appropriate learning rate is essential for effective optimization.
- Optimization Algorithms: Optimizers often employ various optimization algorithms to enhance the training process. These algorithms can include momentum, weight decay, Nesterov momentum, and more. They introduce additional techniques to improve convergence speed, handle noise in the gradients, regularize the model, and overcome local minima.
- Customization: Optimizers in PyTorch offer flexibility and customization options. You can adjust the optimizer's parameters, such as momentum, weight decay, and learning rate scheduling, to fine-tune the optimization process according to your specific requirements and the characteristics of your dataset.

In summary, the optimizer is responsible for updating the model's parameters based on the gradients of the loss function. It aims to minimize the loss and improve the model's performance by iteratively adjusting the parameters using techniques like gradient descent and optimization algorithms. The choice of optimizer and its parameters can significantly impact the training process and the final performance of the model.

### PyTorch Optimizer
PyTorch Optimizer is a standard way to perform different variations of gradient descent in PyTorch. The steps involved in using the PyTorch Optimizer are as follows:
- Create a dataset object to handle the data.
- Create a custom module or class as a subclass of the nn.Module.
- Create a criterion or cost function.
- Create a trainloader object to load the samples.
- Create a model.
- Import the optimizer package from torch and construct an optimizer object. In this case, SGD (stochastic gradient descent) is used.
- Use the parameters from the model object as input to the optimizer constructor.
- Set the gradient to 0.
- Make a prediction using the model.
- Calculate the loss or cost.
- Differentiate the loss with respect to the parameters.
- Apply the optimizer's step method to update the parameters.

### Gradient
Setting the gradient to 0 before calculating the loss in PyTorch is an important step in the training process. Here's why it is done:
- Gradient Accumulation: During the training process, the gradients of the model parameters are accumulated as you perform backpropagation and calculate the gradients for each batch of data. If you don't reset the gradients to 0 before each batch, the gradients will keep accumulating from the previous batches, leading to incorrect gradient updates and potentially slower convergence.
- Avoiding Gradient Interference: By setting the gradient to 0 before calculating the loss, you ensure that the gradients calculated for the current batch are independent of any previous batches. This helps in isolating the impact of each batch on the parameter updates and prevents interference between different batches.
- Memory Efficiency: Resetting the gradients to 0 also helps in managing memory efficiently. Since the gradients are stored in memory during backpropagation, resetting them to 0 after each batch ensures that unnecessary memory is freed up for the subsequent batches.

In summary, setting the gradient to 0 before calculating the loss in PyTorch ensures proper accumulation of gradients, avoids interference between batches, and helps in memory management during training.

### Trainloader object
The trainloader object in PyTorch is used to load the training data in batches during the training process. It serves several purposes:
- Batch Processing: In deep learning, it is common to train models on large datasets. Loading the entire dataset at once may not be feasible due to memory limitations. The trainloader allows you to load the data in smaller batches, enabling efficient processing and utilization of system resources.
- Randomization and Shuffling: The trainloader provides the option to shuffle the data before each epoch. Shuffling the data helps in reducing any bias that may arise due to the order of the samples. It ensures that the model sees a diverse range of samples in each batch, leading to better generalization.
- Parallel Data Loading: The trainloader supports parallel data loading, which means it can load multiple batches simultaneously using multiple workers. This parallelization can significantly speed up the data loading process, especially when working with large datasets.
- Iteration and Accessibility: The trainloader acts as an iterator, allowing you to easily iterate over the batches of data in a for loop. It provides convenient access to the input features and corresponding labels for each batch, making it easier to feed the data into the model during training.

Example

    # Create a trainloader object
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the batches in the trainloader
    for inputs, labels in trainloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: make predictions
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradients
        loss.backward()

        # Update the parameters
        optimizer.step()

In this example, the trainloader is used to load batches of inputs and labels from the train_dataset. The model makes predictions on the inputs, calculates the loss, performs backpropagation to compute gradients, and updates the parameters using the optimizer. This process is repeated for each batch in the trainloader, allowing the model to learn from the entire training dataset over multiple epochs.

Overall, the trainloader plays a crucial role in efficient and effective training by handling batch processing, shuffling, parallel data loading, and providing easy accessibility to the training data.

### Other optimizers
In PyTorch, besides the learning rate, there are several other optimizer options that can be used to customize the optimization process. Some commonly used optimizer options in PyTorch include:
- Momentum: Momentum is a parameter that helps accelerate the optimization process by accumulating the past gradients. It adds a fraction of the previous update to the current update, allowing the optimizer to move faster in the relevant direction. A higher momentum value can help overcome local minima and plateaus.
- Weight Decay: Weight decay, also known as L2 regularization, is a technique used to prevent overfitting by adding a penalty term to the loss function. It encourages the model to have smaller weights, reducing the complexity of the model. The weight decay parameter controls the strength of the regularization.
- Dampening: Dampening is an option specific to the SGD optimizer. It reduces the oscillations that can occur when using momentum. By setting a dampening value less than 1, the update is dampened, leading to smoother convergence.
- Nesterov Momentum: Nesterov momentum is an extension of the standard momentum technique. It adjusts the momentum update by taking into account the gradient at the lookahead position. This helps in better convergence and can improve the optimization process.
- Learning Rate Scheduling: Learning rate scheduling involves changing the learning rate during training. It can be useful in scenarios where a fixed learning rate may not be optimal. PyTorch provides various scheduling options, such as step-based, exponential, and cosine annealing, to adjust the learning rate over time.

## Training, Validation and Test Split

### Purpose
The purpose of splitting the data into training and validation sets is to evaluate the performance of a machine learning model.

The training set is used to train the model by adjusting its parameters based on the input data. The model learns from the training set to make predictions or classify new data points.

The validation set, on the other hand, is used to assess the model's performance and generalization ability. It acts as a proxy for unseen data, allowing us to evaluate how well the model will perform on new, unseen data. By evaluating the model on the validation set, we can make adjustments to improve its performance, such as tuning hyperparameters or modifying the model architecture.

Splitting the data into training and validation sets helps prevent overfitting, which occurs when a model performs well on the training data but fails to generalize to new data. By evaluating the model on a separate validation set, we can get a more accurate estimate of its performance and make informed decisions about model improvements.

### Steps in PyTorch
The steps involved in training and validating a model in PyTorch are as follows:
- Split the data: Split the available data into training and validation sets. The training set is used to train the model, while the validation set is used to evaluate its performance.
- Create a custom module: Define a custom module that represents the model architecture. This module will contain the necessary layers and parameters for the model.
- Define a criterion and optimizer: Specify a loss function (criterion) that measures the error between the model's predictions and the actual values. Choose an optimizer that will update the model's parameters based on the calculated loss.
- Train the model: Iterate over the training data in multiple epochs (passes through the entire dataset). For each epoch, perform the following steps:
    - Make predictions: Use the model to make predictions on the training data.
    - Calculate loss: Compare the model's predictions with the actual values and calculate the loss.
    - Backpropagation: Propagate the loss backward through the model to compute gradients.
    - Update parameters: Use the optimizer to update the model's parameters based on the gradients.
- Validate the model: After training, evaluate the model's performance on the validation set. Use the trained model to make predictions on the validation data and calculate the validation loss.
- Select the best model: Compare the validation loss of different models trained with different hyperparameters or learning rates. Choose the model with the lowest validation loss as the best model.
- Save the model: Save the best model for future use or deployment.

### Overfitting
Overfitting is a phenomenon that occurs when a machine learning model fits the training data too closely, to the point where it fails to generalize well to new, unseen data. In other words, the model becomes too specific to the training data and loses its ability to make accurate predictions on new data points.

Overfitting occurs due to the complexity of the model and the limited amount of training data. When a complex model is trained on a small dataset, it has a higher chance of memorizing the training examples instead of learning the underlying patterns and relationships. As a result, the model becomes overly sensitive to the noise or outliers present in the training data, which may not be representative of the overall population.

This over-reliance on the training data leads to poor performance when the model encounters new data that it hasn't seen before. The model may struggle to generalize and make accurate predictions because it has essentially "overfit" itself to the idiosyncrasies of the training data.

To mitigate overfitting, it is important to strike a balance between model complexity and the amount of training data available. Techniques such as regularization, cross-validation, and early stopping can be employed to prevent overfitting and improve the generalization ability of the model.

To prevent overfitting in machine learning models, several techniques and strategies can be employed. Here are some commonly used ones:
- Cross-validation: Instead of relying solely on a single train-test split, cross-validation involves dividing the data into multiple subsets or folds. The model is trained and evaluated on different combinations of these folds, allowing for a more robust assessment of its performance.
- Regularization: Regularization techniques add a penalty term to the loss function during training to discourage complex models that may overfit. Two popular regularization techniques are L1 regularization (Lasso) and L2 regularization (Ridge), which control the magnitude of the model's coefficients.
- Early stopping: This technique involves monitoring the model's performance on a validation set during training. Training is stopped when the model's performance on the validation set starts to deteriorate, preventing it from overfitting the training data.
- Data augmentation: By artificially increasing the size of the training data through techniques like rotation, scaling, flipping, or adding noise, data augmentation helps expose the model to a wider range of variations and reduces overfitting.
- Feature selection: Selecting relevant features and removing irrelevant or redundant ones can help reduce overfitting. Feature selection techniques like forward selection, backward elimination, or regularization-based feature selection can be employed.
- Ensemble methods: Ensemble methods combine multiple models to make predictions. Techniques like bagging (e.g., Random Forests) and boosting (e.g., Gradient Boosting) can help reduce overfitting by combining the predictions of multiple weak models.
- Model complexity control: Simplifying the model architecture or reducing the number of parameters can help prevent overfitting. This can be achieved by reducing the number of layers or nodes in a neural network or by using simpler models like linear regression instead of complex ones like deep neural networks.
- Increasing training data: Providing more diverse and representative training data can help the model generalize better and reduce overfitting. Collecting more data or using techniques like data synthesis can be beneficial.

### Hyper parameters
In machine learning models, there are several common hyperparameters that can be adjusted to optimize the model's performance. Some of these hyperparameters include:
- Learning rate: This determines the step size at each iteration of the optimization algorithm, such as gradient descent. It controls how quickly the model learns from the data.
- Batch size: It specifies the number of training examples used in each iteration of the optimization algorithm. A smaller batch size can lead to faster convergence, but it may also introduce more noise in the parameter updates.
- Number of hidden units or layers: This determines the complexity and capacity of the model. Increasing the number of hidden units or layers can allow the model to capture more complex patterns but may also increase the risk of overfitting.
- Regularization parameters: Regularization techniques like L1 or L2 regularization help prevent overfitting by adding a penalty term to the loss function. The regularization parameter controls the strength of this penalty.
- Dropout rate: Dropout is a regularization technique that randomly sets a fraction of the input units to zero during training. The dropout rate determines the probability of dropping out each unit, preventing over-reliance on specific features.
- Activation functions: Different activation functions, such as sigmoid, ReLU, or tanh, can be used in the model's hidden layers. Choosing the appropriate activation function can impact the model's ability to capture non-linear relationships in the data.
- Number of iterations or epochs: This determines the number of times the model will iterate over the entire training dataset. Increasing the number of iterations can improve the model's performance, but it may also increase training time.


### Validation
The purpose of validation data is to assess the performance of a machine learning model during the training process and to help in selecting the best model hyperparameters. It serves as an intermediate step between the training data and the test data.

During model training, the training data is used to update the model's parameters and optimize its performance. However, solely relying on the training data to evaluate the model's performance can lead to overfitting, where the model becomes too specific to the training data and fails to generalize well to new data.

To prevent overfitting and ensure that the model performs well on unseen data, a portion of the available data is set aside as validation data. The validation data is not used for training the model but is used to evaluate the model's performance on data it hasn't seen before.

The validation data is used to tune the model's hyperparameters, such as learning rate, batch size, regularization strength, etc. Multiple models with different hyperparameter settings are trained using the training data, and their performance is evaluated on the validation data. The model with the best performance on the validation data, typically measured by a chosen evaluation metric (e.g., accuracy, loss), is selected as the final model.

By using validation data, we can make informed decisions about the model's hyperparameters and select the best-performing model that is likely to generalize well to new, unseen data.

It's important to note that the validation data should not be used for model training or parameter updates. Its sole purpose is to evaluate the model's performance and guide the selection of hyperparameters.

### Test data
To mitigate the challenges of using test data for evaluating model performance, you can employ the following strategies:
- Cross-validation: Instead of relying on a single test set, you can use cross-validation techniques like k-fold cross-validation. This involves splitting the data into multiple subsets and performing evaluation on each subset as both training and test data. This helps to reduce the impact of limited sample size and provides a more robust assessment of model performance.
- Stratified sampling: If your data has imbalanced classes, ensure that the test data is representative of the class distribution. Stratified sampling ensures that each class is proportionally represented in the test set, reducing the bias in evaluation metrics.
- Monitoring for distribution shifts: Continuously monitor the data distribution to detect any shifts that may occur between training and test data. If a significant shift is detected, it may be necessary to retrain or fine-tune the model to adapt to the new distribution.
- Hyperparameter tuning on validation data: Use a separate validation set to tune the hyperparameters of the model. By evaluating different hyperparameter configurations on the validation set, you can select the best-performing model that generalizes well to unseen data.
- Ensemble methods: Combine multiple models or predictions to improve performance. Ensemble methods, such as bagging or boosting, can help mitigate the risk of overfitting and improve the overall robustness of the model.
- External validation: If possible, validate the model's performance on external datasets that are different from the training and test data. This helps to assess the generalizability of the model to new and unseen data.
- Regularization techniques: Regularization methods like L1 or L2 regularization can help prevent overfitting by adding a penalty term to the loss function. This encourages the model to learn simpler and more generalizable patterns.
- Model evaluation metrics: Consider using evaluation metrics beyond just accuracy, especially in scenarios with imbalanced classes. Metrics like precision, recall, F1-score, or area under the ROC curve (AUC-ROC) provide a more comprehensive understanding of the model's performance.


### Evaluation metrics
When assessing model performance on validation data, several common evaluation metrics can be used. These metrics provide insights into how well the model is performing and help in comparing different models or tuning hyperparameters. Here are some commonly used evaluation metrics:
- Accuracy: Accuracy measures the proportion of correctly classified instances out of the total number of instances. It is commonly used for classification problems with balanced classes.
- Precision: Precision measures the proportion of true positive predictions out of all positive predictions. It is useful when the focus is on minimizing false positives.
- Recall (Sensitivity or True Positive Rate): Recall measures the proportion of true positive predictions out of all actual positive instances. It is useful when the focus is on minimizing false negatives.
- F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balanced measure of both precision and recall and is useful when there is an imbalance between classes.
- Area Under the ROC Curve (AUC-ROC): AUC-ROC measures the model's ability to distinguish between positive and negative instances across different probability thresholds. It provides an aggregate measure of the model's performance and is commonly used for binary classification problems.
- Mean Squared Error (MSE): MSE measures the average squared difference between the predicted and actual values. It is commonly used for regression problems.
- Root Mean Squared Error (RMSE): RMSE is the square root of MSE and provides a measure of the average difference between the predicted and actual values. It is also commonly used for regression problems.
- Mean Absolute Error (MAE): MAE measures the average absolute difference between the predicted and actual values. It is less sensitive to outliers compared to MSE and is also used for regression problems.
- R-squared (Coefficient of Determination): R-squared measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It provides an indication of how well the model fits the data.





















