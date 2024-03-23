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

