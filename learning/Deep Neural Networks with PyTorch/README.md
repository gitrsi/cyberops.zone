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
$$
y(x)=x^2
$$
  
$$
{dy(x) \over dx}=2x^1
$$
  

    
    
    
    



