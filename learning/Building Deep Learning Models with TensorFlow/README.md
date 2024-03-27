![Building Deep Learning Models with TensorFlow](https://github.com/gitrsi/cyberops.zone/blob/main/assets/img/Deep_Learning.jpg "Building Deep Learning Models with TensorFlow")

> :bulb: Notes on "Building Deep Learning Models with TensorFlow"



# Neural Networks, Depp Learning with Tensor flow

## Introduction to Tensor Flow
- TensorFlow is an open-source library developed by the Google Brain Team for tasks that require heavy numerical computations, particularly in machine learning and deep neural networks.
- TensorFlow offers both a Python and a C++ API, with the Python API being more complete and easier to use.
- It has great compilation times compared to other deep learning libraries and supports CPUs, GPUs, and distributed processing in a cluster.
- TensorFlow uses a data flow graph structure, which consists of nodes representing mathematical operations and edges representing tensors (multi-dimensional arrays).
- Tensors are the data passed between operations and can be zero-dimensional (scalar values), one-dimensional (vectors), two-dimensional (matrices), and so on.
- Placeholders are used to pass data into the graph, while variables are used to share and persist values manipulated by the program.
- TensorFlow's flexible architecture allows computation to be deployed on different devices, such as CPUs, GPUs, servers, or even mobile devices.
- TensorFlow is well-suited for deep learning applications due to its built-in support for neural networks, trainable mathematical functions, auto-differentiation, and optimizers.

### Meaning of tensor
In TensorFlow all data is passed between operations in a computation graph, and these are passed in the form of Tensors, hence the name of TensorFlow
The word tensor from new latin means "that which stretches". It is a mathematical object that is named "tensor" because an early application of tensors was the study of materials stretching under tension. The contemporary meaning of tensors can be taken as multidimensional arrays. 

Multidimensional arrays? 

Going back a little bit to physics to understand the concept of dimensions:

![Dimension levels](dimension_levels.png "Dimension levels")

The zero dimension can be seen as a point, a single object or a single item.

The first dimension can be seen as a line, a one-dimensional array can be seen as numbers along this line, or as points along the line. One dimension can contain infinite zero dimension/points elements.

The second dimension can be seen as a surface, a two-dimensional array can be seen as an infinite series of lines along an infinite line. 

The third dimension can be seen as volume, a three-dimensional array can be seen as an infinite series of surfaces along an infinite line.

The Fourth dimension can be seen as the hyperspace or spacetime, a volume varying through time, or an infinite series of volumes along an infinite line. And so forth on...

As mathematical objects:
![Mathematical objects](math_objects_dimensions.png "Mathematical objects")



### Architecture:
![Architecture of TensorFlow](tensorflow_architecture.png "Architecture of TensorFlow")

The ingredients of a computation graph in TensorFlow include:
- Nodes: Nodes represent mathematical operations or computations. Each node performs a specific operation on the input data and produces an output.
- Edges: Edges represent the flow of data between nodes. They connect the output of one node to the input of another node. The data flowing through the edges are multi-dimensional arrays called tensors.
- Tensors: Tensors are the data structures that flow through the computation graph. They are multi-dimensional arrays that can be zero-dimensional (scalar values), one-dimensional (vectors), two-dimensional (matrices), or higher-dimensional.
- Placeholders: Placeholders are special nodes in the graph that act as "holes" where data can be fed into the graph during execution. They allow you to define the structure of the graph without providing the actual data. Placeholders need to be initialized with data before executing the graph.
- Variables: Variables are nodes that hold values that can be manipulated by the program. They are used to share and persist values across different executions of the graph. Variables are typically used to store the parameters of a model that are updated during the training process.
- Operations: Operations are the mathematical computations performed by the nodes in the graph. They can include basic arithmetic operations like addition and multiplication, as well as more complex operations like matrix multiplication or convolution.
- Sessions: A session is an environment for executing the computation graph. It encapsulates the state of the graph and allows you to run the computations and retrieve the results. Sessions can be run on different devices like CPUs, GPUs, or even distributed systems.

![Ingredients of a computation graph in TensorFlow](tensorflow_computation_graph_ingredients.png "Ingredients of a computation graph in TensorFlow")

TensorFlow's data flow graph structure is beneficial for visualizing and organizing computations in several ways:
- Visualization: The data flow graph provides a visual representation of the computational flow, making it easier to understand and debug complex models. It allows you to see the connections between different operations and how data flows through the graph.
- Modularity: The graph structure allows you to break down complex computations into smaller, modular units represented by nodes (operations). This modularity makes it easier to manage and organize the code, as you can focus on individual operations and their inputs and outputs.
- Parallelism: The data flow graph structure is a common programming model for parallel computing. It enables TensorFlow to automatically parallelize the execution of operations, taking advantage of multi-core CPUs, GPUs, or distributed computing environments. This parallelism can significantly speed up the computation of large-scale models.
- Optimization: TensorFlow's graph structure allows for automatic optimization of computations. The graph can be optimized by rearranging operations, eliminating redundant calculations, and applying other optimization techniques. This optimization process helps improve the efficiency and performance of the computations.

Advantages of using TensorFlow's Python API over the C++ API are:
- Completeness and Ease of Use: The Python API is more comprehensive and easier to use compared to the C++ API. It provides a higher-level interface that simplifies the process of building and training deep learning models. Python is known for its simplicity and readability, making it more accessible for beginners and researchers.
- Rich Ecosystem: TensorFlow's Python API benefits from a vast ecosystem of libraries and tools that are built around Python. This includes popular libraries like NumPy, Pandas, and Matplotlib, which seamlessly integrate with TensorFlow and provide additional functionality for data manipulation, visualization, and analysis.
- Rapid Prototyping: Python's dynamic nature and interactive development environment make it ideal for rapid prototyping and experimentation. With the Python API, you can quickly iterate on your models, try different architectures, and experiment with various hyperparameters. This agility is crucial in the fast-paced field of deep learning.
- Community Support: Python has a large and active community of developers and researchers, which means there is extensive community support available for TensorFlow's Python API. You can find numerous tutorials, code examples, and forums where you can seek help and collaborate with others working on similar problems.
- Integration with Data Science Workflow: Python is widely used in the data science community, and many data science tools and frameworks are built around it. TensorFlow's Python API seamlessly integrates with these tools, allowing you to incorporate deep learning models into your existing data science workflow.

TensorFlow is commonly used for image-related tasks. Here are some ways TensorFlow can be used with images:
- Image Classification: TensorFlow can be used to build image classification models that can classify images into different categories or classes. You can train models to recognize objects, animals, or even specific features within images.
- Object Detection: TensorFlow provides tools and pre-trained models for object detection, which involves identifying and localizing multiple objects within an image. This is useful for applications like self-driving cars, surveillance systems, or image-based search engines.
- Image Segmentation: TensorFlow can be used for image segmentation, where the goal is to classify each pixel in an image into different categories. This technique is useful for tasks like medical image analysis, where you want to identify and segment specific structures or regions within an image.
- Style Transfer: TensorFlow can be used to apply artistic styles to images using deep learning techniques. Style transfer models can transform images to mimic the style of famous paintings or other artistic styles.
- Image Generation: TensorFlow can generate new images using generative models like Generative Adversarial Networks (GANs) or Variational Autoencoders (VAEs). These models can learn to generate realistic images based on training data.
- Image Super-Resolution: TensorFlow can be used to enhance the resolution and quality of low-resolution images. Super-resolution models can generate high-resolution versions of images, which is useful in applications like medical imaging or enhancing low-quality images.

### Usage
Simple TensorFlow example using numpy:

    import tensorflow as tf
    import numpy as np

    # Create numpy arrays
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    # Create TensorFlow constants from numpy arrays
    x_tf = tf.constant(x)
    y_tf = tf.constant(y)

    # Perform element-wise multiplication
    result = tf.multiply(x_tf, y_tf)

    # Create a TensorFlow session and run the computation
    with tf.Session() as sess:
        output = sess.run(result)
        print(output)

In this example, we create two numpy arrays x and y. We then convert them into TensorFlow constants x_tf and y_tf. We use the tf.multiply() function to perform element-wise multiplication between the two arrays. Finally, we create a TensorFlow session, run the computation using sess.run(), and print the output.


## Tensor Flow 2.x and Eager Execution
### Advantages
Eager Execution mode in TensorFlow offers several benefits that make it a powerful tool for deep learning development. Here are the benefits of using Eager Execution mode:
- Immediate Execution: With Eager Execution, TensorFlow code is executed immediately, line by line, just like ordinary Python code. This allows for instant feedback and makes it easier to debug and understand the behavior of the code.
- Improved Debugging: Eager Execution provides access to intermediate results at any point during the execution. This makes it easier to inspect and debug the code, as you can directly print or visualize the values of tensors and variables.
- Dynamic Computation: Eager Execution allows for dynamic computation graphs. This means that the control flow statements, such as loops and conditionals, can be used directly in TensorFlow code. This flexibility enables more complex and dynamic models to be built.
- Natural Control Flow: Eager Execution mode allows for the use of Python control flow statements like if-else conditions and for-loops directly in TensorFlow code. This makes the code more readable and easier to write, especially for complex models.
- Easy Transition: Eager Execution is the default mode in TensorFlow 2.x, making it easier for developers to transition from TensorFlow 1.x to the latest version. The code written in TensorFlow 1.x can be executed in Eager Execution mode without any modifications.
- Interactive Development: Eager Execution mode enables interactive development and experimentation. You can run TensorFlow operations and immediately see the results, which facilitates rapid prototyping and exploration of different model architectures.
- Compatibility: Eager Execution mode is compatible with other Python libraries and tools, making it easier to integrate TensorFlow with existing workflows and frameworks.

### V 2.x
The major changes in TensorFlow 2.x include:
- Integration with Keras: Keras has become the official high-level API for TensorFlow. It offers user-friendly abstractions for developing deep learning models and is now tightly integrated with TensorFlow.
- Performance optimizations: TensorFlow 2.x includes performance improvements, making it faster and more efficient. It also provides better support for GPU acceleration, allowing for faster training and inference on GPUs.
- Eager Execution: TensorFlow 2.x introduces Eager Execution as the default mode. Eager Execution allows code to be executed immediately, line by line, making TensorFlow code look like ordinary Python code. It enables easier debugging and provides instant access to intermediate results.
- Improved APIs: TensorFlow 2.x offers improved APIs for better usability. It simplifies the process of building and training models, making it more intuitive and user-friendly.

Eager Execution mode in TensorFlow 2.x allows code to be executed immediately, line by line, making TensorFlow code look like ordinary Python code. Here's how Eager Execution mode works:
- By default, Eager Execution mode is enabled in TensorFlow 2.x, so you don't need to make any changes to your code when transitioning between TensorFlow versions.
- In TensorFlow 1.x, when you define a computation graph, the actual computation doesn't happen until you run the graph within a TensorFlow session. Intermediate results are not accessible until then, making it difficult to debug and inspect values during development.
- With Eager Execution mode, as soon as you define and execute operations, the computations are immediately performed and the results are available. This allows you to access intermediate results at any point during the execution, making it easier to debug and understand the behavior of your code.
- Eager Execution mode makes TensorFlow code more intuitive and easier to work with, as it eliminates the need for a separate session and allows for immediate feedback on the results of each operation.
- The data type of tensors also changes in Eager Execution mode. In TensorFlow 1.x, tensors are of type tensorflow.python.framework.ops.Tensor, while in TensorFlow 2.x with Eager Execution enabled, tensors are of type EagerTensor. Eager tensors have additional functionality, allowing you to obtain intermediate results at any time.

### Keras integration
In TensorFlow 2.x, Keras has become the official high-level API for TensorFlow. Here's an explanation of the integration of Keras in TensorFlow:
- Keras is a popular Deep Learning API written in Python. It is known for its user-friendliness and offers abstractions that make it easy to develop deep learning models.
- In TensorFlow 1.x, Keras was a separate library that could be used with TensorFlow as a backend. However, in TensorFlow 2.x, Keras has been integrated as the default high-level API for TensorFlow.
- This integration means that TensorFlow users, especially Python developers, can now develop models more easily using Keras interfaces while leveraging the powerful capabilities of TensorFlow in the backend.
- With the integration, TensorFlow provides a unified and consistent interface for building and training deep learning models. This makes it easier for developers to work with TensorFlow and Keras together.
- The integration also brings several benefits. First, it simplifies the development process by providing a high-level API that abstracts away many low-level details. This allows developers to focus more on the model architecture and less on the implementation details.
- Second, the integration enables seamless interoperability between TensorFlow and Keras. You can use Keras layers, models, and utilities directly within TensorFlow code, making it easier to build complex deep learning models.
- Additionally, TensorFlow 2.x with Keras integration provides improved performance optimizations, multi-GPU support, and improved APIs for better usability for GPU acceleration.

### Usage
TensorFlow example that demonstrates the usage of eager execution with numpy:

    import tensorflow as tf
    import numpy as np

    # Enable eager execution
    tf.enable_eager_execution()

    # Create numpy arrays
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])

    # Perform element-wise multiplication using TensorFlow operations
    result = tf.multiply(x, y)

    # Print the result
    print(result.numpy())

In this example, we first enable eager execution using tf.enable_eager_execution(). This allows TensorFlow operations to be executed immediately and returns numpy arrays as results.

In TensorFlow 1.x each tensor is of type tensorflow.python.framework.ops.Tensor. With eager execution enabled, the type changes to EagerTensor.
While having programmatically similar behavior, eager tensors have additional functionality. This way intermediate results can be obtained at any time.

We then create two numpy arrays x and y. Using TensorFlow's tf.multiply() function, we perform element-wise multiplication between the two arrays. The result is a TensorFlow eager tensor.

Finally, we print the result using result.numpy(), which converts the TensorFlow eager tensor back to a numpy array for easy printing.

Eager execution allows you to interactively work with TensorFlow operations just like regular Python code, making it easier to debug and experiment with your models.


## Deep learning



## Deep neural networks



