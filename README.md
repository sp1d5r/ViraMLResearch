# Vira ML

Machine learning on the ******ain. It's been blurred to prevent spam emails. 

## Stages of Compilation

1) Start off with your ONNX (Open Neural Network Exchange) Format
2) Determine Partitions in Model
3) Propose some training examples 
4) Train lightweight Gaussian Mixture Model wrapper to break up model and verification steps.

## Loading ONNX File

ONNX (Open Neural Network Exchange) facilitates a flexible and interoperable ecosystem for machine learning models by 
defining a common set of operators and a standard file format. This universality allows AI developers to transition 
models between various frameworks, tools, and platforms without getting locked into one ecosystem.

### How ONNX Works

At the heart of ONNX is the concept of a computational graph. This graph represents the model in terms of:

- **Nodes**: Each node represents an operation or a step in the computation, such as convolution, batch normalization, or a mathematical operation like addition or multiplication. These operations are the building blocks of the neural network.

- **Edges**: The edges between nodes signify the flow of data. They represent tensors, the multi-dimensional arrays holding the numerical data that moves between operations.

- **Inputs and Outputs**: The graph has designated inputs and outputs, allowing for data to enter and exit the model. These are often tensors representing features for prediction and the prediction itself, respectively.

- **Attributes**: Nodes can have attributes, which are additional parameters required for the operation. For example, a convolution node might have attributes for kernel size, stride, and padding.

### Graph Structure

An ONNX model's graph structure enables the clear and explicit representation of the model's computation. This structure allows developers to visualize the model's flow, understand its components, and identify areas for optimization or modification.

The computational graph approach also aids in the model's execution across different hardware and software environments, as the abstract representation is independent of the underlying technology stack.

### Advantages of ONNX

- **Interoperability**: With its common framework, ONNX enables models to be shared between various AI development communities, increasing collaboration and reducing redundancy.
  
- **Flexibility**: Developers can easily move models between state-of-the-art tools and choose the best combination for training, optimization, and deployment.

- **Efficiency**: ONNX models are optimized for performance on different hardware, making it easier to deploy AI solutions across diverse environments, from cloud-based services to edge devices.

By leveraging ONNX, Vira ML ensures that its machine learning solutions are robust, portable, and ready for the future of AI across different platforms, including the blockchain.### How ONNX Works

At the heart of ONNX is the concept of a computational graph. This graph represents the model in terms of:

- **Nodes**: Each node represents an operation or a step in the computation, such as convolution, batch normalization, or a mathematical operation like addition or multiplication. These operations are the building blocks of the neural network.

- **Edges**: The edges between nodes signify the flow of data. They represent tensors, the multi-dimensional arrays holding the numerical data that moves between operations.

- **Inputs and Outputs**: The graph has designated inputs and outputs, allowing for data to enter and exit the model. These are often tensors representing features for prediction and the prediction itself, respectively.

- **Attributes**: Nodes can have attributes, which are additional parameters required for the operation. For example, a convolution node might have attributes for kernel size, stride, and padding.


### Code Implementation

We define a wrapper around the model which includes a way to print the nodes in the graph:
```python
Model inputs: ['input_ids', 'attention_mask']
Model outputs: ['start_logits', 'end_logits']
Node name: Shape_0
Node type: Shape
Node inputs: ['input_ids']
Node outputs: ['104']
----------
Node name: Constant_1
Node type: Constant
Node inputs: []
Node outputs: ['105']
----------
Node name: Gather_2
Node type: Gather
Node inputs: ['104', '105']
Node outputs: ['106']
----------
Node name: Constant_3
Node type: Constant
Node inputs: []
Node outputs: ['107']
----------
Node name: Unsqueeze_4
Node type: Unsqueeze
Node inputs: ['106']
Node outputs: ['111']
----------
Node name: Constant_5
Node type: Constant
Node inputs: []
Node outputs: ['113']
----------
Node name: Slice_6
Node type: Slice
Node inputs: ['107', '820', '111', '821', '113']
Node outputs: ['114']
----------
...
----------
Node name: Squeeze_516
Node type: Squeeze
Node inputs: ['816']
Node outputs: ['start_logits']
----------
Node name: Squeeze_517
Node type: Squeeze
Node inputs: ['817']
Node outputs: ['end_logits']
----------

```


## Determining Partitions

Machine learning models are separated out into layers, each layer is can be considered a transformation from inputs in some higher dimentional space 
to an output in some other higher dimensional space. We want to partition this directed graph into sub-graphs. 


**Why?**


If we can partition the structure of the graph based on some factor - in our case it will be comptuation per proof - we can allow parallel 
verification on the network. 