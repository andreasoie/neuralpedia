## Theory of Neural Networks

Brief interpretations of various subjects in the space of Neural Networks

## Basics

Below shows various questions I've thought of as I'm trying to formulate my theoretical understanding of the subjects. The order of the various subject is not entirely in order. A lot of the answers are based on the conceptual understanding of the subject (the actual implementation is just a lot of matrix multiplication, amirite?).

### Fundamentals

> What is it?

A neural networks consists of nodes, synapses and a signal strength between them commonly referred to as a weight. A node represents a group of weighted inputs. The input is connected to nodes, nodes are connected to other nodes and final nodes are connected to outputs. The interconnection between them have a given weight. The value of the weight decides the importance of its value.

> Why do we use them?

We use the principles of neural networks as machine learning algorithms, implemented as software programs in various programming languages. We use neural networks to solve complex nonlinear problems we encounter in real life situations. Wether that is recognizing patterns, finding hidden relationships or forecasting problems.

> How do we create them?

We create neural networks by building layers of nodes connected to each other. Generally, we consider three types of layers: input, hidden and output.

The input layer consist of the input data to the the network represented as nodes. If the input data, a single node could represent greyscale value. Next, the input layer is connected to one or more hidden layers (where the magic happens). Finally, the last hidden layer is connected to the output layer of the neural network.

In each synapse, we have a weight representing the importance of its preceding variable. If the nodes in hidden layers are represented as special functions, we could for instance say that a single node could simply be the sum of all the inputs.

![no_image](https://github.com/Towed-ROV/api/blob/main/docs/nn/nn.png?raw=true)

> Are there any consequences of using these?

There are many ways to interpret this questions. For instance, does the software programs using neural networks have a specific purpose?
We often see in social media that people are seeing the same patterns in their posts. This is a common phenomenon in social media.

From a environmental perspective, the best models usually are expensive and is actually producing a very high carbon footprint.
Ex: the recent GPT-3 model produced the same amount of carbon during training that would be produced by driving 120 passenger cars for a year ([ref](https://fortune.com/2021/04/21/ai-carbon-footprint-reduce-environmental-impact-of-tech-google-research-study/)).

### Layers

#### What is a input layers, and what do we place here?

The input layer is the first layer of the neural network. This is where we hold our dataset. As nodes are practically functions, we could instead call each node a unit in the input layer, as they purely represent a unique attribute in your dataset (ex: bodyweight, height, color).

#### What is a hidden layer, and what types do we have?

A hidden layer, is a layer of nodes. As you only have access to send data into the input layer, and receive data from the output layer, the hidden layers in between are the ones that are actually doing the magic. Each node in the hidden layer have some sort of mathematical functional characteristic. Depending on the desired output, we can design the function accordingly. For instance: using some sort of activation function, say sigmoid, to output values between 0 and 1 - for small or large input values respectively.

Most common types of activation functions hidden layers are sigmoid, relu, softmax, etc. (described in more detail later).

#### What is a output layer, and how do we design this to meet our goal?

The final layer is the output layer. This layer receives the output from the final hidden layer, and then outputs the desired output. Often we see some sort of activation function in the output layer, for example sigmoid, to output values between 0 and 1. This would then be used to determine the probability of the output, representing the models prediction.

#### Whats the deal with linear layers?

When talking about linear layers, we usually mean a layer where each node reprensets a linear function activation. This essentially means that the layer applies a linear transformation to the incoming data by the well-known formula: **y = ax + b** (or output = weight\*input + bias)

#### Whats the deal with convolutional layers?

In convolutional layers, we essentially perform a N-dimensional convolution of the input to generate some sort of output. In digital signal processing, this is refereed to as a filter. In the domain of neural networks, we call it a kernel. The idea is to apply a kernel to the input to generate feature maps which can represent the input data in a more meaningful way. This is highly useful for image processing, where we can use the kernels to detect edges, detect corners, detect blobs, etc in unusual shapes. This is why convolutional neural networks architectures explicitly assumes that the inputs are images.

#### What happens when we use multiple-hidden layers?

Generally what happens when we have multiple hidden layers is that the total depth of the neural networks is increased and the memory consumption is increased. This allows for more complex models to be trained, where it can generate complex relationships in the data. Wether its always better to increase the numbers of layers is another question. However, we can in short say that the more layers we have, the more complex the model will be. In terms of the performance of the layers it might underfit the data, overfit the data
or simply just perfect.

#### Why do we sometimes drop certain layers?

Often when tuning our neural networks, we use a regularization method to prevent overfitting. One method is called dropout, where we with some probability factor $p$, drop certain nodes in the layers. this essentially means we remove some of the synapses in the network.

#### What the deal with fully-connected layers?

A fully-connected network is a network where all the nodes in the input layer are connected to all the nodes in the output layer. The standard neural network is generally fully connected. Ex: the Multi Layer Perceptron (MLP) architecture is fully connected. In modern architectures, we often use fully connected layers at the very end of the network to compile the data extracted from the previous layer to form the final output.

#### Whats the deal with things between linear layers?

The things, refereed to activation functions, are the things that are applied to the nodes in the hidden layers. In order to represent nonlinear relationships, we have to use nonlinear functions. Generally, most nonlinear functions can work, however we often used the common functions such as sigmoid, relu, softmax, etc.

#### Generally layers have added bias, why is that?

This is simply because it helps you adjust steepness of the activation function. In addition, it also helps to prevent weights from becoming zero unintentionally. A good example for vizualizing this is the well-known linear function **y = ax + b**. Having a bias $b$ gives us the ability to move the line up/down to better fit the data. If the bias $b=0$, then the line goes through the origin which effectively makes the output extremely limited.

#### Whats activation layers?

Essentially just another term for activation functions in neural networks.

#### Does the combinations of layers and activation functions matter?

This is highly dependent on what kind of neural network architecture being used.

#### There seems to be a certain combination of layers, most frequently used - why is that?

As stated above, it depends. However, most vision deep neural networks follow the combination of convolution, batch normalization and some sort of activation function, normally relu (See [SegNet](https://arxiv.org/abs/1511.00561), [ResNet](https://arxiv.org/abs/1512.03385))

### Transformations & Augmentation (move?)

#### Why do we often need to normalize our data?

#### Some data may have a wide-spread of values, how do we cope with this?

#### Explain the following terms with examples

##### Batch normalization

#####

### "Pass" functions

#### Whats a forward pass, and what is imporantant to think of when designing this function?

#### Whats a backward pass (backprop), and what is imporantant to think of when designing this function?

### [Activation functions](#activation_functions)

#### Top 3 most-commenly used

### Regulaztion

### Optimizers

#### Top 3 most-commenly used, and why they work

### Loss functions

#### Why do we need a loss function, and why do we often used predefined functions

#### Top 3 most-commenly used, and why they work

### Special questions

#### Why do we always need requires-grad during training, and why do we need to reset this?

#### Why is bias added to layers?

### Why do we certain times only add it to a few layers, but not the rest?

### Most deep neural networks are feedfoward, why is that - and what does backprop have to do with this?

### Do we need batch normalizaiton between actiavation layers? Does this incr/decr accuracy?

### Special events

#### Drop-out layers?

#### Learning rate?

#### Changig learning rate progressivly (momentum) ?

#### How does i.e.: an image look like before/after a normalization (tfms)?

#### Resetting gradient

#### Underfitting

#### Overfitting

#### Measuring performance

#### Performance indicators?

#### How do we use this neaural net?

#### Do we generally always need a good baseline to compare results to?

#### Inference: GPU vs CPU: when to use what, and what do we consider?

#### Deployment strategies

### A Recipe for Training Neural Networks

>

#### Summarized

1.
2.
3.
4. X.

([ref](http://karpathy.github.io/2019/04/25/recipe/))

#### What do we need inside the training loop?

Controversial recommendations e.g. early stopping? small batch sizes?
