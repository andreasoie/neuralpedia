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
Regards to the combinations with batch normalization, we often use batch normalization immediately after a fully connected layer or convolutional layer, and before nonlinear activation functions.

#### There seems to be a certain combination of layers, most frequently used - why is that?

As stated above, it depends. However, most vision deep neural networks follow the combination of convolution, batch normalization and some sort of activation function, normally relu (See [SegNet](https://arxiv.org/abs/1511.00561), [ResNet](https://arxiv.org/abs/1512.03385))

### Transformations & Augmentation (move?)

#### Why do we often need to normalize our data?

We can divide the normalizing process into two steps: pre training and during training.

The most successful DL models are trained on normalized data.
In the pre-processing stage, we often normalize the data to some logical range e.g. 0 to 1. This is done to prevent problems during training such as vanishing or exploding gradients.

For example: a car dataset with a lot of different features, like mileage, horsepower, etc.
Say one car is brand new and has zero milage, whilst another car, e.g. an old indian taxicab, has 100,000,000 milage. During training the high milage value may propagate through the network, ending up causing the gradients to explode (if using e.g. SGD). This effectively makes the network unstable, and will cause trouble to generalize well. The mileage will be prioritized well above the other features.

On the other hand, the common method of batch normalization also normalizes the data to a certain range, but it is done during training after some fully connected / convolutional layer, but prior to nonlinear functions. Sections below shows why we normalize the data in batches.

### Batch normalization

> What is it?

A technique for converting the interlayer outputs of a neural network into a standard format. This effectively'resets' the distribution of the preceding layer's output, allowing the subsequent layer to process it more efficiently.

> Advantages?

Normalization assures that no activation value is too high or too low, and each layer can learn independently of the others, this method leads to quicker learning rates.
The "dropout" rate, or data lost between processing layers, is reduced by normalizing inputs. This enhances accuracy across the network tremendously.

> How does it work?

Batch normalization impacts the output of the previous activation layer by subtracting the batch mean and then dividing by the batch standard deviation to improve the stability of the network.

Since this shifting or scaling of outputs by a randomly initialized parameter reduces the accuracy of the weights in the next layer, a SGD is applied to remove this normalization if the loss function is too high.

Batch normalization provides two more trainable parameters to a layer: the normalized output multiplied by a gamma (standard deviation) parameter, and a beta (mean) parameter. This is why batch normalization and gradient descents work together to allow data to be "denormalized" by modifying only these two weights for each output. By adjusting all other relevant weights, this resulted in decreased data loss and better network stability.

### Propagations

#### What is a forward propagation?

A single forward propagation is a single step in the forward pass of a neural network. This simply means sending the data (e.g. a image) through the network and receiving a output in some form of probability score.

#### What is a backward propagation?

A backward propagation, also called backpropagation, is a single step in the backward pass of a neural network. Its a way of computing gradients of expressions through recursive application of chain rule.

The idea is that we can compute the gradient of the loss function with respect to the weights of the network. The weights would then be adjusted similarly to the equation: $w_i = w_i - \alpha * \partial_i$. Where $\alpha$ is the learning rate, $i$ is the index of the weight and $\partial_i$ is the gradient of the loss function with respect to the weight.

## Activation functions

#### What are they?

As previously described, activation functions are the functions applied to certain hidden layers. In order to solve complex nonlinear problems, we represent nonlinear relationships by nonlinear functions. Generally, most nonlinear functions can work, however we often used the common functions such as sigmoid, relu, softmax, etc.

#### Why are they used?

TODO

#### Do they have any limitations?

TODO

#### Definitions

$\text{ReLU}(x) = (x)^+ = \max(0, x)$

$\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}$

$\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}$

### Regularization

@TODO

#### Methods

There exists many types of techniques to prevent overfitting, however the most commonly known functions are L1 and L2 regularization, max norm and dropout.

**L1 regularization**
: uses the L1 norm, which is the sum of the absolute values of the weights.

**L2 regularization**
: uses the L2 norm, which is the sum of the squares of the weights.

```python
 """ Simple example """

# Tuning
L2_LAMBDA = 0.01

loss = some_loss_function(predicted_values, actual_values)

# Regularization with L2
l2_norm = sum(
        parameter.pow(2.0).sum()
            for param in model.parameters())

# Update loss
loss = loss + L2_LAMBDA * l2_norm
```

**Max norm**: which uses the L-infinity norm, is a technique to prevent the weights from becoming too large.

**Dropout**: see previous explanation.

(L1 & L2 are also scaled by a (small) factor, say gamma, which is a hyperparameter we set prior to training)

### Optimizers

#### What are they?

In layman's terms, optimizers fiddle with the weights to bend and sculpt your model into the most accurate form possible. The loss function provides a sense of direction for the optimizer, indicating whether it is heading in the right or wrong path.

#### How do they work?

More specifically, they use the loss values generated by the loss functions to "optimize" the weights accordingly.

#### So, how do we know which one to use?

Sadly, the is no one-size-fits-all algorithm. Finding the right optimizer totally depends on the problem.
However, there are some common ones that are used in many cases such as SGD, Adam, RMSprop, etc.
In short, use whatever optimizer your familiar with. In practice Adam is currently recommended as the default algorithm to use.

#### Popular algorithms

##### GD & SGD

Gradient Decent: computes the gradient using the whole dataset.

Stochastic Gradient Decent: computes the gradient using a single random sample (realistically, we use a mini-batch of random samples).

Note: SGD achieves faster iterations, but in the trade off of lower convergence rate if training on big datasets.

##### Momentum (not really an optimizer)

The idea behind momentum is essentially to get out of local minima. This is done by using momentum, that is to use the average of the previous steps to determine the next step. We scale momentum by a factor, which is a hyperparameter we set prior to training, between 0 and 1 (e.g. Adam uses 0.9).

Momentum is therefore something other optimizers do include in their implementation.

##### RMSProp

RMSProp, Root Mean Squared, is a variant of SGD. It uses the same principles, but in addition uses a adaptive learning rate. Each parameter is given its own learning rate. This helps the model training by giving a higher learning rate to the weights that need to change a lot, while the ones that are good enough get a lower learning rate.

##### Adam

The Adam algorithm is based on RMSProp and Momentum. The momentum is based on using both the first-order and second-order gradients.

### Loss functions

#### What is it and why is it used?

A loss function is a objective function that computes a single numerical value (often called loss or cost) for the difference between the predicted and actual output e.g. $ loss = some_loss_function(predicted_values, actual_values) $. Essentially, good prediction results in low loss, and bad prediction results in high loss.

Sometimes it isn't always so straight forward as measuring the difference between outputs. Often, we have
figure out what kind of loss type we want to use. Reversely, we might want to sometimes maximize the loss instead of minimizing it - making it a "reward" function.

#### Popular loss functions

##### Mean Absolute Error, or L1

$\begin{equation}
    \text { lossFuncL1 }=\sum_{i=1}^{n}\left|y_{\text {true }}-y_{\text {predicted }}\right|
\end{equation}$

##### Mean Squared Error, or L2

$\begin{equation}
    \text { lossFuncL2 }=\sum_{i=1}^{n}\left(y_{\text {true }}-y_{\text {predicted }}\right)^{2}
\end{equation}$

Note: L2 generally preferred above L1.

##### Cross-Entropy ("log loss")

Instead of measuring the difference between the predicted and actual output, we measure the difference between distributions. We say the cross-entropy between a “true” distribution $p(x)$ and an estimated distribution $q(x)$ is defined as:
$\begin{equation}
    H(p, q)=-\sum_{x} p(x) \log q(x)
\end{equation}$

##### XXXXXXXXX

##### XXXXXXXXX

### Special questions

#### Why do we always need requires-grad during training, and why do we need to reset this?

#### Why is bias added to layers?

### Why do we certain times only add it to a few layers, but not the rest?

### Most deep neural networks are feedforward, why is that - and what does backprop have to do with this?

### Do we need batch normalization between activation layers? Does this incr/decr accuracy?

### Bias and Variance

### Special events

#### Drop-out layers?

#### Learning rate?

#### Changing learning rate progressively (momentum) ?

#### How does i.e.: an image look like before/after a normalization (tfms)?

#### Resetting gradient

#### Underfitting

#### Overfitting

#### Measuring performance

#### Performance indicators?

#### How do we use this neural net?

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
