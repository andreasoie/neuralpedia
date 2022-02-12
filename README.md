# Neural Networks

Brief interpretations of various subjects in the space of Neural Networks

## Basics

Below shows various questions I've thought of as I'm trying to formulate my theoretical understanding of the subjects. The order of the various subject is not entirely in order. A lot of the answers are based on the conceptual understanding of the subject (the actual implementation is just a lot of matrix multiplication, amirite?).

### Fundamentals

**What is it?**

A neural networks consists of nodes, synapses and a signal strength between them commonly referred to as a weight. A node represents a group of weighted inputs. The input is connected to nodes, nodes are connected to other nodes and final nodes are connected to outputs. The interconnection between them have a given weight. The value of the weight decides the importance of its value.

**Why do we use them?**

We use the principles of neural networks as machine learning algorithms, implemented as software programs in various programming languages. We use neural networks to solve complex nonlinear problems we encounter in real life situations. Wether that is recognizing patterns, finding hidden relationships or forecasting problems.

**How do we create them?**

We create neural networks by building layers of nodes connected to each other. Generally, we consider three types of layers: input, hidden and output.

The input layer consist of the input data to the the network represented as nodes. If the input data, a single node could represent greyscale value. Next, the input layer is connected to one or more hidden layers (where the magic happens). Finally, the last hidden layer is connected to the output layer of the neural network.

In each synapse, we have a weight representing the importance of its preceding variable. If the nodes in hidden layers are represented as special functions, we could for instance say that a single node could simply be the sum of all the inputs.

**Are there any consequences of using these?**

There are many ways to interpret this questions. For instance, does the software programs using neural networks have a specific purpose?
We often see in social media that people are seeing the same patterns in their posts. This is a common phenomenon in social media.

From a environmental perspective, the best models usually are expensive and is actually producing a very high carbon footprint.
Ex: the recent GPT-3 model produced the same amount of carbon during training that would be produced by driving 120 passenger cars for a year ([ref](https://fortune.com/2021/04/21/ai-carbon-footprint-reduce-environmental-impact-of-tech-google-research-study/)).

### Layers

##### What is a input layers, and what do we place here?

The input layer is the first layer of the neural network. This is where we hold our dataset. As nodes are practically functions, we could instead call each node a unit in the input layer, as they purely represent a unique attribute in your dataset (ex: bodyweight, height, color).

##### What is a hidden layer, and what types do we have?

A hidden layer, is a layer of nodes. As you only have access to send data into the input layer, and receive data from the output layer, the hidden layers in between are the ones that are actually doing the magic. Each node in the hidden layer have some sort of mathematical functional characteristic. Depending on the desired output, we can design the function accordingly. For instance: using some sort of activation function, say sigmoid, to output values between 0 and 1 - for small or large input values respectively.

Most common types of activation functions hidden layers are sigmoid, relu, softmax, etc. (described in more detail later).

##### What is a output layer, and how do we design this to meet our goal?

The final layer is the output layer. This layer receives the output from the final hidden layer, and then outputs the desired output. Often we see some sort of activation function in the output layer, for example sigmoid, to output values between 0 and 1. This would then be used to determine the probability of the output, representing the models prediction.

##### Whats the deal with linear layers?

When talking about linear layers, we usually mean a layer where each node reprensets a linear function activation. This essentially means that the layer applies a linear transformation to the incoming data by the well-known formula: **y = ax + b** (or output = weight\*input + bias)

##### Whats the deal with convolutional layers?

In convolutional layers, we essentially perform a N-dimensional convolution of the input to generate some sort of output. In digital signal processing, this is refereed to as a filter. In the domain of neural networks, we call it a kernel. The idea is to apply a kernel to the input to generate feature maps which can represent the input data in a more meaningful way. This is highly useful for image processing, where we can use the kernels to detect edges, detect corners, detect blobs, etc in unusual shapes. This is why convolutional neural networks architectures explicitly assumes that the inputs are images.

##### What happens when we use multiple-hidden layers?

Generally what happens when we have multiple hidden layers is that the total depth of the neural networks is increased and the memory consumption is increased. This allows for more complex models to be trained, where it can generate complex relationships in the data. Wether its always better to increase the numbers of layers is another question. However, we can in short say that the more layers we have, the more complex the model will be. In terms of the performance of the layers it might underfit the data, overfit the data
or simply just perfect.

##### Why do we sometimes drop certain layers?

Often when tuning our neural networks, we use a regularization method to prevent overfitting. One method is called dropout, where we with some probability factor $p$, drop certain nodes in the layers. this essentially means we remove some of the synapses in the network.

##### What the deal with fully-connected layers?

A fully-connected network is a network where all the nodes in the input layer are connected to all the nodes in the output layer. The standard neural network is generally fully connected. Ex: the Multi Layer Perceptron (MLP) architecture is fully connected. In modern architectures, we often use fully connected layers at the very end of the network to compile the data extracted from the previous layer to form the final output.

##### Whats the deal with things between linear layers?

The things, refereed to activation functions, are the things that are applied to the nodes in the hidden layers. In order to represent nonlinear relationships, we have to use nonlinear functions. Generally, most nonlinear functions can work, however we often used the common functions such as sigmoid, relu, softmax, etc.

##### Generally layers have added bias, why is that?

This is simply because it helps you adjust steepness of the activation function. In addition, it also helps to prevent weights from becoming zero unintentionally. A good example for vizualizing this is the well-known linear function **y = ax + b**. Having a bias $b$ gives us the ability to move the line up/down to better fit the data. If the bias $b=0$, then the line goes through the origin which effectively makes the output extremely limited.

##### Whats activation layers?

Essentially just another term for activation functions in neural networks.

##### Does the combinations of layers and activation functions matter?

This is highly dependent on what kind of neural network architecture being used.
Regards to the combinations with batch normalization, we often use batch normalization immediately after a fully connected layer or convolutional layer, and before nonlinear activation functions.

##### There seems to be a certain combination of layers, most frequently used - why is that?

As stated above, it depends. However, most vision deep neural networks follow the combination of convolution, batch normalization and some sort of activation function, normally relu (See [SegNet](https://arxiv.org/abs/1511.00561), [ResNet](https://arxiv.org/abs/1512.03385))

### Transformations & Augmentation (move?)

##### Why do we often need to normalize our data?

We can divide the normalizing process into two steps: pre training and during training.

The most successful DL models are trained on normalized data.
In the pre-processing stage, we often normalize the data to some logical range e.g. 0 to 1. This is done to prevent problems during training such as vanishing or exploding gradients.

For example: a car dataset with a lot of different features, like mileage, horsepower, etc.
Say one car is brand new and has zero milage, whilst another car, e.g. an old indian taxicab, has 100,000,000 milage. During training the high milage value may propagate through the network, ending up causing the gradients to explode (if using e.g. SGD). This effectively makes the network unstable, and will cause trouble to generalize well. The mileage will be prioritized well above the other features.

On the other hand, the common method of batch normalization also normalizes the data to a certain range, but it is done during training after some fully connected / convolutional layer, but prior to nonlinear functions. Sections below shows why we normalize the data in batches.

### Batch normalization

**What is it?**

A technique for converting the interlayer outputs of a neural network into a standard format. This effectively'resets' the distribution of the preceding layer's output, allowing the subsequent layer to process it more efficiently. "Keeping activations in check".

**Advantages?**

Normalization assures that no activation value is too high or too low, and each layer can learn independently of the others, this method leads to quicker learning rates.
The "dropout" rate, or data lost between processing layers, is reduced by normalizing inputs. This enhances accuracy across the network tremendously.

**How does it work?**

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
We use certain activation functions depending on what kind of problem we want to solve. For example, if we were to create a MNIST classifier the final activation function of a neural network could be the sofmax function (multi-class classifier) or the sigmoid (binary classifier). For more complex architecture, the solution isnt always so clear. For instance for problems such as image recognition, we change the strategy and often use linear layers as the final layer instead.

#### What function should I use?
Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of “dead” units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout ([ref](https://cs231n.github.io/neural-networks-1/#actfun)).
#### Definitions

$\text{ReLU}(x) = (x)^+ = \max(0, x)$

$\text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}$

$\text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}$

### Regularization
As we're building models to solve different problems with our training data, we often may have to add regularization to our model to ensure its not learning the patterns of the traning data too well. The regularization helps to prevent overfitting, and is a overall hope to make sure the accuracy for both training and validation data is almost similar. We add regularization by adding a penalty (extra loss) to the calculated loss.
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

(L1 & L2 are also scaled by a small factor (hyperparameter))

### Optimizers

#### What are they?

In layman's terms, optimizers fiddle with the weights to bend and sculpt your model into the most accurate form possible. The loss function provides a sense of direction for the optimizer, indicating whether it is heading in the right or wrong path.

#### How do they work?

More specifically, they use the loss values generated by the loss functions to "optimize" the weights accordingly.

#### So, how do we know which one to use?

Sadly, the is no one-size-fits-all algorithm. Finding the right optimizer totally depends on the problem. However, there are some common ones that are used in many cases such as SGD, Adam, RMSprop, etc. In short, use whatever optimizer your familiar with. In practice Adam is currently recommended as the default algorithm to use.

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

A loss function is a objective function that computes a single numerical value (often called loss or cost) for the difference between the predicted and actual output e.g.
$ loss = someLossFunction(predictedValues, actualValues) $
Essentially, good prediction results in low loss, and bad prediction results in high loss.

Sometimes it isn't always so straight forward as measuring the difference between outputs. Often we have
figure out what kind of loss type we want to use. Reversely, we might want to sometimes maximize the loss instead of minimizing it - making it a "reward" function.

#### Popular loss functions

##### Mean Absolute Error, or L1

$\begin{equation}
    \text { lossFuncL1 }=\sum_{i=1}^{n}\left|y_{\text {true }}-y_{\text {predicted }}\right|
\end{equation}$

##### Mean Squared Error, or L2

$\begin{equation}
    \text { lossFuncL2 }=\sum_{i
    =1}^{n}\left(y_{\text {true }}-y_{\text {predicted }}\right)^{2}
\end{equation}$

Note: L2 generally preferred above L1.

##### Cross-Entropy ("log loss")

Instead of measuring the difference between the predicted and actual output, we measure the difference between distributions. We say the cross-entropy between a “true” distribution $p(x)$ and an estimated distribution $q(x)$ is defined as:|
$\begin{equation}
    H(p, q)=-\sum_{x} p(x) \log q(x)
\end{equation}$


#### Spesific topic questions

##### Why do we always need requires-grad during training in PyTorch, and why do we need to reset this?

In PyTorch, we need to specify whether we want to compute gradients with respect to the parameters of the model. This is done by setting requires_grad to True or False on the object (e.g. param.requires_grad = True).

When we then do backpropagation, we accumulate gradients at each leaf node.
If we then continue to train without zeroing the gradients before the forward pass, the gradients will be incorrect.
Basically run .zero_grad() before backward().

##### When do we add bias to layers?

Commonly we set bias=False for Linear/Conv2D -layers when using techniques like BatchNorm right after. This is because BatchNorm, as previously explained, kind of rescales the output from the previous layers making the added bias useless.

##### Most deep neural networks are feedforward, why is that - and what does backprop have to do with this?
Neural networks are called feedforward, as the input data is fed through the network resulting in a output. In a single forward pass, there is not feedback connections. Its a single forward propagation from input layer to output layer. Feedback connections should not be confused with backprop in this context. When we have feedback connections, we're often talking about recurrent neural network opposted to convolutional neural networks.

##### Bias and Variance: the two big sources of error
Bias: error rate on training set.
Variance: error rate betwen training set and test set.

#### Questionnaire
@TODO

- What is a dropout layer?
- How do we change learning rate progressively?
- Why do we need to zero gradients when doing proper backpropagation?
- Techincally, what happens when we underfit and overfit the data?

##### @TODO topics

- Controversial recommendations e.g. early stopping? small batch sizes?
- Measuring performance
- Performance indicators?
- Vizualation techniques for understanding your neural net.
- Bias-Variance & Precision-Recall tradeoff
- Tuning GPU / utilizing max efficiency.

##### Deployment strategies
@TODO

- Local vs. Cloud
- Speed / compute
- Edge devices
- Techniques for inference speed


### Training Neural Networks like [*Karpathy*](http://karpathy.github.io/2019/04/25/recipe/)

**Note 1: Neural net training is a leaky abstraction**

They are not “off-the-shelf” technology the second you deviate slightly from training an ImageNet classifier. By picking on backpropagation and calling it a “leaky abstraction”, but the situation is unfortunately much more dire. Backprop + SGD does not magically make your network work. BatchNorm does not magically make it converge faster. RNNs don’t magically let you “plug in” text. Just because you can formulate your problem as RL doesn’t mean you should. If you insist on using the technology without understanding how it works you are likely to fail. See notes on backpropagation [*here*](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

**Note 2: Neural net training fails silently**

When you break or misconfigure code you will often get some kind of an exception. You plugged in an integer where something expected a string. The function only expected 3 arguments. This import failed. That key does not exist. The number of elements in the two lists isn’t equal. In addition, it’s often possible to create unit tests for a certain functionality.

This is just a start when it comes to training neural nets. Everything could be correct syntactically, but the whole thing isn’t arranged properly, and it’s really hard to tell. The “possible error surface” is large, logical (as opposed to syntactic), and very tricky to unit test. For example, perhaps you forgot to flip your labels when you left-right flipped the image during data augmentation. Your neural network can still (shockingly) work pretty well because it can internally learn to detect flipped images and then it left-right flips its predictions. Or maybe your autoregressive model accidentally takes the thing it’s trying to predict as an input due to an off-by-one bug. Or you tried to clip your gradients but instead clipped the loss, causing the outlier examples to be ignored during training. Or you initialized your weights from a pretrained checkpoint but didn’t use the original mean. Or you just screwed up the settings for regularization strengths, learning rate, its decay rate, model size, etc. Therefore, your misconfigured neural network will throw exceptions only if you’re lucky; most of the time it'll train, but silently work a bit worse.

As a result, (and this is *reeaally* difficult to over-emphasize) a “fast and furious” approach to training neural networks does not work and only leads to suffering. Now, suffering is a perfectly natural part of getting a neural network to work well, but it can be mitigated by being thorough, defensive, paranoid, and obsessed with visualizations of basically every possible thing. The qualities that in my experience correlate most strongly to success in deep learning are patience and attention to detail.

#### Training Recipe <sup>

1. **Become one with the data**
2. **Set up the end-to-end training/evaluation skeleton + get dumb baselines**
3. **Overfit**
4. **Regularize**
5. **Tune**
6. **Squeeze out the juice**

At every step of the way we make concrete hypotheses about what will happen and then either validate them with an experiment or investigate until we find some issue. What we try to prevent very hard is the introduction of a lot of “unverified” complexity at once, which is bound to introduce bugs/misconfigurations that will take forever to find (if ever). If writing your neural net code was like training one, you’d want to use a very small learning rate and guess and then evaluate the full test set after every iteration.

**1. Become one with the data**

The first step to training a neural net is to not touch any neural net code at all and instead begin by thoroughly inspecting your data. This step is critical. I like to spend copious amount of time (measured in units of hours) Scan through examples, understand their distribution and look for patterns. Duplicate examples? Corrupted images / labels? Data imbalances and biases? Pay attention to your own process for classifying the data, which hints at the kinds of architectures we’ll eventually explore. As an example - are very local features enough or do we need global context? How much variation is there and what form does it take? What variation is spurious and could be preprocessed out? Does spatial position matter or do we want to average pool it out? How much does detail matter and how far could we afford to downsample the images? How noisy are the labels?

In addition, since the neural net is effectively a compressed/compiled version of your dataset, you’ll be able to look at your network (mis)predictions and understand where they might be coming from. And if your network is giving you some prediction that doesn’t seem consistent with what you’ve seen in the data, something is off.

Once you get a qualitative sense it is also a good idea to write some simple code to search/filter/sort by whatever you can think of (e.g. type of label, size of annotations, number of annotations, etc.) and visualize their distributions and the outliers along any axis. The outliers especially almost always uncover some bugs in data quality or preprocessing.

**2. Set up the end-to-end training/evaluation skeleton + get dumb baselines**
Now that we understand our data, can we reach for our super fancy Multi-scale ASPP FPN ResNet and begin training awesome models? NO! Our next step is to set up a full training + evaluation skeleton and gain trust in its correctness via a series of experiments. At this stage it is best to pick some simple model that you couldn’t possibly have screwed up somehow - e.g. a linear classifier, or a very tiny ConvNet. We’ll want to train it, visualize the losses, any other metrics (e.g. accuracy), model predictions, and perform a series of ablation experiments with explicit hypotheses along the way.

*Tips & tricks for this stage*:

- **fix random seed**. Always use a fixed random seed to guarantee that when you run the code twice you will get the same outcome. This removes a factor of variation and will help keep you sane.
  
- **simplify**. Make sure to disable any unnecessary fanciness. As an example, definitely turn off any data augmentation at this stage. Data augmentation is a regularization strategy that we may incorporate later, but for now it is just another opportunity to introduce some dumb bug.
  
- **add significant digits to your eval**. When plotting the test loss run the evaluation over the entire (large) test set. Do not just plot test losses over batches and then rely on smoothing them in Tensorboard. We are in pursuit of correctness and are very willing to give up time for staying sane.

- **verify loss @ init**. Verify that your loss starts at the correct loss value. E.g. if you initialize your final layer correctly you should measure -log(1/n_classes) on a softmax at initialization. The same default values can be derived for L2 regression, Huber losses, etc.
  
- **init well**. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves where in the first few iteration your network is basically just learning the bias.
  
- **human baseline**. Monitor metrics other than loss that are human interpretable and checkable (e.g. accuracy). Whenever possible evaluate your own (human) accuracy and compare to it. Alternatively, annotate the test data twice and for each example treat one annotation as prediction and the second as ground truth.
  
- **input-indepent baseline**. Train an input-independent baseline, (e.g. easiest is to just set all your inputs to zero). This should perform worse than when you actually plug in your data without zeroing it out. Does it? i.e. does your model learn to extract any information out of the input at all?
  
- **overfit one batch**. Overfit a single batch of only a few examples (e.g. as little as two). To do so we increase the capacity of our model (e.g. add layers or filters) and verify that we can reach the lowest achievable loss (e.g. zero). I also like to visualize in the same plot both the label and the prediction and ensure that they end up aligning perfectly once we reach the minimum loss. If they do not, there is a bug somewhere and we cannot continue to the next stage.
  
- **verify decreasing training loss**. At this stage you will hopefully be underfitting on your dataset because you’re working with a toy model. Try to increase its capacity just a bit. Did your training loss go down as it should?
  
- **visualize just before the net**. The unambiguously correct place to visualize your data is immediately before your y_hat = model(x) (or sess.run in tf). That is - you want to visualize exactly what goes into your network, decoding that raw tensor of data and labels into visualizations. This is the only “source of truth”. I can’t count the number of times this has saved me and revealed problems in data preprocessing and augmentation.
  
- **visualize prediction dynamics**. I like to visualize model predictions on a fixed test batch during the course of training. The “dynamics” of how these predictions move will give you incredibly good intuition for how the training progresses. Many times it is possible to feel the network “struggle” to fit your data if it wiggles too much in some way, revealing instabilities. Very low or very high learning rates are also easily noticeable in the amount of jitter.
  
- **use backprop to chart dependencies**. Your deep learning code will often contain complicated, vectorized, and broadcasted operations. A relatively common bug I’ve come across a few times is that people get this wrong (e.g. they use view instead of transpose/permute somewhere) and inadvertently mix information across the batch dimension. It is a depressing fact that your network will typically still train okay because it will learn to ignore data from the other examples. One way to debug this (and other related problems) is to set the loss to be something trivial like the sum of all outputs of example i, run the backward pass all the way to the input, and ensure that you get a non-zero gradient only on the i-th input. The same strategy can be used to e.g. ensure that your autoregressive model at time t only depends on 1..t-1. More generally, gradients give you information about what depends on what in your network, which can be useful for debugging.
  
- **generalize a special case**. This is a bit more of a general coding tip but I’ve often seen people create bugs when they bite off more than they can chew, writing a relatively general functionality from scratch. I like to write a very specific function to what I’m doing right now, get that to work, and then generalize it later making sure that I get the same result. Often this applies to vectorizing code, where I almost always write out the fully loopy version first and only then transform it to vectorized code one loop at a time.

**3. Overfit**
At this stage we should have a good understanding of the dataset and we have the full training + evaluation pipeline working. For any given model we can (reproducibly) compute a metric that we trust. We are also armed with our performance for an input-independent baseline, the performance of a few dumb baselines (we better beat these), and we have a rough sense of the performance of a human (we hope to reach this). The stage is now set for iterating on a good model.

The approach I like to take to finding a good model has two stages: first get a model large enough that it can overfit (i.e. focus on training loss) and then regularize it appropriately (give up some training loss to improve the validation loss). The reason I like these two stages is that if we are not able to reach a low error rate with any model at all that may again indicate some issues, bugs, or misconfiguration.

*Tips & tricks for this stage*:

- **picking the model**. To reach a good training loss you’ll want to choose an appropriate architecture for the data. When it comes to choosing this my #1 advice is: Don’t be a hero. I’ve seen a lot of people who are eager to get crazy and creative in stacking up the lego blocks of the neural net toolbox in various exotic architectures that make sense to them. Resist this temptation strongly in the early stages of your project. I always advise people to simply find the most related paper and copy paste their simplest architecture that achieves good performance. E.g. if you are classifying images don’t be a hero and just copy paste a ResNet-50 for your first run. You’re allowed to do something more custom later and beat this.
  
- **adam is safe**. In the early stages of setting baselines I like to use Adam with a learning rate of [3e-4](https://twitter.com/karpathy/status/801621764144971776?lang=en). In my experience Adam is much more forgiving to hyperparameters, including a bad learning rate. For ConvNets a well-tuned SGD will almost always slightly outperform Adam, but the optimal learning rate region is much more narrow and problem-specific. (Note: If you are using RNNs and related sequence models it is more common to use Adam. At the initial stage of your project, again, don’t be a hero and follow whatever the most related papers do.)
  
- **complexify only one at a time**. If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you’d expect. Don’t throw the kitchen sink at your model at the start. There are other ways of building up complexity - e.g. you can try to plug in smaller images first and make them bigger later, etc.

- **do not trust learning rate decay defaults**. If you are re-purposing code from some other domain always be very careful with learning rate decay. Not only would you want to use different decay schedules for different problems, but - even worse - in a typical implementation the schedule will be based current epoch number, which can vary widely simply depending on the size of your dataset. E.g. ImageNet would decay by 10 on epoch 30. If you’re not training ImageNet then you almost certainly do not want this. If you’re not careful your code could secretely be driving your learning rate to zero too early, not allowing your model to converge. In my own work I always disable learning rate decays entirely (I use a constant LR) and tune this all the way at the very end.

**4. Regularize**
Ideally, we are now at a place where we have a large model that is fitting at least the training set. Now it is time to regularize it and gain some validation accuracy by giving up some of the training accuracy. Some tips & tricks:

- **get more data**. First, the by far best and preferred way to regularize a model in any practical setting is to add more real training data. It is a very common mistake to spend a lot engineering cycles trying to squeeze juice out of a small dataset when you could instead be collecting more data. As far as I’m aware adding more data is pretty much the only guaranteed way to monotonically improve the performance of a well-configured neural network almost indefinitely. The other would be ensembles (if you can afford them), but that tops out after ~5 models.
  
- **data augment**. The next best thing to real data is half-fake data - try out more aggressive data augmentation.
  
- **creative augmentation**. If half-fake data doesn’t do it, fake data may also do something. People are finding creative ways of expanding datasets; For example, [domain randomization](https://openai.com/blog/learning-dexterity/), use of [simulation](http://vladlen.info/publications/playing-data-ground-truth-computer-games/), clever [hybrids](https://arxiv.org/abs/1708.01642) such as inserting (potentially simulated) data into scenes, or even GANs.
  
- **pretrain**. It rarely ever hurts to use a pretrained network if you can, even if you have enough data.
  
- **stick with supervised learning**. Do not get over-excited about unsupervised pretraining. Unlike what that blog post from 2008 tells you, as far as I know, no version of it has reported strong results in modern computer vision (though NLP seems to be doing pretty well with BERT and friends these days, quite likely owing to the more deliberate nature of text, and a higher signal to noise ratio).
  
- **smaller input dimensionality**. Remove features that may contain spurious signal. Any added spurious input is just another opportunity to overfit if your dataset is small. Similarly, if low-level details don’t matter much try to input a smaller image.
  
- **smaller model size**. In many cases you can use domain knowledge constraints on the network to decrease its size. As an example, it used to be trendy to use Fully Connected layers at the top of backbones for ImageNet but these have since been replaced with simple average pooling, eliminating a ton of parameters in the process.
  
- **decrease the batch size**. Due to the normalization inside batch norm smaller batch sizes somewhat correspond to stronger regularization. This is because the batch empirical mean/std are more approximate versions of the full mean/std so the scale & offset “wiggles” your batch around more.
  
- **drop**. Add dropout. Use dropout2d (spatial dropout) for ConvNets. Use this sparingly/carefully because dropout [does not seem to play nice](https://arxiv.org/abs/1801.05134) with batch normalization.

- **weight decay**. Increase the weight decay penalty.
  
- **early stopping**. Stop training based on your measured validation loss to catch your model just as it’s about to overfit.
- **try a larger model**. I mention this last and only after early stopping but I’ve found a few times in the past that larger models will of course overfit much more eventually, but their “early stopped” performance can often be much better than that of smaller models.

Finally, to gain additional confidence that your network is a reasonable classifier, I like to visualize the network’s first-layer weights and ensure you get nice edges that make sense. If your first layer filters look like noise then something could be off. Similarly, activations inside the net can sometimes display odd artifacts and hint at problems.

**5. Tune**
You should now be “in the loop” with your dataset exploring a wide model space for architectures that achieve low validation loss. A few tips and tricks for this step:

- **random over grid search**. For simultaneously tuning multiple hyperparameters it may sound tempting to use grid search to ensure coverage of all settings, but keep in mind that it is [best to use random search instead](https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf). Intuitively, this is because neural nets are often much more sensitive to some parameters than others. In the limit, if a parameter $\alpha$ matters but changing $\beta$ has no effect then you’d rather sample a more throughly than at a few fixed points multiple times.
  
- **hyper-parameter optimization**. There is a large number of fancy bayesian hyper-parameter optimization toolboxes around and a few of my friends have also reported success with them, but my personal experience is that the state of the art approach to exploring a nice and wide space of models and hyperparameters is to use an intern :) Just kidding.


**6. Squeeze out the juice**
Once you find the best types of architectures and hyper-parameters you can still use a few more tricks to squeeze out the last pieces of juice out of the system:

- **ensembles**. Model ensembles are a pretty much guaranteed way to gain 2% of accuracy on anything. If you can’t afford the computation at test time look into distilling your ensemble into a network using [dark knowledge](https://arxiv.org/abs/1503.02531).
- **leave it training**. I’ve often seen people tempted to stop the model training when the validation loss seems to be leveling off. In my experience networks keep training for unintuitively long time. One time I accidentally left a model training during the winter break and when I got back in January it was SOTA (“state of the art”).
**Conclusion**
Once you make it here you’ll have all the ingredients for success: you have a deep understanding of the technology, the dataset and the problem, you’ve set up the entire training/evaluation infrastructure and achieved high confidence in its accuracy, and you’ve explored increasingly more complex models, gaining performance improvements in ways you’ve predicted each step of the way. You’re now ready to read a lot of papers, try a large number of experiments, and get your SOTA results. Good luck!



#### Simple training loop
```python

import torch
import torchvision

tfms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
    # something 1 
    # something 2
    # something X
])

data = torchvision.datasets.ImageFolder(root='YOUR_DATA', transform=tfms)


model = MegaTurboModel()

N_EPOCHS = 1000
LEARNING_RATE = 3e-4

loss_function = torch.nn.SuperDuperHexagonLoss()
optimizer = torch.optim.UltraMegaOptimizer(model.parameters(), lr=LEARNING_RATE)

for i in range(N_EPOCHS):
    for j, (img, label) in enumerate(data):
        
        # Forward
        output = model(img)
        loss = loss_function(output, label)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
