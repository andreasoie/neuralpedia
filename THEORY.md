## Theory of Neural Networks

Brief interpretations of various subjects in the space of Neural Networks

## Basics

Below shows various questions I've though of as I'm trying to formulate my theorical understanding of the subjects.

### Fundamentals

> What is it?

> How do we use them?

> How do we create them?

> Whats the consequence (from perspective X) ?

### Layers

#### What is a input layers, and what do we place here?

#### What is a hidden layer, and what types do we have?

#### What is a output layer, and how do we design this to meet our goal?

#### Whats the deal with linear layers?

#### Whats the deal with convolutional layers?

#### What happens when we use multiple-hidden layers?

#### Why do we sometimes drop certain layers?

#### What the deal with fully-connected layers?

#### Whats the deal with things between linear layers?

#### Sometimes linear layers have added bias, why is that?

#### Whats the deal with activation layers?

#### Does the combinations of linear operations and activation layers matter?

#### There seems to be a certain combation of layers, most frequently used - why is that?

### Transformations & Augmentation (move?)

#### Why do we often need to normalize our data?

#### Some data may have a wide-spread of values, how do we cope with this?

#### Explain the following terms with examples

##### Batch normalization

#####

### "Pass" functions

#### Whats a forward pass, and what is imporantant to think of when designing this function?

#### Whats a backward pass (backprop), and what is imporantant to think of when designing this function?

### Activation functions

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

> http://karpathy.github.io/2019/04/25/recipe/

#### Summarized

1.
2.
3.
4. X.

#### What do we need inside the training loop?

Controversial recommendations e.g. early stopping? small batch sizes?
