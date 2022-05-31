README
================

A **Parameter Encoder Neural Network (PENN)** is an explainable machine
learning technique that solves two problems associated with traditional
XAI algorithms:

1.  It permits the calculation of local parameter distributions.
    Parameter distributions are often more interesting than feature
    contributions — particularly in economic and financial applications
    — since the parameters disentangle the effect from the observation
    (the contribution can roughly be defined as the product of effect
    and observation).
2.  It solves a problem of biased contributions that is inherent to many
    traditional XAI algorithms (I will write a dedicated article
    exploring this concept in the near future). Particularly in the
    setting where neural networks are powerful — in interactive,
    dependent processes — traditional XAI can be biased, by attributing
    effect to each feature independently.

At the end of the tutorial, I will have estimated the following highly
nonlinear parameter functions for a simulated regression with three
variables:

## Preliminaries

For more details on the mathematical background of the parameter encoder
neural network, see the [paper](https://arxiv.org/abs/2106.05536).

## Example data

We will use a nonlinear simulated data set with `k = 3` feature in `x`
and a continuous target `y`, where `beta[1]` has the shape of a
sine-curve, `beta[2]` has no effect on the output (i.e. this is simply a
correlated nuisance term), and `beta[3]` has a threshold shape with 3
different regimes:

``` python
import numpy as np

# For reproducibility
np.random.seed(42)

k = 3
n = 1000

Sigma = [[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]]
x = np.random.multivariate_normal([0.0]*k, Sigma, n)
eps = np.random.normal(size=n)
betas = np.zeros((n,k))
betas[:,0] = np.sin(x[:,0]) * 5.0
betas[x[:,2]>0.75,2] = 5.0
betas[x[:,2]<-0.75,2] = -5.0
y = (x * betas).sum(axis=1) + eps
```

## Building a Parameter Encoder NN from scratch

The following code chunks construct a PENN model using `keras`. The
separate functions are combined into a `PENN` class in `PENN.py`. We
begin by loading the necessary modules:

``` python
# Load backend functions
from keras import backend as b

from keras.layers import Dense, Input, Lambda, Multiply, Add
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import Model

from scipy.spatial import distance_matrix

import tensorflow as tf
# For the loss function to work, we need to switch off eager execution
tf.compat.v1.disable_eager_execution()
# For reproducibility
tf.random.set_seed(42)
```

Next we construct the inference network with 2 layers and 10 hidden
nodes in each layer. We use a sigmoid activation function. The inference
network is completed by the output nodes for the parameter
distributions, `mu` and `sigma`:

``` python
def build(k, n, mc_draws=100, size=10, l2_penalty=0.001):
    k = k
    n = n
    mc_draws = mc_draws
    size = size
    l2_penalty = l2_penalty

    # Model inputs
    input_inference_nn = Input(k, name='input_inference_nn')
    input_model = Input(k, name='input_model')
    input_knn_prior = Input(batch_shape=(n, n), name='input_knn_prior')
    input_mc = Input(tensor=b.random_normal((n, mc_draws, k)), name='input_mc')
    inputs = [input_inference_nn,
                   input_model,
                   input_knn_prior,
                   input_mc]

    # Hidden layers in inference network
    encoder_layer_1 = Dense(size,
                                 activation='sigmoid',
                                 kernel_regularizer=l2(l2_penalty))(input_inference_nn)
    encoder_layer_2 = Dense(size,
                                 activation='sigmoid',
                                 kernel_regularizer=l2(l2_penalty))(encoder_layer_1)

    # Parameters layers for mu and sigma
    mu = Dense(k, kernel_regularizer=l2(l2_penalty), name='mu')(encoder_layer_2)
    sigma_squared = Dense(k, activation='exponential', 
                          kernel_regularizer=l2(l2_penalty))(encoder_layer_2)
    sigma = Lambda(lambda i: b.sqrt(i), name='sigma')(sigma_squared)

    # Variational layer generates a sample from the parameter posterior
    sample = Multiply()([sigma, input_mc])
    sample = Add()([sample, mu])

    # Generate predictions
    output = Multiply()([sample, input_model])
    output = Lambda(lambda i: b.sum(i, axis=2, keepdims=True), output_shape=(n, mc_draws, 1))(output)

    # Build overall and parameter-specific models
    model = Model(inputs, output)

    return model
```

``` python
model = build(k, n)

mu_model = Model(model.inputs, model.get_layer('mu').output)
sigma_model = Model(model.inputs, model.get_layer('sigma').output)
```

``` python
lam = 4
gam = 0.04

def loss(y, y_pred):
    mse = b.mean(b.square(y_pred - y))
    mu_ = model.get_layer('mu').output
    sigma_ = model.get_layer('sigma').output
    input_knn_prior_ = model.inputs[2]
    prior_mu = b.dot(input_knn_prior_, mu_)
    prior_sigma = b.dot(input_knn_prior_, sigma_) + b.dot(input_knn_prior_, b.square(mu_ - prior_mu))

    kl = b.mean(b.mean((b.log(b.sqrt(sigma_)) -
                        b.log(b.sqrt(prior_sigma))) -
         ((sigma_ + b.square(mu_ - prior_mu)) / (2 * prior_sigma)) + 0.5, axis=1))

    return mse - kl * lam

model.compile(loss=loss, optimizer=Adam(learning_rate=0.05, clipnorm=1, clipvalue=0.5))
```

``` python
knn_prior = distance_matrix(x, x)
gam = knn_prior[knn_prior>0.0].min() + gam * (
    knn_prior[knn_prior > 0.0].max() - knn_prior[knn_prior>0.0].min()
)
knn_prior /= gam
idx = knn_prior < 1.0
knn_prior[idx] = 1.0
knn_prior[~idx] = 0.0
knn_prior = (knn_prior.T / knn_prior.sum(axis=1)).T

data = {
    'input_inference_nn': x,
    'input_model': x,
    'input_knn_prior': knn_prior,
    'input_mc': np.zeros((n, 100, k))
}

y = np.repeat(y[:, np.newaxis, np.newaxis], 100, axis=1)

model.fit(data, y, batch_size=n, epochs=1000, verbose=0)
```

    ## <keras.callbacks.History object at 0x7f5256922e80>

``` python
mu = mu_model.predict(data, batch_size=n)
```

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->
