import numpy as np
from keras import backend as b
from keras.layers import Dense, Input, Lambda, Multiply, Add
from keras.regularizers import l2
from keras.optimizers import Adam
from keras import Model
from scipy.spatial import distance_matrix
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class PENN:
    def __init__(self, k, n, mc_draws=100, size=10, l2_penalty=0.001):
        """
        The __init__ function initializes the neural network
        :param k: the number of features
        :param n: the number of observations
        :param mc_draws: the number of Monte Carlo draws for the variational layer
        :param size: the number of nodes in each of the hidden layer
        :param l2_penalty: kernel regularization in the inference network
        """
        self.model = self.build(k, n, mc_draws, size, l2_penalty)
        self.mu_model = Model(self.model.inputs, self.model.get_layer('mu').output)
        self.sigma_model = Model(self.model.inputs, self.model.get_layer('sigma').output)

    def build(self, k, n, mc_draws=100, size=10, l2_penalty=0.001):
        self.k = k
        self.n = n
        self.mc_draws = mc_draws
        self.size = size
        self.l2_penalty = l2_penalty

        # Model inputs
        self.input_inference_nn = Input(self.k, name='input_inference_nn')
        self.input_model = Input(self.k, name='input_model')
        self.input_knn_prior = Input(batch_shape=(self.n, self.n), name='input_knn_prior')
        self.input_mc = Input(tensor=b.random_normal((self.n, self.mc_draws, self.k)), name='input_mc')
        self.inputs = [self.input_inference_nn,
                       self.input_model,
                       self.input_knn_prior,
                       self.input_mc]

        # Hidden layers in inference network
        self.encoder_layer_1 = Dense(self.size,
                                     activation='sigmoid',
                                     kernel_regularizer=l2(self.l2_penalty))(self.input_inference_nn)
        self.encoder_layer_2 = Dense(self.size,
                                     activation='sigmoid',
                                     kernel_regularizer=l2(self.l2_penalty))(self.encoder_layer_1)

        # Parameters layers for mu and sigma
        self.mu = Dense(self.k, kernel_regularizer=l2(self.l2_penalty), name='mu')(self.encoder_layer_2)
        self.sigma_squared = Dense(self.k, activation='exponential', kernel_regularizer=l2(self.l2_penalty))(self.encoder_layer_2)
        self.sigma = Lambda(lambda i: b.sqrt(i), name='sigma')(self.sigma_squared)

        # Variational layer generates a sample from the parameter posterior
        self.sample = Multiply()([self.sigma, self.input_mc])
        self.sample = Add()([self.sample, self.mu])

        # Generate predictions
        self.output = Multiply()([self.sample, self.input_model])
        self.output = Lambda(lambda i: b.sum(i, axis=2, keepdims=True), output_shape=(n, mc_draws, 1))(self.output)

        # Build overall and parameter-specific models
        model = Model(self.inputs, self.output)

        return model

    def compile(self, lam, gam):
        self.lam = lam
        self.gam = gam

        def loss(y, y_pred):
            mse = b.mean(b.square(y_pred - y))
            mu_ = self.model.get_layer('mu').output
            sigma_ = self.model.get_layer('sigma').output
            input_knn_prior_ = self.model.inputs[2]
            prior_mu = b.dot(input_knn_prior_, mu_)
            prior_sigma = b.dot(input_knn_prior_, sigma_) + b.dot(input_knn_prior_, b.square(mu_ - prior_mu))

            kl = b.mean(b.mean((b.log(b.sqrt(sigma_)) -
                                b.log(b.sqrt(prior_sigma))) -
                 ((sigma_ + b.square(mu_ - prior_mu)) / (2 * prior_sigma)) + 0.5, axis=1))

            return mse - kl * self.lam

        self.model.compile(loss=loss, optimizer=Adam(learning_rate=0.05, clipnorm=1, clipvalue=0.5))

    def fit(self, x, y, epochs=1000):
        knn_prior = distance_matrix(x, x)
        gam = knn_prior[knn_prior>0.0].min() + self.gam * (
            knn_prior[knn_prior > 0.0].max() - knn_prior[knn_prior>0.0].min()
        )
        knn_prior /= gam
        idx = knn_prior < 1.0
        knn_prior[idx] = 1.0
        knn_prior[~idx] = 0.0
        knn_prior = (knn_prior.T / knn_prior.sum(axis=1)).T

        self.data = {
            'input_inference_nn': x,
            'input_model': x,
            'input_knn_prior': knn_prior,
            'input_mc': np.zeros((self.n, self.mc_draws, self.k))
        }

        y = np.repeat(y[:, np.newaxis, np.newaxis], self.mc_draws, axis=1)

        self.model.fit(self.data, y, batch_size=self.n, epochs=epochs, verbose=0)

    def get_mu(self):
        return self.mu_model.predict(self.data, batch_size=self.n)

