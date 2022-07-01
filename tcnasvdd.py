
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Concatenate,
    Flatten,
    Reshape,
    TimeDistributed,
    Masking,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tcn import TCN
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K
from sklearn.metrics import roc_auc_score


class SphereTCN:
    """
    ############################################################
    # TCN-Autoencoding Support Vector Machine Data Description #
    ############################################################
    
    """
    def __init__(
        self,
        input_dim=468,
        dilations=[1, 2, 8],
        nb_filters=20,
        learning_rate=0.001,
        c_learning_rate=1,
        kernel_size=3,
        latent_dim=10,
        init_c="normal",
        patience_roc=20,
        T=2,
        z_activation=None,
        use_batch_norm=False,
        use_layer_norm=False,
        use_weight_norm=False,
        dropout_rate=0,
        epochs=100,
        batch_size=32,
        verbose=1,
        random_state=None,
    ):
        """
        Args:
            input_dim (int, optional): [Dimension of the input data]. Defaults to 468.
            dilations (list, optional): [Number of dilations to be used in TCN blocks]. Defaults to [1, 2, 8].
            nb_filters (int, optional): [Number of filters used in TCN blocks]. Defaults to 20.
            learning_rate (float, optional): [Learning rate for the network]. Defaults to 0.001.
            c_learning_rate (int, optional): [Learning rate for the hypersphere center]. Defaults to 1.
            kernel_size (int, optional): [Number of channels to use in TCN blocks]. Defaults to 3.
            latent_dim (int, optional): [Latent dimension of the model]. Defaults to 10.
            init_c (str, optional): [Type of initialization for `c`, either "normal" or "uniform"]. Defaults to "normal".
            patience_roc (int, optional): [How many epochs should you wait untill the ROC on the validation set increases]. Defaults to 20.
            T (int, optional): [Number of iterations to tune `gamma`, the parameter that balances the anaomaly score and the loss]. Defaults to 2.
            z_activation ([type], optional): [Type of activation to use in the latent dimension]. Defaults to None.
            use_batch_norm (bool, optional): [description]. Defaults to False.
            use_layer_norm (bool, optional): [description]. Defaults to False.
            use_weight_norm (bool, optional): [description]. Defaults to False.
            dropout_rate (int, optional): [description]. Defaults to 0.
            epochs (int, optional): [Number of epochs to train the model]. Defaults to 100.
            batch_size (int, optional): [Size of batches seen during a single epoch]. Defaults to 32.
            verbose (int, optional): Defaults to 1.
            random_state ([type], optional): [Seed, for determinism and parameter initialization]. Defaults to None.
        """
        self.input_dim = input_dim
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.patience_roc = patience_roc
        self.T = T
        self.z_activation = z_activation
        self._init_c = init_c
        self.learning_rate = learning_rate
        self.c_learning_rate = c_learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state
        self._set_seed()
        self._reset_graph()
        self._build_model()
        self.best_roc = 0

        self._set_seed()
        self.free_tf_memory()
        self._reset_graph()
        self._build_model()

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.tcn_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        # self.c_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.c_learning_rate , decay_rate=0.1, decay_steps=self.epochs, staircase=True, name=None)
        self.c_optimizer = tf.keras.optimizers.Adagrad(
            learning_rate=self.c_learning_rate
        )

    def _reset_graph(self):
        tf.keras.backend.clear_session()

    def free_tf_memory(self):
        tf.keras.backend.clear_session()

    def _set_seed(self):
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)

    def _build_model(self, set=False):
        """Builds the TCN-Autoencoding Support Vector Machine model.
        Args:
            set (bool, optional): Use when temporarily storing weights during training. Defaults to False.
        """
        self._reset_graph()
        self._set_seed()

        inputs = Input(shape=(self.input_dim, 1), name="input")
        mask = Masking(mask_value=0)(inputs)
        x = TCN(
            dilations=self.dilations,
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=1,
            padding="causal",
            use_skip_connections=True,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_weight_norm=self.use_layer_norm,
            return_sequences=True,
        )(mask)
        x = Flatten()(x)

        x_code = Dense(
            units=self.latent_dim, activation=self.z_activation, name="net_output"
        )(x)

        x = Dense(self.input_dim)(x_code)
        x = Reshape((self.input_dim, 1))(x)
        x = TCN(
            dilations=self.dilations[::-1],
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=1,
            padding="causal",
            use_skip_connections=True,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_weight_norm=self.use_layer_norm,
            return_sequences=True,
        )(x)
        x = Masking(mask_value=0)(x)
        output_ae = TimeDistributed(Dense(1))(x)

        if set == False:
            self.tcnae = Model(
                inputs=inputs,
                outputs=[
                    K.cast(output_ae, dtype=tf.float32),
                    K.cast(x_code, dtype=tf.float32),
                ],
                name="tcnae",
            )

        if set == True:
            self.tmp = Model(
                inputs=inputs,
                outputs=[
                    K.cast(output_ae, dtype=tf.float32),
                    K.cast(x_code, dtype=tf.float32),
                ],
                name="tcnae",
            )

        if self.verbose >= 1 and set:
            print(self.tcnae.summary())

    def _tune_gamma(self, dataloader):
        """Tunes the gamma parameter of the loss function."""
        print("Tuning gamma")
        gamma = float(0)
        for k in range(self.T):
            R = float(0)
            RE = float(0)
            pbar_b = Progbar(target=len(dataloader))
            for step, (x_batch, y_batch) in enumerate(dataloader):
                outputs, code = self.tcnae(x_batch, training=False)
                R = K.cast(R, dtype=tf.float32)
                RE = K.cast(RE, dtype=tf.float32)
                R += tf.math.reduce_sum(code) ** 2
                L = self.loss_fn(outputs, y_batch)
                RE += K.cast(L, dtype=tf.float32)
                pbar_b.update(step)
            R / len(dataloader)
            RE / len(dataloader)
            R = K.cast(R, dtype=tf.float32)
            # pbar.update(step, values=[("R", R), ("RE", RE), ("gamma", gamma)])
            gamma += RE / R
        gamma = gamma / self.T
        self.gamma = tf.constant(gamma.numpy())

    def _init_c__(self):
        """Initializes the hypersphere center C
        """
        if self._init_c == "normal":
            self.C = tf.Variable(
                tf.random.normal(shape=(self.latent_dim,), dtype=tf.float32),
                trainable=True,
            )
        elif self._init_c == "uniform":
            self.C = tf.Variable(
                tf.random.uniform(shape=(self.latent_dim,), dtype=tf.float32),
                trainable=True,
            )

    def _set_dataloaders(self, X, X_val=None):
        """Sets the dataloaders for training and validation.

        Args:
            X (nd.array): Training set
            X_val (nd.array, optional): Validation set (contains anomalies). Defaults to None.
        """
        self.train_data = tf.data.Dataset.from_tensor_slices((X, X)).batch(
            self.batch_size
        )
        if X_val is not None:
            self.val_data = tf.data.Dataset.from_tensor_slices((X_val, X_val)).batch(
                self.batch_size
            )
        else:
            self.val_data = None

    @tf.function
    def train_ae(self, x_batch, target_batch, K_):
        """Trains the autoencoder.
        Args:
            x_batch : shape (batch_size, input_dim, 1)
            target_batch : shape (batch_size, input_dim, 1)
            K_ (float): How much of the batch to use for training the auto-encoder.

        Returns:
            _type_: _description_
        """
        with tf.GradientTape() as tape:
            outputs, code = self.tcnae(x_batch[: int(len(x_batch) * K_)], training=True)
            R = tf.math.reduce_sum((code - self.C) ** 2)
            # R = tf.keras.backend.cast(R, tf.float64)
            LOSS_tmp = K.cast(
                self.loss_fn(outputs, target_batch[: int(len(x_batch) * K_)]),
                dtype=tf.float32,
            )
            train_loss = LOSS_tmp + self.gamma * R
        grads = tape.gradient(train_loss, self.tcnae.trainable_variables)
        self.tcn_optimizer.apply_gradients(zip(grads, self.tcnae.trainable_variables))
        return train_loss, LOSS_tmp, R, outputs

    @tf.function
    def train_svdd(self, x_batch, K_):
        """Trains the SVDD. i.e the hypersphere center C.
        Args:
            x_batch (_type_): shape (batch_size, input_dim, 1)
            K_: How much of the batch to use for training the SVDD. (Will use 1 - K_)
        """
        with tf.GradientTape() as tape:
            _, code = self.tcnae(x_batch[int(len(x_batch) * K_) :], training=False)
            center = tf.math.reduce_mean(code)
            center_loss = self.loss_fn(self.C, center)
        grad = tape.gradient(center_loss, self.C)
        self.c_optimizer.apply_gradients(zip([grad], [self.C]))

    def fit(self, X, K_, X_val=None, Y_val=None):
        self._set_dataloaders(X, X_val)
        self._tune_gamma(self.train_data)
        print(f"GAMMA tuned : {self.gamma}")
        self._init_c__()
        self.L1 = np.zeros(self.epochs)
        self.L2 = np.zeros(self.epochs)
        self.L3 = np.zeros(self.epochs)
        self.L1_v = np.zeros(self.epochs)
        self.L2_v = np.zeros(self.epochs)
        self.L3_v = np.zeros(self.epochs)
        self.ROC = np.zeros(self.epochs)

        loss = 0
        aeloss = 0
        svdd_loss = 0

        wait_roc = 0

        pbar_e = Progbar(target=self.epochs)
        for epoch in range(self.epochs):
            loss = 0
            aeloss = 0
            svddloss = 0

            _loss = 0
            val_aeloss = 0
            val_svddloss = 0

            print(f"Epoch : {epoch}")
            pbar_e = Progbar(
                target=len(self.train_data),
                stateful_metrics=[
                    "loss",
                    "aeloss",
                    "svddloss",
                    "val_loss",
                    "val_aeloss",
                    "val_svddloss",
                ],
            )

            # pbar_b = Progbar(target=len(self.train_data))
            for step, (x_batch, target_batch) in enumerate(self.train_data):

                train_loss, aeloss, R, outputs = self.train_ae(
                    x_batch, target_batch, K_
                )
                loss += train_loss
                # aeloss += self.loss_fn(outputs[:int(len(x_batch) * K_)], target_batch[:int(len(x_batch) * K_)])
                svddloss += R
                self.train_svdd(x_batch, K_)
                # pbar_b.update(step, values=[("loss", loss), ("aeloss", aeloss), ("svddloss", svddloss)])

            if self.val_data is not None:
                pbar_v = Progbar(target=len(self.val_data))
                for step, (x_batch, target_batch) in enumerate(self.val_data):
                    outputs, x_code = self.tcnae(x_batch, training=False)
                    R = tf.math.reduce_sum((x_code - self.C) ** 2)
                    # R = tf.keras.backend.cast(R, tf.float64)
                    AE_val = K.cast(
                        self.loss_fn(outputs, target_batch), dtype=tf.float32
                    )
                    val_loss = AE_val + self.gamma * R
                    _loss += val_loss
                    val_aeloss += self.loss_fn(outputs, target_batch)
                    val_svddloss += R
                    pbar_v.update(step + 1, finalize=False)

                _loss /= len(self.val_data)
                val_aeloss /= len(self.val_data)
                val_svddloss /= len(self.val_data)

                val_scores = self.predict(X_val)
                current_roc = roc_auc_score(y_true=Y_val, y_score=val_scores)

                loss /= len(self.train_data)
                aeloss /= len(self.train_data)
                svddloss /= len(self.train_data)
            else:
                current_roc = 0
            self.L1[epoch] = loss
            self.L2[epoch] = aeloss
            self.L3[epoch] = svddloss
            self.L1_v[epoch] = _loss
            self.L2_v[epoch] = val_aeloss
            self.L3_v[epoch] = val_svddloss
            self.ROC[epoch] = current_roc

            pbar_e.update(
                epoch + 1,
                values=[
                    ("loss", loss),
                    ("aeloss", aeloss),
                    ("svddloss", svddloss),
                    ("val_loss", _loss),
                    ("val_aeloss", val_aeloss),
                    ("val_svddloss", val_svddloss),
                    ("ROC", current_roc),
                ],
            )
            wait_roc += 1
            if current_roc > self.best_roc:
                print("New best ROC found!")
                self.best_roc = current_roc
                wait_roc = 0
                best_roc_weights = self.tcnae.get_weights()
                best_C_roc = self.C.numpy()

            if wait_roc >= self.patience_roc:
                print("Early stopping")
                self._build_model(set=True)
                self.tmp.set_weights(best_roc_weights)
                self.MODEL = [self.tmp, best_C_roc]
                return self.MODEL

        if epoch == (self.epochs - 1):
            if self.val_data is not None:
                self._build_model(set=True)
                self.tmp.set_weights(best_roc_weights)
                self.MODEL = [self.tmp, best_C_roc]
                return self.MODEL
            else:
                return [self.tcnae, self.gamma, self.C]

    def predict(self, X=None):
        """Predicts the scores for the given data.

        Args:
            X (nd.array optional):  Defaults to None.

        Returns:
            scores: Anoamly scores according to loss function
        """
        x_val_hat, code = self.tcnae(X, training=False)
        scores = np.array((((x_val_hat - X)) ** 2)).mean(
            axis=1
        ).flatten() + self.gamma * np.sum(((code - self.C) ** 2), axis=1)
        return scores

    @staticmethod
    def predict_bis(model, X, gamma, C):
        """Utils function to predict the scores for a learned model.
        Args:
            model (tensorflow.keras.models): _description_
            X (_type_): _description_
            gamma (_type_): _description_
            C (_type_): _description_

        Returns:
            _type_: _description_
        """
        x_hat, code = model.predict(X)
        scores = np.array((((x_hat - X)) ** 2)).mean(axis=1).flatten() + gamma * np.sum(
            ((code - C) ** 2), axis=1
        )
        return scores
