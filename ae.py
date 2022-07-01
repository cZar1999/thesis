from tensorflow.keras.layers import (
    Flatten,
    Reshape,
    Dense,
    Input,
    TimeDistributed,
    Lambda,
    Masking,
    RepeatVector,
    Dropout,
)
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.utils import Progbar
import tensorflow.keras.backend as K


class AutoEncoder:
    def __init__(
        self,
        patience_roc=100,
        input_dim=100,
        hidden_neurons=None,
        hidden_activation="relu",
        output_activation="sigmoid",
        n_epochs=100,
        batch_size=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        l2_regularizer=0.1,
        verbose=1,
        random_state=None,
    ):
        self.input_dim = input_dim
        self.patience_roc = patience_roc
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.verbose = verbose
        self.random_state = random_state
        self._set_seed()
        self.free_tf_memory()
        self._build_model()

    def _build_model(self, set=False):
        inp = Input(shape=(self.input_dim,))
        x = Dense(self.hidden_neurons[0], activity_regularizer=l2(self.l2_regularizer))(
            inp
        )
        x = Dropout(self.dropout_rate)(x)
        for unit in self.hidden_neurons[1:-1]:
            x = Dense(unit, activity_regularizer=l2(self.l2_regularizer))(x)
            x = Dropout(self.dropout_rate)(x)
        code = Dense(
            self.hidden_neurons[-1],
            input_shape=(self.input_dim,),
            activity_regularizer=l2(self.l2_regularizer),
            activation=self.output_activation,
            name="code",
        )(x)

        x = Dense(
            units=self.hidden_neurons[::-1][1],
            activity_regularizer=l2(self.l2_regularizer),
        )(code)
        x = Dropout(self.dropout_rate)(x)

        for unit in self.hidden_neurons[::-1][2:]:
            x = Dense(unit, activity_regularizer=l2(self.l2_regularizer))(x)
            x = Dropout(self.dropout_rate)(x)
        

        x = Dense(units=self.input_dim, activation="sigmoid")(x)
        # x = tf.expand_dims(x, axis=2)
        if set == False:
            self.model = Model(inputs=inp, outputs=[x, code])
        else:
            self.tmp = Model(inputs=inp, outputs=[x, code])

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
    def _train_step(self, x_batch, target_batch):
        with tf.GradientTape() as tape:
            output, _ = self.model(x_batch, training=True)
            # import pdb; pdb.set_trace()
            loss = self.loss_fn(output, target_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, X, X_val, Y_val, plot=False, **fit_params):
        # self.compile()
        self._set_dataloaders(X=X, X_val=X_val)
        self.loss_hist = np.zeros(self.n_epochs)
        self.loss_hist_v = np.zeros(self.n_epochs)
        self.ROC = np.zeros(self.n_epochs)

        wait_roc = 0
        best_roc= 0
        for epoch in range(self.n_epochs):
            loss = 0
            val_loss = 0
            pbar_e = Progbar(
                target=len(self.train_data),
                stateful_metrics=[
                    "loss",
                    "val_loss",
                    "val_roc",
                ],
            )
            pbar_b = Progbar(target=len(self.train_data))
            for step, (x_batch, target_batch) in enumerate(self.train_data):
                loss = self._train_step(x_batch, target_batch)
                loss = loss
                pbar_b.update(step, values=[("loss", loss)], finalize=False)

            if self.val_data is not None:
                pbar_v = Progbar(target=len(self.val_data))
                for step, (x_batch, y_batch) in enumerate(self.val_data):
                    output, _ = self.model(x_batch, training=False)
                    val_loss = self.loss_fn(output, x_batch)
                    pbar_v.update(step +1, finalize=False)

                recon = self.model.predict(X_val)[0]
                val_scores = AutoEncoder.get_scores(X_val, recon)
                current_roc = roc_auc_score(y_true=Y_val, y_score=val_scores)

                print(f"ROC!: {current_roc}")
                val_loss /= len(self.val_data)
                self.ROC[epoch] = current_roc
                self.loss_hist_v[epoch] = val_loss

            loss /= len(self.train_data)
            val_loss /= len(self.val_data)
            self.loss_hist[epoch] = loss
            pbar_e.update(epoch + 1, values=[("loss", loss), ("val_loss", val_loss), ("val_roc", current_roc)], finalize=True)

            if current_roc > best_roc:
                best_roc = current_roc
                best_roc_weights = self.model.get_weights()
                wait_roc = 0
            
            if wait_roc > self.patience_roc:
                self._build_model(set=True)
                self.tmp.set_weights(best_roc_weights)
                print("FOUND ROC")
                return self.tmp

            print(f"Wait roc: {wait_roc}")
            wait_roc +=1

    @staticmethod
    def get_scores(X_true, X_recon):
        return tf.keras.metrics.mean_squared_error(X_true.squeeze(), X_recon.squeeze())

    def _set_seed(self):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    def free_tf_memory(self):
        tf.keras.backend.clear_session()
