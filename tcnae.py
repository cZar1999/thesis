from tcn import TCN
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Flatten,
    Reshape,
    Dense,
    Input,
    TimeDistributed,
    Lambda,
    Masking,
    RepeatVector,
)
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping
import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score


class TCNAE:
    def __init__(
        self,
        input_dim=468,
        patience_roc=10,
        dilations=[1, 2, 6, 10],
        nb_filters=20,
        kernel_size=20,
        latent_dim=10,
        nb_stacks=1,
        padding="causal",
        use_skip_connections=True,
        patience=20,
        n_epochs=100,
        batch_size=128,
        learning_rate=0.001,
        use_batch_norm=False,
        use_layer_norm=False,
        use_weight_norm=False,
        activation="relu",
        dropout_rate=0,
        dropout_dense=0,
        random_state=42,
    ):
        self.input_dim = input_dim
        self.patience_roc = patience_roc
        self.dilations = dilations
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.use_skip_connections = use_skip_connections
        self.patience = patience
        self.early_stopping = EarlyStopping(
            patience=self.patience, monitor="val_loss", restore_best_weights=False
        )
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.callbacks = [self.early_stopping, TerminateOnNaN()]
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dropout_dense = dropout_dense
        self.random_state = random_state
        self.build_model()

    def _reset_graph(self):
        K.clear_session()

    def _set_seed(self):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    @staticmethod
    def _crop_output(x):
        padding = K.cast(K.not_equal(x[1], 0), dtype=K.floatx())
        return x[0] * padding

    def _build_encoder(self, x):
        x = TCN(
            dilations=self.dilations,
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=self.nb_stacks,
            padding=self.padding,
            use_skip_connections=self.use_skip_connections,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_weight_norm=self.use_weight_norm,
            return_sequences=True,
        )(x)
        x = Flatten()(x)
        x = Dense(units=self.latent_dim, activation=self.activation)(x)
        return x

    def _build_decoder(self, x):
        x = Dense(self.input_dim)(x)
        x = Reshape((self.input_dim, 1))(x)
        x = TCN(
            dilations=self.dilations[::-1],
            nb_filters=self.nb_filters,
            kernel_size=self.kernel_size,
            nb_stacks=self.nb_stacks,
            padding=self.padding,
            use_skip_connections=self.use_skip_connections,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            use_weight_norm=self.use_weight_norm,
            return_sequences=True,
        )(x)
        x = TimeDistributed(Dense(1))(x)
        return x

    def build_model(self, set=False):
        self._reset_graph()
        self._set_seed()

        self.inputs_ = Input(shape=(self.input_dim, 1))
        mask = Masking(mask_value=0.0, input_shape=(self.input_dim, 1))(self.inputs_)
        coded = self._build_encoder(mask)
        decoded = self._build_decoder(coded)
        output = Masking(mask_value=0.0, input_shape=(self.input_dim, 1))(decoded)
        output = Lambda(TCNAE._crop_output, output_shape=(self.input_dim, 1))(
            [output, self.inputs_]
        )
        if not set: 
          self.model = Model(inputs=self.inputs_, outputs=[output, coded])
        else:
          self.tmp = Model(inputs=self.inputs_, outputs=[output, coded])

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )
    
    def _set_dataloaders(self, X, X_val=None):
        """Sets the dataloaders for training and validation.

        Args:
            X (nd.array): Training set
            X_val (nd.array, optional): Validation set (contains anomalies). Defaults to None.
        """
        self.train_data = tf.data.Dataset.from_tensor_slices(X).batch(
            self.batch_size
        )
        if X_val is not None:
            self.val_data = tf.data.Dataset.from_tensor_slices(X_val).batch(
                self.batch_size
            )
        else:
            self.val_data = None


    @tf.function
    def _train_step(self, x_batch, target_batch):
        with tf.GradientTape() as tape:
            output, _ = self.model(x_batch, training=True)
            loss = self.loss_fn(output, target_batch)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, output

    def fit(self, X, X_val, Y_val, plot=False, **fit_params):
        # self.compile()
        self._set_dataloaders(X=X, X_val=X_val)
        self.loss_hist = np.zeros(self.n_epochs)
        self.loss_hist_v = np.zeros(self.n_epochs)
        self.ROC = np.zeros(self.n_epochs)

        wait_roc = 0
        best_roc = 0
        for epoch in range(self.n_epochs):
            loss = 0
            val_loss = 0
            pbar_e = Progbar(
                    target=len(self.train_data),
                    stateful_metrics=[
                        "loss",
                        "val_loss",
                        "val_roc",
                    ]
                )
            pbar_b = Progbar(target=len(self.train_data))
            for step, x_batch in enumerate(self.train_data):
                loss, output = self._train_step(x_batch, x_batch)
                loss += loss #self.loss_fn(x_batch, output)
                pbar_b.update(step, values=[("loss", loss)], finalize=False)

            if self.val_data is not None:
                pbar_v = Progbar(target=len(self.val_data))
                for step, x_batch in enumerate(self.val_data):
                    output,_ = self.model(x_batch, training=False)
                    val_loss += self.loss_fn(output,x_batch)
                    pbar_v.update(step + 1,values=[("val_loss", loss)], finalize=False)
                
                recon = self.predict(X_val)[0] # index 0 since we want the reconstruction
                val_scores = TCNAE.get_scores(X_val,recon)
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
                wait_roc =0

            if wait_roc > self.patience_roc:
                self.build_model(set=True)
                self.tmp.set_weights(best_roc_weights)
                print("FOUND ROC")
                return self.tmp
            print(f"Wait roc: {wait_roc}")
            wait_roc +=1

            if epoch == (self.n_epochs -1):
                return self.model

          
    def predict(self, X, **predict_params):
        return self.model.predict(x=X, **predict_params)

    def evaluate(self, X):
        return self.model.evaluate(X, X, batch_size=self.batch_size)

    def _reset_graph(self):
        tf.keras.backend.clear_session()

    def _set_seed(self):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

    def free_tf_memory(self):
        tf.keras.backend.clear_session()

    @staticmethod
    def get_scores(X_true, X_recon):
        return tf.keras.metrics.mean_squared_error(X_true.squeeze(), X_recon.squeeze())

    @staticmethod
    def extract_latent(m):
        NEW = Sequential()
        j = 0
        for i in range(0, len(m.layers)):
            if type(m.layers[i]) == Dense:
                j = 1
            if j == 2:
                break
            else:
                NEW.add(m.get_layer(index=i))
        return NEW