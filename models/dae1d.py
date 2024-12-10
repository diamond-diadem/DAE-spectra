import os

os.environ['KERAS_BACKEND'] = 'torch'

import keras
from keras import layers, Model, Input

import matplotlib.pyplot as plt


class DiamondDAE1D:
    """
    DiamondDAE1D: A 1D Convolutional Denoising Autoencoder for Spectral Data.

    This class implements a configurable denoising autoencoder architecture designed 
    for one-dimensional data, such as spectral data or time-series. The model can be 
    used to compress noisy input sequences into a latent representation and decode 
    them back into clean sequences.

    The architecture is flexible, supporting both even and odd numbers of convolutional 
    layers (`n_convlayers`). When `n_convlayers` is odd, a latent layer is automatically 
    added, and its dimensionality can be customized via the `latent_dim` parameter.

    ### Key Features:
    - Fully configurable number of convolutional layers, filters, kernel size, and activations.
    - Optional latent layer for architectures with odd numbers of layers.
    - Supports custom latent dimensions for fine-tuned compression.

    ### Attributes:
    - `n_convlayers` (int):
        Total number of convolutional layers in the model, including the encoder, decoder, 
        and optionally a latent layer. Must be greater than or equal to 2.
    - `filters` (list[int]):
        List of integers specifying the number of filters for each convolutional layer 
        in the encoder. The decoder mirrors these filters in reverse order.
    - `kernel_size` (int):
        Size of the kernel for all convolutional and transposed convolutional layers.
    - `activations` (str):
        Activation function to use in all hidden layers (e.g., 'relu').
    - `output_activation` (str):
        Activation function to use in the output layer (e.g., 'linear').
    - `latent_dim` (int):
        Dimensionality of the latent layer. Used only if `n_convlayers` is odd. Defaults 
        to the number of filters in the last encoder layer (`filters[-1]`).
    - `model` (keras.Model):
        The underlying Keras functional model created when `build_model` is called.

    ### Methods:
    - `build_model(input_shape)`:
        Builds the functional model based on the provided input shape.
    - `summary()`:
        Prints a summary of the model, showing all layers, shapes, and parameters.
    - `get_model()`:
        Returns the underlying Keras functional model.

    ### Example Usage:

    ```python
    # Example with even number of layers
    dae_even = DiamondDAE1D(
        n_convlayers=6,
        filters=[32, 64, 128],
        kernel_size=3,
        activations="relu",
        output_activation="linear"
    )
    dae_even.build_model(input_shape=(100, 1))
    dae_even.summary()

    # Example with odd number of layers and custom latent dimension
    dae_odd = DiamondDAE1D(
        n_convlayers=5,
        filters=[32, 64],
        kernel_size=3,
        activations="relu",
        output_activation="linear",
        latent_dim=256
    )
    dae_odd.build_model(input_shape=(100, 1))
    dae_odd.summary()
    ```

    ### Notes:
    - **Number of Layers (`n_convlayers`)**:
        - If `n_convlayers` is even, the architecture consists of a symmetric encoder-decoder.
        - If `n_convlayers` is odd, an additional latent layer is added between the encoder 
          and decoder, with its dimensionality controlled by `latent_dim`.
    - **Filters**:
        - The length of the `filters` list must equal `n_convlayers // 2`, representing 
          the number of layers in the encoder. The decoder mirrors these filters.
    - **Input Shape**:
        - The `input_shape` parameter in `build_model` must be a tuple `(sequence_length, channels)`, 
          where `sequence_length` is the length of the input sequence, and `channels` is the 
          number of input features.

    ### Example Architecture:
    For `n_convlayers=5`, `filters=[32, 64]`, and `latent_dim=256`:
    ```
    Model: "DiamondDAE1D"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_layer (InputLayer)    [(None, 100, 1)]          0         
                                                                     
     encoder_conv_0 (Conv1D)     (None, 100, 32)          128       
                                                                     
     encoder_pool_0 (MaxPooling1 (None, 50, 32)           0         
                                                                     
     encoder_conv_1 (Conv1D)     (None, 50, 64)           6208      
                                                                     
     encoder_pool_1 (MaxPooling1 (None, 25, 64)           0         
                                                                     
     latent_layer (Conv1D)       (None, 25, 256)          49408     
                                                                     
     decoder_upsample_0 (UpSampl (None, 50, 256)          0         
                                                                     
     decoder_conv_0 (Conv1DTrans (None, 50, 64)           49216     
                                                                     
     decoder_upsample_1 (UpSampl (None, 100, 64)          0         
                                                                     
     decoder_conv_1 (Conv1DTrans (None, 100, 32)          6176      
                                                                     
     output_layer (Conv1DTranspo (None, 100, 1)           97        
    =================================================================
    Total params: 111,233
    Trainable params: 111,233
    Non-trainable params: 0
    _________________________________________________________________
    ```
    """


    def __init__(self, n_convlayers, filters, kernel_size, activations="relu", output_activation="linear", latent_dim=None):
        """
        Initialize the autoencoder.

        Args:
            n_convlayers (int): Total number of convolutional layers (including encoder and decoder).
            filters (list[int]): List of filters for each encoder layer.
            kernel_size (int): Kernel size for all convolutions.
            activations (str): Activation function for hidden layers.
            output_activation (str): Activation function for the output layer.
            latent_dim (int): Dimension of the latent layer. If None, uses the same dimension as the last encoder layer.
        """
        # Validate the number of filters
        if len(filters) != n_convlayers // 2:
            raise ValueError(
                f"Expected {n_convlayers // 2} elements in `filters`, "
                f"but got {len(filters)}. `filters` must match encoder layers."
            )

        self.n_convlayers = n_convlayers
        self.filters = filters
        self.kernel_size = kernel_size
        self.activations = activations
        self.output_activation = output_activation
        self.latent_dim = latent_dim or filters[-1]

        # The Keras functional model will be created in `build_model`
        self.model = None

    def build_model(self, input_shape):
        """
        Build the Keras functional model.

        Args:
            input_shape (tuple): Shape of the input data (sequence_length, channels).
        """
        inputs = Input(shape=input_shape, name="input_layer")
        x = inputs

        # Encoder
        for i, f in enumerate(self.filters):
            x = layers.Conv1D(
                filters=f,
                kernel_size=self.kernel_size,
                activation=self.activations,
                padding="same",
                name=f"encoder_conv_{i}"
            )(x)
            x = layers.MaxPooling1D(name=f"encoder_pool_{i}")(x)

        # Latent layer (only if n_convlayers is odd)
        if self.n_convlayers % 2 != 0:
            x = layers.Conv1D(
                filters=self.latent_dim,
                kernel_size=self.kernel_size,
                activation=self.activations,
                padding="same",
                name="latent_layer"
            )(x)

        # Decoder
        for i, f in enumerate(reversed(self.filters)):
            x = layers.UpSampling1D(name=f"decoder_upsample_{i}")(x)
            x = layers.Conv1DTranspose(
                filters=f,
                kernel_size=self.kernel_size,
                activation=self.activations,
                padding="same",
                name=f"decoder_conv_{i}"
            )(x)

        # Output layer
        outputs = layers.Conv1DTranspose(
            filters=1,
            kernel_size=self.kernel_size,
            activation=self.output_activation,
            padding="same",
            name="output_layer"
        )(x)

        # Create the functional model
        self.model = Model(inputs, outputs, name="DiamondDAE1D")

    def summary(self):
        """Print the summary of the model."""
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")
        self.model.summary()

    def get_model(self):
        """
        Returns the underlying Keras model.
        """
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")
        return self.model
    
    def train_model(
        self, 
        x_train, 
        y_train, 
        x_val=None, 
        y_val=None, 
        batch_size=32, 
        epochs=10, 
        optimizer=None, 
        learning_rate=0.001, 
        loss="mse", 
        metrics=None, 
        verbose=1
    ):
        """
        Train the model with the given training and validation data.

        Args:
            x_train: Training input data.
            y_train: Training target data.
            x_val: Validation input data (optional).
            y_val: Validation target data (optional).
            batch_size (int): Batch size for training (default: 32).
            epochs (int): Number of training epochs (default: 10).
            optimizer: Optimizer to use for training. If None, uses Adam with specified learning rate.
            learning_rate (float): Learning rate for the optimizer (default: 0.001).
            loss: Loss function (default: 'mse').
            metrics: List of metrics to monitor during training (default: None).
            verbose (int): Verbosity mode (default: 1).

        Returns:
            History object containing training history.
        """
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")

        # Handle optimizer
        if optimizer is None:
            # Default optimizer: Adam with specified learning rate
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            # If an optimizer is provided, try to set the learning rate
            try:
                if hasattr(optimizer, "learning_rate"):
                    optimizer.learning_rate = learning_rate
                else:
                    print(f"Warning: Optimizer {type(optimizer).__name__} does not support setting `learning_rate` directly.")
            except Exception as e:
                print(f"Could not set learning rate for optimizer {type(optimizer).__name__}: {e}")

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics or [])

        # Train the model
        history = self.model.fit(
            x_train, 
            y_train, 
            validation_data=(x_val, y_val) if x_val is not None and y_val is not None else None, 
            batch_size=batch_size, 
            epochs=epochs, 
            verbose=verbose
        )

        return history
    
    def evaluate_model(self, x_test, y_test, batch_size=32, verbose=1):
        """
        Evaluate the model on test data.

        Args:
            x_test: Test input data.
            y_test: Test target data.
            batch_size (int): Batch size for evaluation (default: 32).
            verbose (int): Verbosity mode (default: 1).

        Returns:
            Dictionary containing the evaluation metrics.
        """
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")

        results = self.model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)

        # Handle single metric (loss only)
        if not isinstance(results, (list, tuple)):
            return {self.model.metrics_names[0]: results}

        # Handle multiple metrics
        return {metric: value for metric, value in zip(self.model.metrics_names, results)}

    
    def save_model(self, filepath):
        """
        Save the model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")
        
        self.model.save(filepath)
    
    @staticmethod
    def load_model(filepath):
        """
        Load a model from a file.

        Args:
            filepath (str): Path to the saved model.

        Returns:
            A new instance of DiamondDAE1D with the loaded model.
        """
        from keras.models import load_model
        loaded_model = load_model(filepath)
        instance = DiamondDAE1D.__new__(DiamondDAE1D)
        instance.model = loaded_model
        return instance
    
    def predict(self, x_input, batch_size=32, verbose=0):
        """
        Generate predictions for the input data.

        Args:
            x_input: Input data for prediction.
            batch_size (int): Batch size for prediction (default: 32).
            verbose (int): Verbosity mode (default: 0).

        Returns:
            Predicted output.
        """
        if self.model is None:
            raise ValueError("The model has not been built yet. Call `build_model()` first.")

        return self.model.predict(x_input, batch_size=batch_size, verbose=verbose)

    def visualize_training(self, history):
        """
        Visualize training and validation loss/metrics.

        Args:
            history: History object returned by the `fit` method.
        """
        if not hasattr(history, "history"):
            raise ValueError("The provided object is not a valid training history.")

        # Plot loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        if "val_loss" in history.history:
            plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        # Plot metrics (if available)
        if "mae" in history.history or "accuracy" in history.history:
            metric = "mae" if "mae" in history.history else "accuracy"
            plt.subplot(1, 2, 2)
            plt.plot(history.history[metric], label=f"Training {metric}")
            if f"val_{metric}" in history.history:
                plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
            plt.title(metric.capitalize())
            plt.xlabel("Epochs")
            plt.ylabel(metric.capitalize())
            plt.legend()

        plt.tight_layout()
        plt.show()



