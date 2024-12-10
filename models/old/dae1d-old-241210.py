import os

os.environ['KERAS_BACKEND'] = 'torch'

from keras import layers, Model, Input


class DiamondDAE1D:
    """
    DiamondDAE1D: A 1D Convolutional Denoising Autoencoder for Spectral Data.

    This class implements a denoising autoencoder architecture specifically designed for 
    one-dimensional data, such as spectral data. The model learns to encode noisy input 
    sequences into a compressed latent representation and then decodes them back into 
    clean output sequences.

    The architecture is fully configurable, with support for custom numbers of layers, 
    filters, kernel sizes, and activation functions.

    Attributes:
        filters (list[int]): 
            List of integers representing the number of filters for each convolutional 
            layer in the encoder. The decoder uses the same filters in reverse order.
        kernel_size (int): 
            Size of the convolutional kernel to be used in all layers.
        activations (str): 
            Activation function applied to the hidden layers (e.g., 'relu').
        output_activation (str): 
            Activation function applied to the output layer (e.g., 'linear').
        model (keras.Model): 
            The underlying Keras functional model that performs the autoencoding.

    Methods:
        build_model(input_shape):
            Builds the functional model based on the provided input shape.
        summary():
            Prints a summary of the model, showing all layers and their parameters.
        get_model():
            Returns the underlying Keras functional model.

    Usage:
        # Instantiate the class
        dae = DiamondDAE1D(
            n_convlayers=6,
            filters=[32, 64, 128],
            kernel_size=3,
            activations="relu",
            output_activation="linear"
        )

        # Build the model with input shape
        dae.build_model(input_shape=(100, 1))

        # Print model summary
        dae.summary()

        # Retrieve the Keras model
        functional_model = dae.get_model()

    Notes:
        1. The total number of layers (`n_convlayers`) must be evenly split between the encoder 
           and decoder. The `filters` list specifies the number of layers in the encoder.
        2. This class is particularly well-suited for 1D time-series or spectral data, where the 
           sequences have a fixed length.
        3. The model supports further customization by extending or modifying the methods 
           provided in the class.

    Example Architecture:
        If `n_convlayers=6` and `filters=[32, 64, 128]`, the architecture will look like:
            - Input Layer: (sequence_length, 1)
            - Encoder:
                - Conv1D: 32 filters
                - MaxPooling1D
                - Conv1D: 64 filters
                - MaxPooling1D
                - Conv1D: 128 filters
                - MaxPooling1D
            - Decoder:
                - UpSampling1D
                - Conv1DTranspose: 128 filters
                - UpSampling1D
                - Conv1DTranspose: 64 filters
                - UpSampling1D
                - Conv1DTranspose: 32 filters
            - Output Layer:
                - Conv1DTranspose: 1 filter
    """


    def __init__(self, n_convlayers, filters, kernel_size, activations="relu", output_activation="linear"):
        """
        Initialize the autoencoder.

        Args:
            n_convlayers (int): Total number of convolutional layers (including encoder and decoder).
            filters (list[int]): List of filters for each encoder layer.
            kernel_size (int): Kernel size for all convolutions.
            activations (str): Activation function for hidden layers.
            output_activation (str): Activation function for the output layer.
        """
        # Validate the number of filters
        if len(filters) != n_convlayers // 2:
            raise ValueError(
                f"Expected {n_convlayers // 2} elements in `filters`, "
                f"but got {len(filters)}. `filters` must match encoder layers."
            )

        self.filters = filters
        self.kernel_size = kernel_size
        self.activations = activations
        self.output_activation = output_activation

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
