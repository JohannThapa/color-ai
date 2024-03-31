import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class TensorFlowModel:
    def __init__(self, input_shape, num_classes):
        """
        Initializes Model with input shape and number of classes.
        Args:
        - input_shape (tuple): Shape of input data.
        - num_classes (int): Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the TensorFlow model architecture.
        Returns:
        - tf.keras.Model: Compiled TensorFlow model.
        """
        input_layer = Input(shape=self.input_shape)
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        """
        Trains the TensorFlow model.
        Args:
        - x_train (numpy.ndarray): Input training data.
        - y_train (numpy.ndarray): Target training labels.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        """
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, x_test):
        """
        Performs prediction using the TensorFlow model.
        Args:
        - x_test (numpy.ndarray): Input test data.
        Returns:
        - numpy.ndarray: Predicted labels.
        """
        return self.model.predict(x_test)
