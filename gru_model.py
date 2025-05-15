#105106819 Suman Sutparai
import tensorflow as tf
from ml.base_model import BaseModel

class GRUModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = None

    def build(self, input_shape):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(64, input_shape=input_shape, return_sequences=True))
        model.add(tf.keras.layers.GRU(32))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model

    def train(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.build(X_train.shape[1:])
        super().train(X_train, y_train, X_val, y_val)

    def predict(self, X):
        return self.model.predict(X)

    def predict_sequence(self, X, steps):
        """Predict a sequence of future values.
        
        Args:
            X (np.ndarray): Input sequence.
            steps (int): Number of steps to predict.
            
        Returns:
            np.ndarray: Predicted sequence.
        """
        predictions = []
        current_input = X.copy()
        
        for _ in range(steps):
            # Make prediction
            pred = self.predict(current_input)
            predictions.append(pred[0, 0])
            
            # Update input sequence
            current_input = np.roll(current_input, -1, axis=1)
            current_input[0, -1] = pred[0, 0]
            
        return np.array(predictions) 