#105106819 Suman Sutparai
# Base model for TBRGS
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config=None):
        """Initialize the base model.
        
        Args:
            config (dict, optional): Configuration settings. Defaults to None.
        """
        self.model = None
        self.config = config or {}

    @abstractmethod
    def build_model(self, input_shape):
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32, verbose=1):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path):
        if self.model is not None:
            self.model.save(path)
        else:
            raise ValueError("Model has not been built.") 