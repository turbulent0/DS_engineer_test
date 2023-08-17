
# 1. DigitClassificationInterface (Interface):
# This interface defines the structure that each classification model class should follow.
from abc import ABC, abstractmethod

class DigitClassificationInterface(ABC):
    @abstractmethod
    def predict(self, image):
        pass

# 2. CNNClassifier (Class implementing DigitClassificationInterface):

import tensorflow as tf

class CNNClassifier(DigitClassificationInterface):
    def __init__(self):
        # Initialize your CNN model here

     def predict(self, image):
        # Implement prediction logic for CNN
        return predicted_value


# 3. RandomForestClassifier (Class implementing DigitClassificationInterface):
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifier(DigitClassificationInterface):
    def __init__(self):
        # Initialize your Random Forest model here

     def predict(self, image):
        # Implement prediction logic for Random Forest
        return predicted_value


# 4.RandomClassifier (Class implementing DigitClassificationInterface)

import random

class RandomClassifier(DigitClassificationInterface):
    def predict(self, image):
        return random.randint(0, 9)  # Return a random integer between 0 and 9


# 5. DigitClassifier (Main Class):
# This class takes the name of the algorithm as an input parameter and provides a consistent interface for making predictions using any of the available classification models.
class DigitClassifier:
    def __init__(self, algorithm):
        self.algorithm = algorithm

        if self.algorithm == 'cnn':
            self.model = CNNClassifier()
        elif self.algorithm == 'rf':
            self.model = RandomForestClassifier()
        elif self.algorithm == 'rand':
            self.model = RandomClassifier()
        else:
            raise ValueError("Invalid algorithm name")

    def predict(self, image):
        return self.model.predict(image)

