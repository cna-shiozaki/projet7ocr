import os

import numpy as np

import pandas as pd
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split

@dataclass
class BaselineModel():
    name : str = "Random Baseline"

    def train(self, training_data):
        """ Utiliser training data pour entrainer le modèle """
        pass

    def predict(self, X : np.array) -> list:
        return [ np.random.random() for x in X]



