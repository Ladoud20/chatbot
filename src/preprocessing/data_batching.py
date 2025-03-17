import numpy as np


class DataBatching:

    def __init__(self, texts, num_batchs=50):
        self.texts = texts
        self.num_batchs = num_batchs

    def batch(self):
        return np.array_split(self.texts, self.num_batchs)