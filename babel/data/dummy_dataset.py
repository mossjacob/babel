import torch
import numpy as np

from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    Returns dummy of a given shape for each __getitem__ call
    Returns same value for both x and y
    """

    def __init__(self, shape: int, length: int, mode: str = "zeros"):
        assert mode in ["zeros", "random"]
        self.shape = shape  # Shape of each individual feature vector
        self.length = length
        self.mode = mode

    def __get_random_vals(self):
        rand = np.random.random(size=self.shape)
        return torch.from_numpy(rand).type(torch.FloatTensor)

    def __get_zero_vals(self):
        zeros = np.zeros(self.shape)
        return torch.from_numpy(zeros).type(torch.FloatTensor)

    def __len__(self):
        return self.length

    def __getitem__(self, _idx):
        """Return dummy values"""
        if self.mode == "zeros":
            x = self.__get_zero_vals()
        elif self.mode == "random":
            x = self.__get_random_vals()
        else:
            raise ValueError(f"Unrecognized mode: {self.mode}")

        return x, x