import logging
import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from typing import *
from anndata import AnnData


from .dummy_dataset import DummyDataset
from .single_cell_dataset import SingleCellDatasetSplit


def obs_names_from_dataset(dset: Dataset) -> Union[List[str], None]:
    """Extract obs names from a dataset, or None if this fails"""
    if isinstance(dset, DummyDataset):
        return None
    elif isinstance(dset, (SplicedDataset, PairedDataset)):
        return dset.obs_names
    elif isinstance(dset, EncodedDataset):
        return dset.obs_names
    elif isinstance(dset, SingleCellDatasetSplit):
        return list(dset.obs_names)
    elif isinstance(dset.data_raw, AnnData):
        return list(dset.data_raw.obs_names)
    return None


class SplicedDataset(Dataset):
    """
    Combines two datasets into one, where the first denotes X and the second denotes Y.
    A spliced datset indicates that the inputs of x should predict the outputs of y.
    Tries to match var names when possible

    Flat mode also assumes that the input datasets are also flattened/catted
    """

    def __init__(self, dataset_x, dataset_y, flat_mode: bool = False):
        assert isinstance(
            dataset_x, Dataset
        ), f"Bad type for dataset_x: {type(dataset_x)}"
        assert isinstance(
            dataset_y, Dataset
        ), f"Bad type for dataset_y: {type(dataset_y)}"
        assert len(dataset_x) == len(dataset_y), "Mismatched length"
        self.flat_mode = flat_mode

        self.obs_names = None
        x_obs_names = obs_names_from_dataset(dataset_x)
        y_obs_names = obs_names_from_dataset(dataset_y)
        if x_obs_names is not None and y_obs_names is not None:
            logging.info("Checking obs names for two input datasets")
            for i, (x, y) in enumerate(zip(x_obs_names, y_obs_names)):
                if x != y:
                    raise ValueError(
                        f"Datasets have a different label at the {i}th index: {x} {y}"
                    )
            self.obs_names = list(x_obs_names)
        elif x_obs_names is not None:
            self.obs_names = x_obs_names
        elif y_obs_names is not None:
            self.obs_names = y_obs_names
        else:
            raise ValueError("Both components of combined dataset appear to be dummy")

        self.dataset_x = dataset_x
        self.dataset_y = dataset_y

    def get_feature_labels(self) -> List[str]:
        """Return the names of the combined features"""
        return list(self.dataset_x.data_raw.var_names) + list(
            self.dataset_y.data_raw.var_names
        )

    def get_obs_labels(self) -> List[str]:
        """Return the names of each example"""
        return self.obs_names

    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, i):
        """Assumes both return a single output"""
        pair = (self.dataset_x[i][0], self.dataset_y[i][1])
        if self.flat_mode:
            raise NotImplementedError(f"Flat mode is not defined for spliced dataset")
        return pair


class PairedDataset(SplicedDataset):
    """
    Combines two datasets into one, where input is now (x1, x2) and
    output is (y1, y2). A Paired dataset simply combines x and y
    by returning the x input and y input as a tuple, and the x output
    and y output as a tuple, and does not "cross" between the datasets
    """

    # Inherits the init from SplicedDataset since we're doing the same thing - recording
    # the two different datasets
    def __getitem__(self, i):
        x1 = self.dataset_x[i]
        x2 = self.dataset_y[i]
        x_pair = (x1[0], x2[0])
        y_pair = (x1[1], x2[1])
        if not self.flat_mode:
            return x_pair, y_pair
        else:
            retval = torch.cat(x_pair), torch.cat(y_pair)
            return retval


class CattedDataset(Dataset):
    """
    Given several datasets, return a "catted" version
    """

    def __init__(self, dsets: Iterable[Dataset], shuffle: bool = True):
        self.dsets = dsets
        self.lengths = [len(d) for d in self.dsets]
        self.cumsum = np.cumsum(self.lengths)
        self.total_length = sum(self.lengths)
        self.idx_map = np.arange(self.total_length)
        if shuffle:
            np.random.shuffle(self.idx_map)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int):
        i = self.idx_map[idx]
        # Index of the dataset that we want
        dset_idx = np.searchsorted(self.cumsum, i)
        # Index within that dataset
        return self.dsets[dset_idx][i % self.cumsum[dset_idx - 1]]


class EncodedDataset(Dataset):
    """
    Sits on top of a PairedDataset that encodes each point
    such that we return (encoded(x), y)
    """

    def __init__(self, sc_dataset: PairedDataset, model, input_mode: str = "RNA"):
        # Mode is the source for the encoded representation
        assert input_mode in ["RNA", "ATAC"], f"Unrecognized mode: {input_mode}"
        rna_encoded, atac_encoded = model.get_encoded_layer(sc_dataset)
        if input_mode == "RNA":
            encoded = rna_encoded
        else:
            encoded = atac_encoded
        self.encoded = AnnData(encoded, obs=pd.DataFrame(index=sc_dataset.obs_names))
        self.obs_names = sc_dataset.obs_names

    def __len__(self):
        return self.encoded.shape[0]

    def __getitem__(self, idx: int):
        """Returns the idx-th item as (encoded(x), y)"""
        enc = self.encoded.X[idx]
        enc_tensor = torch.from_numpy(enc).type(torch.FloatTensor)
        return enc_tensor, enc_tensor
