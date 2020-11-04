"""
Code for loading in single-cell datasets
"""

import os
import sys
import platform
import glob
import subprocess
import shlex
import shutil
import random
import logging
import functools
import itertools
import multiprocessing
import gzip
import collections
import re

from typing import *

import intervaltree

import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.sparse
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import Dataset

import sortedcontainers

import adata_utils
import plot_utils
import utils

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(DATA_DIR)
SNARESEQ_DATA_DIR = os.path.join(DATA_DIR, "snareseq_GSE126074")
assert os.path.isdir(SNARESEQ_DATA_DIR)
TENX_PBMC_DATA_DIR = os.path.join(DATA_DIR, "10x_PBMC")
assert os.path.isdir(TENX_PBMC_DATA_DIR)
SC_RNA_ATAC_DIR = os.path.join(DATA_DIR, "sc_rnaseq_atacseq")
assert os.path.isdir(SC_RNA_ATAC_DIR)
MM9_GTF = os.path.join(DATA_DIR, "Mus_musculus.NCBIM37.67.gtf.gz")
assert os.path.isfile(MM9_GTF)
HG38_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.100.gtf.gz")
assert os.path.isfile(HG38_GTF)
HG19_GTF = os.path.join(DATA_DIR, "Homo_sapiens.GRCh37.87.gtf.gz")
assert os.path.isfile(HG19_GTF)

SNARESEQ_ATAC_CELL_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.barcodes.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)
SNARESEQ_ATAC_CELL_INFO.index.name = "barcodes"

SNARESEQ_ATAC_PEAK_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.peaks.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)
SNARESEQ_ATAC_PEAK_INFO.index.name = "peaks"

SNARESEQ_ATAC_DATA_KWARGS = {
    "fname": os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_chromatin.counts.mtx.gz"
    ),
    "cell_info": SNARESEQ_ATAC_CELL_INFO,
    "gene_info": SNARESEQ_ATAC_PEAK_INFO,
    "transpose": True,
    "selfsupervise": True,  # Doesn't actually do anything
    "binarize": True,  # From SNAREseq paper methods section (SCALE also binarizes, uses either CE or MSE loss)
    "autosomes_only": True,
    "split_by_chrom": True,
    "concat_outputs": True,
    "filt_gene_min_counts": 5,  # From SNAREseq paper methods section: "peaks with fewer than five counts overall"
    "filt_gene_min_cells": 5,  # From SCALE - choose to keep peaks seek in >= 5 cells
    "filt_gene_max_cells": 0.1,  # From SNAREseq paper methods section: filter peaks expressing in more than 10% of cells
    "pool_genomic_interval": 500,  # Smaller bin size because we can handle it
    "normalize": False,  # True,
    "log_trans": False,  # True,
    "y_mode": "x",
    "calc_size_factors": False,  # True,
    "return_sf": False,
}

SNARESEQ_RNA_CELL_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.barcodes.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)
SNARESEQ_RNA_CELL_INFO.index.name = "barcodes"

SNARESEQ_RNA_GENE_INFO = pd.read_csv(
    os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.genes.tsv.gz"
    ),
    sep="\t",
    header=None,
    index_col=0,
)
SNARESEQ_RNA_GENE_INFO.index.name = "gene"

SNARESEQ_RNA_DATA_KWARGS = {
    "fname": os.path.join(
        SNARESEQ_DATA_DIR, "GSE126074_AdBrainCortex_SNAREseq_cDNA.counts.mtx.gz"
    ),
    "cell_info": SNARESEQ_RNA_CELL_INFO,
    "gene_info": SNARESEQ_RNA_GENE_INFO,
    "transpose": True,
    "selfsupervise": True,
    "binarize": False,
    "gtf_file": MM9_GTF,
    "autosomes_only": True,
    "sort_by_pos": True,
    "split_by_chrom": True,
    "concat_outputs": True,
    "binarize": False,
    "filt_cell_min_genes": 200,  # SNAREseq paper: minimum of 200 genes
    "filt_cell_max_genes": 2500,  # SNAREseq paper: maximum of 2500 genes
    "normalize": True,
    "log_trans": True,
    "clip": 0.5,  # Clip the bottom and top 0.5%
    "y_mode": "size_norm",
    "calc_size_factors": True,
    "return_sf": False,
    "cluster_res": 1.5,
}

TENX_PBMC_ATAC_BINS_FILE = os.path.join(TENX_PBMC_DATA_DIR, "pbmc_merged_atac_bins.txt")
TENX_PBMC_ATAC_DATA_KWARGS = {
    "fname": (
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep1_filtered_feature_bc_matrix.h5"),
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep2_filtered_feature_bc_matrix.h5"),
    ),
    "reader": functools.partial(
        utils.sc_read_multi_files,
        reader=lambda x: repool_atac_bins(
            utils.sc_read_10x_h5_ft_type(x, "Peaks"),
            utils.read_delimited_file(TENX_PBMC_ATAC_BINS_FILE),
        ),
    ),
    "transpose": False,
    "selfsupervise": True,  # Doesn't actually do anything
    "binarize": True,  # From SNAREseq paper methods section (SCALE also binarizes, uses either CE or MSE loss)
    "autosomes_only": True,
    "split_by_chrom": True,
    "concat_outputs": True,
    "filt_gene_min_counts": 5,  # From SNAREseq paper methods section: "peaks with fewer than five counts overall"
    "filt_gene_min_cells": 5,  # From SCALE - choose to keep peaks seek in >= 5 cells
    "filt_gene_max_cells": 0.1,  # From SNAREseq paper methods section: filter peaks expressing in more than 10% of cells
    "pool_genomic_interval": 0,  # Do not pool
    "normalize": False,  # True,
    "log_trans": False,  # True,
    "y_mode": "x",
    "calc_size_factors": False,  # True,
    "return_sf": False,
}

TENX_PBMC_RNA_DATA_KWARGS = {
    "fname": (
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep1_filtered_feature_bc_matrix.h5"),
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep2_filtered_feature_bc_matrix.h5"),
    ),
    "reader": functools.partial(
        utils.sc_read_multi_files,
        reader=lambda x: utils.sc_read_10x_h5_ft_type(x, "Gene Expression"),
    ),
    "transpose": False,  # We do not transpose because the h5 is already cell x gene
    "gtf_file": HG38_GTF,
    "autosomes_only": True,
    "sort_by_pos": True,
    "split_by_chrom": True,
    "concat_outputs": True,
    "selfsupervise": True,
    "binarize": False,
    "filt_cell_min_genes": 200,  # SNAREseq paper: minimum of 200 genes
    "filt_cell_max_genes": 7000,  # SNAREseq paper: maximum of 2500 genes
    "normalize": True,
    "log_trans": True,
    "clip": 0.5,  # Clip the bottom and top 0.5%
    "y_mode": "size_norm",  # The output that we learn to predict
    "calc_size_factors": True,
    "return_sf": False,
    "cluster_res": 1.5,
}


@functools.lru_cache(4)
def sc_read_mtx(fname: str, dtype: str = "float32"):
    """Helper function for reading mtx files so we can cache the result"""
    return sc.read_mtx(filename=fname, dtype=dtype)


class SingleCellDataset(Dataset):
    """
    Given a sparse matrix file, load in dataset

    If transforms is given, it is applied after all the pre-baked transformations. These
    can be things like sklearn MaxAbsScaler().fit_transform
    """

    def __init__(
        self,
        fname: Union[str, List[str]],
        reader: Callable = sc_read_mtx,
        transpose: bool = True,
        mode: str = "train",
        data_split_by_cluster: str = "leiden",  # Specify as leiden
        valid_cluster_id: int = 0,  # Only used if data_split_by_cluster is on
        test_cluster_id: int = 1,
        data_split_by_cluster_log: bool = True,
        predefined_split: List[str] = None,
        cell_info: pd.DataFrame = None,
        gene_info: pd.DataFrame = None,
        selfsupervise: bool = True,
        binarize: bool = False,
        filt_cell_min_counts=None,  # All of these are off by default
        filt_cell_max_counts=None,
        filt_cell_min_genes=None,
        filt_cell_max_genes=None,
        filt_gene_min_counts=None,
        filt_gene_max_counts=None,
        filt_gene_min_cells=None,
        filt_gene_max_cells=None,
        pool_genomic_interval: Union[int, List[str]] = 0,
        top_n_features: int = 0,
        calc_size_factors: bool = True,
        normalize: bool = True,
        log_trans: bool = True,
        clip: float = 0,
        sort_by_pos: bool = False,
        split_by_chrom: bool = False,
        concat_outputs: bool = False,  # Instead of outputting a list of tensors, concat
        autosomes_only: bool = False,
        # high_confidence_clustering_genes: List[str] = [],  # used to build clustering
        x_dropout: bool = False,
        y_mode: str = "size_norm",
        sample_y: bool = False,
        return_sf: bool = True,
        return_pbulk: bool = False,
        filter_features: dict = {},
        filter_samples: dict = {},
        transforms: List[Callable] = [],
        gtf_file: str = MM9_GTF,  # GTF file mapping genes to chromosomes, unused for ATAC
        cluster_res: float = 2.0,
        cache_prefix: str = "",
    ):
        """
        Clipping is performed AFTER normalization
        Binarize will turn all counts into binary 0/1 indicators before running normalization code
        If pool_genomic_interval is -1, then we pool based on proximity to gene
        """
        assert mode in [
            "train",
            "valid",
            "trainvalid",
            "test",
            "all",
        ], f"Unrecognized mode: {mode}"
        assert y_mode in [
            "size_norm",
            "log_size_norm",
            "raw_count",
            "log_raw_count",
            "x",
        ], f"Unrecognized mode for y output: {y_mode}"
        if y_mode == "size_norm":
            assert calc_size_factors
        self.mode = mode
        self.selfsupervise = selfsupervise
        self.x_dropout = x_dropout
        self.y_mode = y_mode
        self.sample_y = sample_y
        self.binarize = binarize
        self.calc_size_factors = calc_size_factors
        self.return_sf = return_sf
        self.return_pbulk = return_pbulk
        self.transforms = transforms
        self.cache_prefix = cache_prefix
        self.sort_by_pos = sort_by_pos
        self.split_by_chrom = split_by_chrom
        self.concat_outputs = concat_outputs
        self.autosomes_only = autosomes_only
        self.cluster_res = cluster_res
        self.data_split_by_cluster = data_split_by_cluster
        self.valid_cluster_id = valid_cluster_id
        self.test_cluster_id = test_cluster_id
        self.data_split_by_cluster_log = data_split_by_cluster_log
        self.predefined_split = predefined_split

        self.data_raw = reader(fname)
        assert isinstance(
            self.data_raw, AnnData
        ), f"Reader must return AnnData but got {type(self.data_raw)}"
        if not isinstance(self.data_raw.X, scipy.sparse.csr_matrix):
            self.data_raw.X = scipy.sparse.csr_matrix(
                self.data_raw.X
            )  # Convert to sparse matrix

        if transpose:
            self.data_raw = self.data_raw.T

        # Filter out undesirable var/obs
        # self.__filter_obs_metadata(filter_samples=filter_samples)
        # self.__filter_var_metadata(filter_features=filter_features)
        self.data_raw = adata_utils.filter_adata(
            self.data_raw, filt_cells=filter_samples, filt_var=filter_features
        )

        # Attach obs/var annotations
        if cell_info is not None:
            assert isinstance(cell_info, pd.DataFrame)
            if self.data_raw.obs is not None and not self.data_raw.obs.empty:
                self.data_raw.obs = self.data_raw.obs.join(
                    cell_info, how="left", sort=False
                )
            else:
                self.data_raw.obs = cell_info
            assert (
                self.data_raw.shape[0] == self.data_raw.obs.shape[0]
            ), f"Got discordant shapes for data and obs: {self.data_raw.shape} {self.data_raw.obs.shape}"

        if gene_info is not None:
            assert isinstance(gene_info, pd.DataFrame)
            if (
                self.data_raw.var is not None and not self.data_raw.var.empty
            ):  # Is not None and is not empty
                self.data_raw.var = self.data_raw.var.join(
                    gene_info, how="left", sort=False
                )
            else:
                self.data_raw.var = gene_info
            assert (
                self.data_raw.shape[1] == self.data_raw.var.shape[0]
            ), f"Got discordant shapes for data and var: {self.data_raw.shape} {self.data_raw.var.shape}"

        if sort_by_pos:
            genes_reordered, chroms_reordered = reorder_genes_by_pos(
                self.data_raw.var_names, gtf_file=gtf_file, return_chrom=True
            )
            self.data_raw = self.data_raw[:, genes_reordered]

        self.__annotate_chroms(gtf_file)
        if self.autosomes_only:
            autosomal_idx = [
                i
                for i, chrom in enumerate(self.data_raw.var["chrom"])
                if utils.is_numeric(chrom.strip("chr"))
            ]
            self.data_raw = self.data_raw[:, autosomal_idx]

        # Sort by the observation names so we can combine datasets
        sort_order_idx = np.argsort(self.data_raw.obs_names)
        self.data_raw = self.data_raw[sort_order_idx, :]
        # NOTE pooling occurs AFTER feature/observation filtering
        if pool_genomic_interval:
            self.__pool_features(pool_genomic_interval=pool_genomic_interval)
            # Re-annotate because we have lost this information
            self.__annotate_chroms(gtf_file)

        # Preprocess the data now that we're done filtering
        if self.binarize:
            # If we are binarizing data we probably don't care about raw counts
            # self.data_raw.raw = self.data_raw.copy()  # Store original counts
            self.data_raw.X[self.data_raw.X.nonzero()] = 1  # .X here is a csr matrix

        adata_utils.annotate_basic_adata_metrics(self.data_raw)
        adata_utils.filter_adata_cells_and_genes(
            self.data_raw,
            filter_cell_min_counts=filt_cell_min_counts,
            filter_cell_max_counts=filt_cell_max_counts,
            filter_cell_min_genes=filt_cell_min_genes,
            filter_cell_max_genes=filt_cell_max_genes,
            filter_gene_min_counts=filt_gene_min_counts,
            filter_gene_max_counts=filt_gene_max_counts,
            filter_gene_min_cells=filt_gene_min_cells,
            filter_gene_max_cells=filt_gene_max_cells,
        )
        self.data_raw = adata_utils.normalize_count_table(  # Normalizes in place
            self.data_raw,
            size_factors=calc_size_factors,
            top_n=top_n_features,
            normalize=normalize,
            log_trans=log_trans,
        )

        if clip > 0:
            assert isinstance(clip, float) and 0.0 < clip < 50.0
            clip_low, clip_high = np.percentile(
                self.data_raw.X.flatten(), [clip, 100.0 - clip]
            )
            assert (
                clip_low < clip_high
            ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
            self.data_raw.X = np.clip(self.data_raw.X, clip_low, clip_high)

        # Apply any final transformations
        if self.transforms:
            for trans in self.transforms:
                self.data_raw.X = trans(self.data_raw.X)

        # Make sure the data is a sparse matrix
        if not isinstance(self.data_raw.X, scipy.sparse.csr_matrix):
            self.data_raw.X = scipy.sparse.csr_matrix(self.data_raw.X)

        # Do all normalization before we split to make sure all folds get the same normalization
        if self.predefined_split is not None:
            logging.info("Got predefined split, ignoring mode")
            self.data_raw = self.data_raw[
                [i for i in self.predefined_split if i in self.data_raw.obs.index],
            ]
        else:
            if self.data_split_by_cluster:
                self.__split_train_valid_test_cluster(
                    clustering_key=self.data_split_by_cluster
                    if isinstance(self.data_split_by_cluster, str)
                    else "leiden",
                    valid_cluster={str(self.valid_cluster_id)},
                    test_cluster={str(self.test_cluster_id)},
                )
            else:
                self.__split_train_valid_test()

        self.size_factors = (
            torch.from_numpy(self.data_raw.obs.size_factors.values).type(
                torch.FloatTensor
            )
            if self.return_sf
            else None
        )
        self.cell_sim_mat = (
            euclidean_sim_matrix(self.size_norm_counts) if self.sample_y else None
        )  # Skip calculation if we don't need

        # Perform file backing if necessary
        self.data_raw_cache_fname = ""
        if self.cache_prefix:
            self.data_raw_cache_fname = self.cache_prefix + ".data_raw.h5ad"
            logging.info(f"Setting cache at {self.data_raw_cache_fname}")
            self.data_raw.filename = self.data_raw_cache_fname
            if hasattr(self, "_size_norm_counts"):
                size_norm_cache_name = self.cache_prefix + ".size_norm_counts.h5ad"
                logging.info(
                    f"Setting size norm counts cache at {size_norm_cache_name}"
                )
                self._size_norm_counts.filename = size_norm_cache_name
            if hasattr(self, "_size_norm_log_counts"):
                size_norm_log_cache_name = (
                    self.cache_prefix + ".size_norm_log_counts.h5ad"
                )
                logging.info(
                    f"Setting size log norm counts cache at {size_norm_log_cache_name}"
                )
                self._size_norm_log_counts.filename = size_norm_log_cache_name

    def __annotate_chroms(self, gtf_file: str = "") -> None:
        """Annotates chromosome information on the var field, without the chr prefix"""
        # gtf_file can be empty if we're using atac intervals
        feature_chroms = (
            get_chrom_from_intervals(self.data_raw.var_names)
            if list(self.data_raw.var_names)[0].startswith("chr")
            else get_chrom_from_genes(self.data_raw.var_names, gtf_file)
        )
        self.data_raw.var["chrom"] = feature_chroms

    def __pool_features(self, pool_genomic_interval: Union[int, List[str]]):
        n_obs = self.data_raw.n_obs
        if isinstance(pool_genomic_interval, int):
            if pool_genomic_interval > 0:
                # WARNING This will wipe out any existing var information
                idx, names = get_indices_to_combine(
                    list(self.data_raw.var.index), interval=pool_genomic_interval
                )
                data_raw_aggregated = combine_array_cols_by_idx(  # Returns np ndarray
                    self.data_raw.X, idx,
                )
                data_raw_aggregated = scipy.sparse.csr_matrix(data_raw_aggregated)
                self.data_raw = AnnData(
                    data_raw_aggregated,
                    obs=self.data_raw.obs,
                    var=pd.DataFrame(index=names),
                )
            elif pool_genomic_interval < 0:
                assert (
                    pool_genomic_interval == -1
                ), f"Invalid value: {pool_genomic_interval}"
                # Pool based on proximity to genes
                data_raw_aggregated, names = combine_by_proximity(self.data_raw)
                self.data_raw = AnnData(
                    data_raw_aggregated,
                    obs=self.data_raw.obs,
                    var=pd.DataFrame(index=names),
                )
            else:
                raise ValueError(f"Invalid integer value: {pool_genomic_interval}")
        elif isinstance(pool_genomic_interval, (list, set, tuple)):
            idx = get_indices_to_form_target_intervals(
                self.data_raw.var.index, target_intervals=pool_genomic_interval
            )
            data_raw_aggregated = scipy.sparse.csr_matrix(
                combine_array_cols_by_idx(self.data_raw.X, idx,)
            )
            self.data_raw = AnnData(
                data_raw_aggregated,
                obs=self.data_raw.obs,
                var=pd.DataFrame(index=pool_genomic_interval),
            )
        else:
            raise TypeError(
                f"Unrecognized type for pooling features: {type(pool_genomic_interval)}"
            )
        assert self.data_raw.n_obs == n_obs

    def __split_train_valid_test(self):
        """
        Split the dataset into the appropriate split
        """
        if self.mode != "all":  # Subset data based on which set we're getting
            logging.warning(
                f"Constructing {self.mode} random data split - not recommended due to potential leakage between data split"
            )
            indices = np.arange(self.data_raw.n_obs)
            (
                indices_train,
                indices_valid,
                indices_test,
            ) = shuffle_indices_train_valid_test(
                indices, shuffle=True, seed=1234, valid=0.15, test=0.15
            )
            if self.mode == "train":
                self.data_raw = self.data_raw[indices_train]
            elif self.mode == "valid":
                self.data_raw = self.data_raw[indices_valid]
            elif self.mode == "trainvalid":
                self.data_raw = self.data_raw[
                    np.concatenate((indices_train, indices_valid))
                ]
            elif self.mode == "test":
                self.data_raw = self.data_raw[indices_test]
            else:
                raise ValueError(f"Unrecognized data split mode: {self.mode}")

    def __split_train_valid_test_cluster(
        self, clustering_key: str = "leiden", valid_cluster={"0"}, test_cluster={"1"}
    ) -> None:
        """
        Splits the dataset into appropriate split based on clustering
        Retains similarly sized splits as train/valid/test random from above
        """
        assert not valid_cluster.intersection(
            test_cluster
        ), f"Overlap between valid and test clusters: {valid_cluster} {test_cluster}"
        if clustering_key not in ["leiden", "louvain"]:
            raise ValueError(
                f"Invalid clustering key for data splits: {clustering_key}"
            )
        if self.mode != "all":
            logging.info(
                f"Constructing {self.mode} {clustering_key} {'log' if self.data_split_by_cluster_log else 'linear'} clustered data split with valid test cluster {valid_cluster} {test_cluster}"
            )
            cluster_labels = (
                self.size_norm_log_counts.obs[clustering_key]
                if self.data_split_by_cluster_log
                else self.size_norm_counts.obs[clustering_key]
            )
            cluster_labels_counter = collections.Counter(cluster_labels.to_list())
            assert not valid_cluster.intersection(
                test_cluster
            ), "Valid and test clusters overlap"

            train_idx, valid_idx, test_idx = [], [], []
            for i, label in enumerate(cluster_labels):
                if label in valid_cluster:
                    valid_idx.append(i)
                elif label in test_cluster:
                    test_idx.append(i)
                else:
                    train_idx.append(i)

            assert train_idx, "Got empty training split"
            assert valid_idx, "Got empty validation split"
            assert test_idx, "Got empty test split"

            if self.mode == "train":
                idx_to_keep = train_idx
            elif self.mode == "valid":
                idx_to_keep = valid_idx
            elif self.mode == "test":
                idx_to_keep = test_idx
            else:
                raise ValueError(f"Unrecognzied data split mode: {self.mode}")

            self.data_raw = self.data_raw[idx_to_keep]
            if hasattr(self, "_size_norm_counts"):
                self._size_norm_counts = self._size_norm_counts[idx_to_keep]
                assert self._size_norm_counts.shape == self.data_raw.shape
            if hasattr(self, "_size_norm_log_counts"):
                self._size_norm_log_counts = self._size_norm_log_counts[idx_to_keep]
                assert self._size_norm_log_counts.shape == self.data_raw.shape
            logging.info(
                f"Built {self.mode} clustered data split with {self.data_raw.shape[0]} samples"
            )
        else:
            logging.info("Got data split for all data, skipping data splitting")

    def __sample_similar_cell(self, i, threshold=5, leakage=0.1) -> None:
        """
        Sample a similar cell for the ith cell
        Uses a very naive approach where we separately sample things above
        and below the threshold. 0.6 samples about 13.89 neighbors, 0.5 samples about 60.14
        Returns index of that similar cell
        """

        def exp_sample(sims):
            """Sample from the vector of similarities"""
            w = np.exp(sims)
            assert not np.any(np.isnan(w)), "Got NaN in exp(s)"
            assert np.sum(w) > 0, "Got a zero-vector of weights!"
            w_norm = w / np.sum(w)
            idx = np.random.choice(np.arange(len(w_norm)), p=w_norm)
            return idx

        assert self.cell_sim_mat is not None
        sim_scores = self.cell_sim_mat[i]
        high_scores = sim_scores[np.where(sim_scores > threshold)]
        low_scores = sim_scores[np.where(sim_scores <= threshold)]
        if np.random.random() < leakage:
            idx = exp_sample(low_scores)
        else:
            idx = exp_sample(high_scores)
        return idx

    @functools.lru_cache(32)
    def __get_chrom_idx(self) -> Dict[str, np.ndarray]:
        """Helper func for figuring out which feature indexes are on each chromosome"""
        chromosomes = sorted(
            list(set(self.data_raw.var["chrom"]))
        )  # Sort to guarantee consistent ordering
        chrom_to_idx = collections.OrderedDict()
        for chrom in chromosomes:
            chrom_to_idx[chrom] = np.where(self.data_raw.var["chrom"] == chrom)
        return chrom_to_idx

    def __get_chrom_split_features(self, i):
        """Given an index, get the features split by chromsome, returning in chromosome-sorted order"""
        if self.x_dropout:
            raise NotImplementedError
        features = torch.from_numpy(
            utils.ensure_arr(self.data_raw.X[i]).flatten()
        ).type(torch.FloatTensor)
        assert len(features.shape) == 1  # Assumes one dimensional vector of features

        chrom_to_idx = self.__get_chrom_idx()
        retval = tuple([features[indices] for _chrom, indices in chrom_to_idx.items()])
        if self.concat_outputs:
            retval = torch.cat(retval)
        return retval

    def __len__(self):
        """Number of examples"""
        return self.data_raw.n_obs

    def __getitem__(self, i):
        # TODO compatibility with slices
        expression_data = (
            torch.from_numpy(utils.ensure_arr(self.data_raw.X[i]).flatten()).type(
                torch.FloatTensor
            )
            if not self.split_by_chrom
            else self.__get_chrom_split_features(i)
        )
        if self.x_dropout and not self.split_by_chrom:
            # Apply dropout to the X input
            raise NotImplementedError

        # Handle case where we are shuffling y a la noise2noise
        # Only use shuffled indices if it is specifically enabled and if we are doing TRAINING
        # I.e. validation/test should never be shuffled
        y_idx = (
            self.__sample_similar_cell(i)
            if (self.sample_y and self.mode == "train")
            else i  # If not sampling y and training, return the same idx
        )
        if self.y_mode.endswith("raw_count"):
            target = torch.from_numpy(
                utils.ensure_arr(self.data_raw.raw.var_vector(y_idx))
            ).type(torch.FloatTensor)
        elif self.y_mode.endswith("size_norm"):
            target = torch.from_numpy(self.size_norm_counts.var_vector(y_idx)).type(
                torch.FloatTensor
            )
        elif self.y_mode == "x":
            target = torch.from_numpy(
                utils.ensure_arr(self.data_raw.X[y_idx]).flatten()
            ).type(torch.FloatTensor)
        else:
            raise NotImplementedError(f"Unrecognized y_mode: {self.y_mode}")
        if self.y_mode.startswith("log"):
            target = torch.log1p(target)  # scapy is also natural logaeritm of 1+x

        # Structure here is a series of inputs, followed by a fixed tuple of expected output
        retval = [expression_data]
        if self.return_sf:
            sf = self.size_factors[i]
            retval.append(sf)
        # Build expected truth
        if self.selfsupervise:
            if not self.return_pbulk:
                retval.append(target)
            else:  # Return both target and psuedobulk
                ith_cluster = self.data_raw.obs.iloc[i]["leiden"]
                pbulk = torch.from_numpy(
                    self.get_cluster_psuedobulk().var_vector(ith_cluster)
                ).type(torch.FloatTensor)
                retval.append((target, pbulk))
        elif self.return_pbulk:
            ith_cluster = self.data_raw.obs.iloc[i]["leiden"]
            pbulk = torch.from_numpy(
                self.get_cluster_psuedobulk().var_vector(ith_cluster)
            ).type(torch.FloatTensor)
            retval.append(pbulk)
        else:
            raise ValueError("Neither selfsupervise or retur_pbulk is specified")

        return tuple(retval)

    def get_per_chrom_feature_count(self) -> List[int]:
        """
        Return the number of features from each chromosome
        If we were to split a catted feature vector, we would split
        into these sizes
        """
        chrom_to_idx = self.__get_chrom_idx()
        return [len(indices[0]) for _chrom, indices in chrom_to_idx.items()]

    @property
    def size_norm_counts(self):
        """Computes and stores table of normalized counts w/ size factor adjustment and no other normalization"""
        if not hasattr(self, "_size_norm_counts"):
            self._size_norm_counts = self._set_size_norm_counts()
        assert self._size_norm_counts.shape == self.data_raw.shape
        return self._size_norm_counts

    def _set_size_norm_counts(self) -> AnnData:
        raw_counts_anndata = AnnData(
            scipy.sparse.csr_matrix(self.data_raw.raw.X),
            obs=pd.DataFrame(index=self.data_raw.obs_names),
            var=pd.DataFrame(index=self.data_raw.var_names),
        )
        sc.pp.normalize_total(raw_counts_anndata, inplace=True)
        # After normalizing, do clustering
        plot_utils.preprocess_anndata(
            raw_counts_anndata,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res,
        )
        return raw_counts_anndata

    @property
    def size_norm_log_counts(self):
        """Compute and store adata of counts with size factor adjustment and log normalization"""
        if not hasattr(self, "_size_norm_log_counts"):
            self._size_norm_log_counts = self._set_size_norm_log_counts()
        assert self._size_norm_log_counts.shape == self.data_raw.shape
        return self._size_norm_log_counts

    def _set_size_norm_log_counts(self) -> AnnData:
        retval = self.size_norm_counts.copy()  # Generates a new copy
        # Apply log to it
        sc.pp.log1p(retval, chunked=True, copy=False, chunk_size=10000)
        plot_utils.preprocess_anndata(
            retval,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res,
        )
        return retval

    @functools.lru_cache(4)
    def get_cluster_psuedobulk(self, mode="leiden", normalize=True):
        """
        Return a dictionary mapping each cluster label to the normalized psuedobulk
        estimate for that cluster
        If normalize is set to true, then we normalize such that each cluster's row
        sums to the median count from each cell
        """
        assert mode in self.data_raw.obs.columns
        cluster_labels = sorted(list(set(self.data_raw.obs[mode])))
        norm_counts = self.get_normalized_counts()
        aggs = []
        for cluster in cluster_labels:
            cluster_cells = np.where(self.data_raw.obs[mode] == cluster)
            pbulk = norm_counts.X[cluster_cells]
            pbulk_aggregate = np.sum(pbulk, axis=0, keepdims=True)
            if normalize:
                pbulk_aggregate = (
                    pbulk_aggregate
                    / np.sum(pbulk_aggregate)
                    * self.data_raw.uns["median_counts"]
                )
                assert np.isclose(
                    np.sum(pbulk_aggregate), self.data_raw.uns["median_counts"]
                )
            aggs.append(pbulk_aggregate)
        retval = AnnData(
            np.vstack(aggs), obs={mode: cluster_labels}, var=self.data_raw.var,
        )
        return retval


class SingleCellProteinDataset(Dataset):
    """
    Very simple dataset of CLR-transformed protein counts
    """

    # This is separate because it's so simple that tying it into SingleCellDataset
    # would be more confusing

    def __init__(
        self,
        raw_counts_files: Iterable[str],
        obs_names: List[str] = None,
        transpose: bool = True,
    ):
        self.raw_counts = utils.sc_read_multi_files(
            raw_counts_files,
            transpose=transpose,
            var_name_sanitization=lambda x: x.strip("_TotalSeqB"),
            feature_type="Antibody Capture",
        )
        assert np.all(
            utils.ensure_arr(self.raw_counts.X) >= 0
        ), "Got negative raw counts"

        # Subset
        if obs_names is not None:
            logging.info(f"Subsetting protein dataset to {len(obs_names)} cells")
            self.raw_counts = self.raw_counts[list(obs_names)]
        else:
            logging.info("Full protein dataset with no subsetting")

        # Normalize
        # Since this normalization is independently done PER CELL we don't have to
        # worry about doing this after we do subsetting
        clr_counts = clr_transform(utils.ensure_arr(self.raw_counts.X))
        # Use data_raw to be more similar to SingleCellDataset
        self.data_raw = AnnData(
            clr_counts, obs=self.raw_counts.obs, var=self.raw_counts.var,
        )
        self.data_raw.raw = self.raw_counts

    def __len__(self):
        return self.data_raw.n_obs

    def __getitem__(self, i: int):
        clr_vec = utils.ensure_arr(self.data_raw.X[i]).flatten()
        clr_tensor = torch.from_numpy(clr_vec).type(torch.FloatTensor)
        return clr_tensor, clr_tensor


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
        assert len(dataset_x) == len(
            dataset_y
        ), f"Paired dataset must have the same length"
        self.flat_mode = flat_mode

        self.obs_names = None
        if hasattr(dataset_x, "data_raw") and hasattr(dataset_y, "data_raw"):
            logging.info("Checking obs names for two input datasets")
            x_obs_names = dataset_x.data_raw.obs_names
            y_obs_names = dataset_y.data_raw.obs_names
            for i, (x, y) in enumerate(zip(x_obs_names, y_obs_names)):
                if x != y:
                    raise ValueError(
                        f"Datasets have a different label at the {i}th index: {x} {y}"
                    )
            self.obs_names = list(x_obs_names)
        elif hasattr(dataset_x, "data_raw"):
            self.obs_names = list(dataset_x.data_raw.obs_names)
        elif hasattr(dataset_y, "data_raw"):
            self.obs_names = list(dataset_y.data_raw.obs_names)
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
        x_pair = (self.dataset_x[i][0], self.dataset_y[i][0])
        y_pair = (self.dataset_x[i][1], self.dataset_y[i][1])
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
        self.sc_dataset = sc_dataset
        assert input_mode in ["RNA", "ATAC"], f"Unrecognized mode: {input_mode}"
        rna_encoded, atac_encoded = model.get_encoded_layer(sc_dataset)
        if input_mode == "RNA":
            self.encoded = rna_encoded
        else:
            self.encoded = atac_encoded

    def __len__(self):
        return self.encoded.shape[0]

    def __getitem__(self, idx: int):
        """Returns the idx-th item as (encoded(x), y)"""
        x_orig = self.sc_dataset[idx]
        enc = self.encoded[idx]
        return torch.from_numpy(enc).type(torch.FloatTensor), x_orig[-1]


class SingleCellRnaDataset(Dataset):
    """
    Loads in single-cell RNA-seq dataset from Duren et al.
    https://doi.org/10.1073/pnas.1805681115
    """

    def __init__(
        self,
        mode: str = "train",
        selfsupervise: bool = True,
        normalize: bool = True,
        raw_count_as_y=True,
        return_sf=True,
    ):
        """
        If selfsupervise is True, the y label for each output is input itself
        If log2trans is true, log2-transform the expression values to have a better behaved range
        If g_order is true, reorder gene features by genomic location in the mouse genome
        """
        assert mode in ["train", "valid", "test", "all"], f"Unrecognized mode: {mode}"
        self.mode = mode
        self.selfsupervise = selfsupervise
        self.raw_count_as_y = raw_count_as_y
        self.return_sf = return_sf
        if not self.selfsupervise:
            raise NotImplementedError(
                f"scRNA has defined labels, must be self supervised"
            )

        self.data_raw = sc.read_text(
            os.path.join(SC_RNA_ATAC_DIR, "GSE115968_scRNA-seq_RA_D4.txt.gz"),
            delimiter="\t",
        ).T  # Data needs to be transposed
        if normalize:
            self.data_raw = adata_utils.normalize_count_table(
                self.data_raw, size_factors=True, normalize=True, log_trans=True
            )

        if self.mode != "all":  # Drop data based on which set we're getting
            indices = np.arange(self.data_raw.n_obs)
            (
                indices_train,
                indices_valid,
                indices_test,
            ) = shuffle_indices_train_valid_test(indices, shuffle=True, seed=1234)
            if self.mode == "train":
                self.data_raw = self.data_raw[indices_train]
            elif self.mode == "valid":
                self.data_raw = self.data_raw[indices_valid]
            elif self.mode == "test":
                self.data_raw = self.data_raw[indices_test]
            else:
                raise ValueError(f"Unrecognized mode: {self.mode}")
        assert not np.any(pd.isnull(self.data_raw))

        self.data = torch.from_numpy(self.data_raw.X).type(torch.FloatTensor)
        self.size_factors = torch.from_numpy(
            self.data_raw.obs.size_factors.values
        ).type(torch.FloatTensor)
        self.data_counts = torch.from_numpy(self.data_raw.raw.X).type(torch.FloatTensor)

    def __len__(self):
        """Number of examples"""
        return self.data_raw.n_obs

    def __getitem__(self, i):
        """ith example"""
        expression_data = self.data[i]
        target = self.data[i] if not self.raw_count_as_y else self.data_counts[i]

        if self.return_sf:
            sf = self.size_factors[i]
            return expression_data, sf, target
        else:
            return expression_data, target


class SingleCellAtacDataset(Dataset):
    """
    Loads in single-cell ATAC-seq dataset from Duren et al.
    https://doi.org/10.1073/pnas.1805681115
    """

    def __init__(
        self, mode: str = "train", selfsupervise: bool = True, log2trans: bool = True
    ):
        assert mode in ["train", "valid", "test", "all"], f"Unrecognized mode: {mode}"
        self.mode = mode
        self.selfsupervise = selfsupervise
        if not self.selfsupervise:
            raise NotImplementedError(
                f"scATAC has defined labels, must be self supervised"
            )

        self.data_raw = pd.read_csv(
            os.path.join(SC_RNA_ATAC_DIR, "GSE115970_scATAC-seq_RA_D4.txt.gz"),
            delimiter="\t",
            index_col=0,
        )
        if log2trans:
            self.data_raw = np.log2(self.data_raw + 1)  # Add 1 so log(0) doesn't error
        # Drop subset of data based on train/valid/test
        if self.mode != "all":
            indices = np.arange(len(self.data_raw.columns))
            np.random.seed(1234)  # For reproducible subsampling
            np.random.shuffle(indices)  # Shuffles inplace
            num_train = int(round(len(self.data_raw.columns) * 0.7))
            num_valid = int(round(len(self.data_raw.columns) * 0.15))
            num_test = len(self.data_raw.columns) - num_train - num_valid
            assert num_train > 0 and num_valid > 0 and num_test > 0
            assert num_train + num_valid + num_test == len(self.data_raw.columns)
            indices_train = indices[:num_train]
            indices_valid = indices[num_train : num_train + num_valid]
            indices_test = indices[-num_test:]
            if self.mode == "train":
                self.data_raw = self.data_raw.iloc[:, indices_train]
            elif self.mode == "valid":
                self.data_raw = self.data_raw.iloc[:, indices_valid]
            elif self.mode == "test":
                self.data_raw = self.data_raw.iloc[:, indices_test]
            else:
                raise ValueError(f"Unrecognized mode: {self.mode}")

        assert not np.any(pd.isnull(self.data_raw))

        self.feature_names = self.data_raw.index
        self.sample_names = self.data_raw.columns
        self.data = torch.from_numpy(self.data_raw.values.T).type(torch.FloatTensor)

    def __len__(self):
        """Number of examples"""
        return len(self.sample_names)

    def __getitem__(self, i):
        """ith example"""
        expression_data = self.data[i]
        target = self.data[i]

        return expression_data, target


class SimSingleCellRnaDataset(Dataset):
    """Loads in the simulated single cell dataset"""

    def __init__(
        self,
        counts_fname: str,
        labels_fname: str = None,
        mode: str = "train",
        normalize: bool = True,
        selfsupervise: bool = True,
        return_sf=True,
        y_mode: str = "size_norm",
    ):
        assert mode in ["all", "train", "valid", "test"]
        self.mode = mode
        self.selfsupervise = selfsupervise
        self.y_mode = y_mode
        self.return_sf = return_sf

        self.data_raw = sc.read_csv(counts_fname, first_column_names=True)
        if normalize:
            # Note that we normalize the ENTIRE DATASET as a whole
            # We don't subset till later, so all data splits have the same normalization
            self.data_raw = adata_utils.normalize_count_table(self.data_raw)

        self.labels = None
        if labels_fname:
            labels_df = pd.read_csv(labels_fname)
            labels_raw = list(labels_df["Group"])
            _uniq, self.labels = np.lib.arraysetops.unique(
                labels_raw, return_inverse=True
            )

        if self.mode != "all":
            (
                indices_train,
                indices_valid,
                indices_test,
            ) = shuffle_indices_train_valid_test(
                np.arange(self.data_raw.n_obs), test=0, valid=0.2,
            )
            if self.mode == "train":
                self.data_raw = self.data_raw[indices_train]
                if self.labels is not None:
                    self.labels = self.labels[indices_train]
            elif self.mode == "valid":
                self.data_raw = self.data_raw[indices_valid]
                if self.labels is not None:
                    self.labels = self.labels[indices_valid]
            elif self.mode == "test":
                self.data_raw = self.data_raw[indices_test]
                if self.labels is not None:
                    self.labels = self.labels[indices_test]
            else:
                raise ValueError(f"Unrecognized mode: {self.mode}")
        assert not np.any(pd.isnull(self.data_raw))

        self.features_names = self.data_raw.var_names
        self.sample_names = self.data_raw.obs_names
        self.data = torch.from_numpy(self.data_raw.X).type(torch.FloatTensor)
        self.data_counts = torch.from_numpy(self.data_raw.raw.X).type(torch.FloatTensor)
        self.size_factors = torch.from_numpy(
            self.data_raw.obs.size_factors.values
        ).type(torch.FloatTensor)
        self.size_norm_counts = self.get_normalized_counts()
        if self.labels is not None:
            self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)
            assert len(self.labels) == len(
                self.data_raw
            ), f"Got mismatched sizes {len(self.labels)} {len(self.data_raw)}"

    def get_normalized_counts(self):
        """Return table of normalized counts w/ size factor adjustment and no other normalization"""
        raw_counts = self.data_raw.raw.X
        raw_counts_anndata = AnnData(
            raw_counts, obs=self.data_raw.obs, var=self.data_raw.var
        )
        sc.pp.normalize_total(raw_counts_anndata, inplace=True)
        return raw_counts_anndata

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, i):
        expression_data = self.data[i]
        # If we need raw counts, extract from sparse array
        if self.y_mode.endswith("raw_count"):
            target = self.data_counts[i]
        elif self.y_mode.endswith("size_norm"):
            target = torch.from_numpy(self.size_norm_counts.X[i]).type(
                torch.FloatTensor
            )
        elif self.y_mode == "x":
            target = expression_data
        else:
            raise NotImplementedError(f"Unrecognized y_mode: {self.y_mode}")

        if self.y_mode.startswith("log"):
            target = torch.log1p(target)  # scapy is also natural logaeritm of 1+x

        if self.return_sf:
            sf = self.size_factors[i]
            return expression_data, sf, target
        return expression_data, target


def sparse_var(x: Union[scipy.sparse.csr_matrix, scipy.sparse.csc_matrix], axis=0):
    """
    Return variance of sparse matrix
    """
    assert isinstance(x, (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix))

    retval = []
    if axis == 0:
        x = x.tocsc()
        for i in range(x.shape[1]):
            retval.append(np.var(x.getcol(i).toarray()))
    elif axis == 1:
        x = x.tocsr()
        for j in range(x.shape[0]):
            retval.append(np.var(x.getrow(i).toarray()))
    else:
        raise ValueError("Axis should be 0 or 1")
    return np.array(retval)


def _cell_distance_matrix_helper(
    i: int, j: int, mat: np.ndarray, dist_func=scipy.spatial.distance.cosine
):
    """
    Helper function for computing pairwise distances
    Return tuple of ((i, j), retval)
    """
    if isinstance(mat, np.ndarray):
        d = dist_func(mat[i], mat[j])
    else:
        d = dist_func(mat[i].toarray().flatten(), mat[j].toarray().flatten())
    return (i, j), d


def cell_distance_matrix(
    mat: np.ndarray,
    top_pcs: int = 0,
    dist_func=scipy.spatial.distance.cosine,
    threads: int = 12,
) -> np.ndarray:
    """
    Return pairwise cell distance (i.e. smaller values indicate greater similarity)
    Distance function should be symmetric
    """
    if top_pcs:
        raise NotImplementedError
    assert len(mat.shape) == 2
    if isinstance(mat, AnnData):
        mat = mat.X
    # assert isinstance(mat, np.ndarray)
    n_obs, n_var = mat.shape

    pfunc = functools.partial(
        _cell_distance_matrix_helper, mat=mat, dist_func=dist_func
    )
    pool = multiprocessing.Pool(threads)
    mapped_values = pool.starmap(pfunc, itertools.product(range(n_obs), range(n_obs)))
    pool.close()
    pool.join()

    retval = np.zeros((n_obs, n_obs), dtype=float)
    for (i, j), s in mapped_values:
        retval[i, j] = s
    return retval


def euclidean_sim_matrix(mat: np.ndarray):
    """
    Given a matrix where rows denote observations, calculate a square matrix of similarities
    Larger values indicate greater similarity
    """
    assert (
        len(mat.shape) == 2
    ), f"Input must be 2 dimensiona, but got {len(mat.shape)} dimensions"
    if isinstance(mat, AnnData):
        mat = mat.X  # We only read data here so this is ok
    assert isinstance(
        mat, np.ndarray
    ), f"Could not convert input of type {type(mat)} into np array"
    n_obs = mat.shape[0]
    retval = np.zeros((n_obs, n_obs), dtype=float)

    for i in range(n_obs):
        for j in range(i):
            s = np.linalg.norm(mat[i] - mat[j], ord=None)
            retval[i, j] = s
            retval[j, i] = s
    retval = retval / (np.max(retval))
    # Up to this point the values here are distances, where smaller = more similar
    # for i in range(n_obs):
    #     retval[i, i] = 1.0
    # Set 0 to be some minimum distance
    retval = np.divide(1, retval, where=retval > 0)
    retval[retval == 0] = np.max(retval)
    retval[np.isnan(retval)] = np.max(retval)
    return retval


def shuffle_indices_train_valid_test(
    idx, valid: float = 0.15, test: float = 0.15, shuffle=True, seed=1234
):
    """Given an array of indices, return them partitioned into train, valid, and test indices"""
    np.random.seed(1234)  # For reproducible subsampling
    indices = np.copy(idx)  # Make a copy because shuffling occurs in place
    np.random.shuffle(indices)  # Shuffles inplace
    num_valid = int(round(len(indices) * valid)) if valid > 0 else 0
    num_test = int(round(len(indices) * test)) if test > 0 else 0
    num_train = len(indices) - num_valid - num_test
    assert num_train > 0 and num_valid >= 0 and num_test >= 0
    assert num_train + num_valid + num_test == len(
        indices
    ), f"Got mismatched counts: {num_train} + {num_valid} + {num_test} != {len(indices)}"

    indices_train = indices[:num_train]
    indices_valid = indices[num_train : num_train + num_valid]
    indices_test = indices[-num_test:]

    return indices_train, indices_valid, indices_test


@functools.lru_cache(maxsize=2, typed=True)
def read_gtf_gene_to_pos(
    fname: str = MM9_GTF,
    acceptable_types: List[str] = None,
    addtl_attr_filters: dict = None,
    extend_upstream: int = 0,
    extend_downstream: int = 0,
) -> Dict[str, Tuple[str, int, int]]:
    """
    Given a gtf file, read it in and return as a ordered dictionary mapping genes to genomic ranges
    Ordering is done by chromosome then by position
    """
    # https://uswest.ensembl.org/info/website/upload/gff.html
    gene_to_positions = collections.defaultdict(list)
    gene_to_chroms = collections.defaultdict(set)

    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if line.startswith(b"#"):
                continue
            line = line.decode()
            (
                chrom,
                entry_type,
                entry_class,
                start,
                end,
                score,
                strand,
                frame,
                attrs,
            ) = line.strip().split("\t")
            assert strand in ("+", "-")
            if acceptable_types and entry_type not in acceptable_types:
                continue
            attr_dict = dict(
                [t.strip().split(" ", 1) for t in attrs.strip().split(";") if t]
            )
            if addtl_attr_filters:
                tripped_attr_filter = False
                for k, v in addtl_attr_filters.items():
                    if k in attr_dict:
                        if isinstance(v, str):
                            if v != attr_dict[k].strip('"'):
                                tripped_attr_filter = True
                                break
                        else:
                            raise NotImplementedError
                if tripped_attr_filter:
                    continue
            gene = attr_dict["gene_name"].strip('"')
            start = int(start)
            end = int(end)
            assert (
                start <= end
            ), f"Start {start} is not less than end {end} for {gene} with strand {strand}"
            if extend_upstream:
                if strand == "+":
                    start -= extend_upstream
                else:
                    end += extend_upstream
            if extend_downstream:
                if strand == "+":
                    end += extend_downstream
                else:
                    start -= extend_downstream

            gene_to_positions[gene].append(start)
            gene_to_positions[gene].append(end)
            gene_to_chroms[gene].add(chrom)

    slist = sortedcontainers.SortedList()
    for gene, chroms in gene_to_chroms.items():
        if len(chroms) != 1:
            logging.warn(
                f"Got multiple chromosomes for gene {gene}: {chroms}, skipping"
            )
            continue
        positions = gene_to_positions[gene]
        t = (chroms.pop(), min(positions), max(positions), gene)
        slist.add(t)

    retval = collections.OrderedDict()
    for chrom, start, stop, gene in slist:
        retval[gene] = (chrom, start, stop)
    return retval


def gene_pos_dict_to_range(gene_pos_dict: dict, remove_overlaps: bool = False):
    """
    Converts the dictionary of genes to positions to a intervaltree
    of chromsomes to positions, each corresponding to a gene
    """
    retval = collections.defaultdict(
        intervaltree.IntervalTree
    )  # Chromosomes to genomic ranges
    genes = list(gene_pos_dict.keys())  # Ordered
    for gene in genes:
        chrom, start, stop = gene_pos_dict[gene]
        retval[chrom][start:stop] = gene

    if remove_overlaps:
        raise NotImplementedError

    return retval


def reorder_genes_by_pos(
    genes, gtf_file=MM9_GTF, return_genes=False, return_chrom=False
):
    """Reorders list of genes by their genomic coordinate in the given gtf"""
    genes_set = set(genes)
    genes_list = list(genes)
    assert len(genes_set) == len(genes), f"Got duplicates in genes"

    genes_to_pos = read_gtf_gene_to_pos(gtf_file)
    genes_intersection = [
        g for g in genes_to_pos if g in genes_set
    ]  # In order of position
    logging.info(f"{len(genes_intersection)} genes with known positions")
    genes_to_idx = {}
    for i, g in enumerate(genes_intersection):
        genes_to_idx[g] = i  # Record position of each gene in the ordered list

    slist = sortedcontainers.SortedList()  # Insert into a sorted list
    skip_count = 0
    for gene in genes_intersection:
        slist.add((genes_to_idx[gene], gene))

    genes_reordered = [g for _i, g in slist]
    if return_genes:  # Return the genes themselves in order
        retval = genes_reordered
    else:  # Return the indices needed to rearrange the genes in order
        retval = np.array([genes_list.index(gene) for gene in genes_reordered])
    chroms = [genes_to_pos[g][0] for _i, g in slist]
    assert len(chroms) == len(retval)

    if return_chrom:
        retval = (retval, chroms)
    return retval


def get_chrom_from_genes(genes: List[str], gtf_file=MM9_GTF):
    """
    Given a list of genes, return a list of chromosomes that those genes are on
    For missing: NA
    """
    gene_to_pos = read_gtf_gene_to_pos(gtf_file)
    retval = [gene_to_pos[gene][0] if gene in gene_to_pos else "NA" for gene in genes]
    return retval


def get_chrom_from_intervals(intervals: List[str], strip_chr: bool = True):
    """
    Given a list of intervals, return a list of chromosomes that those are on

    >>> get_chrom_from_intervals(['chr2:100-200', 'chr3:100-222'])
    ['2', '3']
    """
    retval = [interval.split(":")[0].strip() for interval in intervals]
    if strip_chr:
        retval = [chrom.strip("chr") for chrom in retval]
    return retval


def get_shared_samples(
    file1, file2, key="sample", reader=pd.read_csv, **reader_kwargs
) -> set:
    """Return the shared samples between tables described by file1 and file2 as a set"""
    table1 = reader(file1, **reader_kwargs)
    table2 = reader(file2, **reader_kwargs)

    samples1 = set(table1[key])
    samples2 = set(table2[key])

    retval = samples1.intersection(samples2)
    return retval


def _read_mtx_helper(lines, shape, dtype):
    """Helper function for read_mtx"""
    retval = scipy.sparse.dok_matrix(shape, dtype=dtype)
    for line in lines:
        x, y, v = line.decode().strip().split()
        retval[int(x) - 1, int(y) - 1] = dtype(v)
    return retval


def read_mtx(fname, dtype=int, chunksize=100000):
    """Read the mtx file"""
    # Reads in the full file, then splits the (i, j x) values into
    # chunks, convert each chunk into a sparse matrix in parallel,
    # and add them all up for final output
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        header = source.readline().decode()
        assert header.startswith("%")
        nrows, ncols, nelems = map(int, source.readline().strip().decode().split())
        data_lines = source.readlines()  # Skips the first two lines

    # Generator for chunks of data
    data_lines_chunks = (
        data_lines[i : i + chunksize] for i in range(0, len(data_lines), chunksize)
    )
    pfunc = functools.partial(_read_mtx_helper, shape=(nrows, ncols), dtype=dtype)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    matrices = pool.imap_unordered(pfunc, data_lines_chunks, chunksize=1)
    pool.close()
    pool.join()

    retval = scipy.sparse.csr_matrix((nrows, ncols), dtype=dtype)
    for mat in matrices:
        retval += mat
    retval = AnnData(retval, obs=None, var=None)
    return retval


def interval_string_to_tuple(x: str) -> Tuple[str, int, int]:
    tokens = x.split(":")
    assert len(tokens) == 2, f"Malformed interval string: {x}"
    chrom, interval = tokens
    if not chrom.startswith("chr"):
        logging.warn(f"Got noncanonical chromsome in {x}")
    start, stop = map(int, interval.split("-"))
    assert start < stop, f"Got invalid interval span: {x}"
    return (chrom, start, stop)


def tuple_to_interval_string(t: Tuple[str, int, int]) -> str:
    return f"{t[0]}:{t[1]}-{t[2]}"


def interval_strings_to_itree(
    interval_strings: List[str],
) -> Dict[str, intervaltree.IntervalTree]:
    """
    Given a list of interval strings, return an itree per chromosome
    The data field is the index of the interval in the original list
    """
    interval_tuples = [interval_string_to_tuple(x) for x in interval_strings]
    retval = collections.defaultdict(intervaltree.IntervalTree)
    for i, (chrom, start, stop) in enumerate(interval_tuples):
        retval[chrom][start:stop] = i
    return retval


def get_indices_to_combine(genomic_intervals: List[str], interval: int = 1000):
    """
    Given a list of *sorted* genomic intervals in string format e.g. ["chr1:100-200", "chr1:300-400"]
    Return a list of indices to combine to create new intervals of given size, as well as new interval
    strings
    """
    # First convert to a list of tuples (chr, start, stop)
    interval_tuples = [interval_string_to_tuple(x) for x in genomic_intervals]

    curr_chrom, curr_start, _ = interval_tuples[0]  # Initial valiues
    curr_indices, ret_indices, ret_names = [], [], []
    curr_end = curr_start + interval
    for i, (chrom, start, stop) in enumerate(interval_tuples):
        if (
            chrom != curr_chrom or stop > curr_end
        ):  # Reset on new chromosome or extending past interval
            ret_indices.append(curr_indices)
            ret_names.append(
                tuple_to_interval_string((curr_chrom, curr_start, curr_end))
            )
            curr_chrom, curr_start = chrom, start
            curr_end = curr_start + interval
            curr_indices = []
        assert start >= curr_start, f"Got funky coord: {chrom} {start} {stop}"
        assert stop > start
        curr_indices.append(i)

    ret_indices.append(curr_indices)
    ret_names.append(tuple_to_interval_string((curr_chrom, curr_start, curr_end)))

    return ret_indices, ret_names


def get_indices_to_form_target_intervals(
    genomic_intervals: List[str], target_intervals: List[str]
) -> List[List[int]]:
    """
    Given a list of genomic intervals in string format, and a target set of similar intervals
    Return a list of indices to combine to map into the target
    """
    source_itree = interval_strings_to_itree(genomic_intervals)
    target_intervals = [interval_string_to_tuple(x) for x in target_intervals]

    retval = []
    for chrom, start, stop in target_intervals:
        overlaps = source_itree[chrom].overlap(start, stop)
        retval.append([o.data for o in overlaps])
    return retval


def combine_array_cols_by_idx(arr, idx, combine_func=np.sum) -> scipy.sparse.csr_matrix:
    """Given an array and indices, combine the specified columns, returning as a csr matrix"""
    if isinstance(arr, np.ndarray):
        arr = scipy.sparse.csc_matrix(arr)
    elif isinstance(arr, pd.DataFrame):
        arr = scipy.sparse.csc_matrix(arr.to_numpy(copy=True))
    elif isinstance(arr, scipy.sparse.csr_matrix):
        arr = arr.tocsc()
    elif isinstance(arr, scipy.sparse.csc_matrix):
        pass
    else:
        raise TypeError(f"Cannot combine array cols for type {type(arr)}")

    new_cols = []
    for indices in idx:
        if not indices:
            next_col = scipy.sparse.csc_matrix(np.zeros((arr.shape[0], 1)))
        else:
            col_set = np.hstack([arr.getcol(i).toarray() for i in indices])
            next_col = scipy.sparse.csc_matrix(
                combine_func(col_set, axis=1, keepdims=True)
            )
        new_cols.append(next_col)
    new_mat = scipy.sparse.hstack(new_cols)
    new_mat_sparse = scipy.sparse.csr_matrix(new_mat)
    assert (
        len(new_mat_sparse.shape) == 2
    ), f"Returned matrix is expected to be 2 dimensional, but got shape {new_mat_sparse.shape}"
    # print(arr.shape, new_mat_sparse.shape)
    return new_mat_sparse


def combine_by_proximity(
    arr, gtf_file=MM9_GTF, start_extension: int = 10000, stop_extension: int = 10000
):
    def find_nearest(query: tuple, arr):
        """Find the index of the item in array closest to query"""
        # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
        start_distances = np.abs(query[0] - arr)
        stop_distances = np.abs(query[1] - arr)
        min_distances = np.minimum(start_distances, stop_distances)
        idx = np.argmin(min_distances)
        return idx

    if isinstance(arr, AnnData):
        d = arr.X if isinstance(arr.X, np.ndarray) else arr.X.toarray()
        arr = pd.DataFrame(d, index=arr.obs_names, columns=arr.var_names,)
    assert isinstance(arr, pd.DataFrame)

    gene_to_pos = read_gtf_gene_to_pos(
        gtf_file,
        acceptable_types=["protein_coding"],
        addtl_attr_filters={"gene_biotype": "protein_coding"},
    )
    genomic_ranges_to_gene = gene_pos_dict_to_range(gene_to_pos)
    genes_to_intervals = collections.defaultdict(list)  # Maps to the ith intervals
    unassigned_count = 0
    for i, g_interval in enumerate(arr.columns):
        chrom, g_range = g_interval.split(":")
        chrom_stripped = chrom.strip("chr")
        if chrom_stripped not in genomic_ranges_to_gene:
            logging.warn("Chromoome not found: {chrom}")

        start, stop = map(int, g_range.split("-"))
        assert start < stop, f"Got illegal genomic range: {g_interval}"
        start_extended, stop_extended = start - start_extension, stop + stop_extension

        overlapping_genes = list(
            genomic_ranges_to_gene[chrom_stripped][start_extended:stop_extended]
        )
        if overlapping_genes:
            if len(overlapping_genes) == 1:
                hit = overlapping_genes.pop()  # There is only one hit so we extract it
                hit_gene = hit.data
            else:  # Pick the closer hit
                hit_starts = np.array([h.begin for h in overlapping_genes])
                hit_ends = np.array([h.end for h in overlapping_genes])
                hit_pos_combined = np.concatenate((hit_starts, hit_ends))
                hit_genes = [h.data for h in overlapping_genes] * 2
                nearest_idx = find_nearest(
                    (start_extended, stop_extended), hit_pos_combined
                )
                hit_gene = hit_genes[nearest_idx]
            genes_to_intervals[hit_gene].append(i)
        else:
            unassigned_count += 1
    logging.warn(f"{unassigned_count}/{len(arr.columns)} peaks not assigned to a gene")
    genes = list(genes_to_intervals.keys())
    aggregated = combine_array_cols_by_idx(arr, [genes_to_intervals[g] for g in genes])
    return aggregated, genes


def _tuple_merger(x: tuple, y: tuple, token: str = ";"):
    """
    Given two tuples, update their fields

    >>> _tuple_merger( ('chr11_117985551_117988553', 'IL10RA', '0', 'promoter'), ('chr11_117985551_117988553', 'IL10RA', '0', 'promoter') )
    ('chr11_117985551_117988553', 'IL10RA', '0', 'promoter')
    >>> _tuple_merger( ('chr11_117985551_117988553', 'FOO', '0', 'promoter'), ('chr11_117985551_117988553', 'IL10RA', '0', 'promoter') )
    ('chr11_117985551_117988553', 'FOO;IL10RA', '0', 'promoter')
    """
    assert len(x) == len(y)
    retval = []
    for i, j in zip(x, y):
        i_tokens = set(i.split(token))
        j_tokens = set(j.split(token))
        new = token.join(sorted(list(i_tokens.union(j_tokens))))
        retval.append(new)

    return tuple(retval)


def harmonize_atac_intervals(
    intervals_1: List[str], intervals_2: List[str]
) -> List[str]:
    """
    Given two files describing intervals, harmonize them by merging overlapping
    intervals for each chromosome
    """

    def interval_list_to_itree(l: List[str],) -> Dict[str, intervaltree.IntervalTree]:
        """convert the dataframe to a intervaltree"""
        retval = collections.defaultdict(intervaltree.IntervalTree)
        for s in l:
            chrom, span = s.split(":")
            start, stop = map(int, span.split("-"))
            retval[chrom][start:stop] = None
        return retval

    itree1 = interval_list_to_itree(intervals_1)
    itree2 = interval_list_to_itree(intervals_2)

    # Merge the two inputs
    merged_itree = {}
    for chrom in itree1.keys():
        if chrom not in itree2:
            continue
        combined = itree1[chrom] | itree2[chrom]
        combined.merge_overlaps()
        merged_itree[chrom] = combined

    retval = []
    interval_spans = []
    for chrom, intervals in merged_itree.items():
        for i in sorted(intervals):
            interval_spans.append(i.end - i.begin)
            i_str = f"{chrom}:{i.begin}-{i.end}"
            retval.append(i_str)

    logging.info(
        f"Average/SD interval size after merging: {np.mean(interval_spans):.4f} {np.std(interval_spans):.4f}"
    )
    return retval


def liftover_intervals(
    intervals: List[str],
    chain_file: str = os.path.join(DATA_DIR, "hg19ToHg38.over.chain.gz"),
) -> Tuple[List[str], List[str]]:
    """
    Given a list of intervals in format chr:start-stop, lift them over acccording to the chain file
    and return the new coordinates, as well as those that were unmapped.
    This does NOT reorder the regions
    >>> liftover_intervals(["chr1:10134-10369", "chr1:804533-805145"])
    (['chr1:10134-10369', 'chr1:869153-869765'], [])
    >>> liftover_intervals(["chr1:804533-805145", "chr1:10134-10369"])
    (['chr1:869153-869765', 'chr1:10134-10369'], [])
    """
    assert os.path.isfile(chain_file), f"Cannot find chain file: {chain_file}"
    liftover_binary = shutil.which("liftOver")
    assert liftover_binary, "Cannot find liftover binary"

    # Write to a temporary file, pass that temporary file into liftover, read output
    tmp_id = random.randint(1, 10000)
    tmp_fname = f"liftover_intermediate_{tmp_id}.txt"
    tmp_out_fname = f"liftover_output_{tmp_id}.txt"
    tmp_unmapped_fname = f"liftover_unmapped_{tmp_id}.txt"

    with open(tmp_fname, "w") as sink:
        sink.write("\n".join(intervals) + "\n")

    cmd = f"{liftover_binary} {tmp_fname} {chain_file} {tmp_out_fname} {tmp_unmapped_fname} -positions"
    retcode = subprocess.call(shlex.split(cmd))
    assert retcode == 0, f"liftover exited with error code {retcode}"

    # Read in the output
    with open(tmp_out_fname) as source:
        retval = [l.strip() for l in source]
    with open(tmp_unmapped_fname) as source:
        unmapped = [l.strip() for l in source if not l.startswith("#")]
    assert len(retval) + len(unmapped) == len(intervals)

    if unmapped:
        logging.warning(f"Found unmapped regions: {len(unmapped)}")

    # Remove temporary files
    os.remove(tmp_fname)
    os.remove(tmp_out_fname)
    os.remove(tmp_unmapped_fname)
    # Fix the leftover intermediate file
    # This cannot be run in parallel mode
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bedmapped"):
        os.remove(fname)
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bedunmapped"):
        os.remove(fname)
    for fname in glob.glob(f"liftOver_{platform.node()}_*.bed"):
        os.remove(fname)

    return retval, unmapped


def liftover_atac_adata(
    adata: AnnData, chain_file: str = os.path.join(DATA_DIR, "hg19ToHg38.over.chain.gz")
) -> AnnData:
    """
    Lifts over the ATAC bins
    """
    lifted_var_names, unmapped_var_names = liftover_intervals(
        list(adata.var_names), chain_file=chain_file
    )
    keep_var_names = [n for n in adata.var_names if n not in unmapped_var_names]
    adata_trimmed = adata[:, keep_var_names]
    adata_trimmed.var_names = lifted_var_names
    return adata_trimmed


def repool_atac_bins(x: AnnData, target_bins: List[str]) -> AnnData:
    """
    Re-pool data from x to match the given target bins, summing overlapping entries
    """
    # TODO compare against __pool_features and de-duplicate code
    idx = get_indices_to_form_target_intervals(
        x.var.index, target_intervals=target_bins
    )
    data_raw_aggregated = scipy.sparse.csr_matrix(combine_array_cols_by_idx(x.X, idx,))
    retval = AnnData(
        data_raw_aggregated, obs=x.obs, var=pd.DataFrame(index=target_bins),
    )
    return retval


def atac_intervals_to_bins_per_chrom(intervals: Iterable[str]) -> List[int]:
    """"""
    cnt = collections.Counter([i.split(":")[0] for i in intervals])
    retval = {}
    for k in sorted(list(cnt.keys())):
        retval[k] = cnt[k]
    return list(retval.values())


def read_diff_exp_genes_to_marker_genes(
    fname: str,
    geq_filt_dict: Dict[str, float] = {},
    leq_filt_dict: Dict[str, float] = {},
) -> Dict[str, List[str]]:
    """
    Given a file of differentially expressed genes per cluster
    return a mapping from cluster to its signature genes
    geq_filt_dict is a list of key and values for which the
    table has to surpass in order to be included, and vice versa
    for the leq_filt_dict
    """
    df = pd.read_csv(fname)
    retval = collections.defaultdict(list)
    for _i, row in df.iterrows():
        skip = False
        for key, val in geq_filt_dict.items():
            if not row[key] >= val:
                skip = True
        for key, val in leq_filt_dict.items():
            if not row[key] <= val:
                skip = True
        if skip:
            continue
        retval[row["Cluster"]].append(row["Gene"])
    return retval


def clr_transform(x: np.ndarray, add_pseudocount: bool = True) -> np.ndarray:
    """
    Centered logratio transformation. Useful for protein data, but

    >>> clr_transform(np.array([0.1, 0.3, 0.4, 0.2]), add_pseudocount=False)
    array([-0.79451346,  0.30409883,  0.5917809 , -0.10136628])
    >>> clr_transform(np.array([[0.1, 0.3, 0.4, 0.2], [0.1, 0.3, 0.4, 0.2]]), add_pseudocount=False)
    array([[-0.79451346,  0.30409883,  0.5917809 , -0.10136628],
           [-0.79451346,  0.30409883,  0.5917809 , -0.10136628]])
    """
    assert isinstance(x, np.ndarray)
    if add_pseudocount:
        x = x + 1.0
    if len(x.shape) == 1:
        denom = scipy.stats.mstats.gmean(x)
        retval = np.log(x / denom)
    elif len(x.shape) == 2:
        # Assumes that each row is an independent observation
        # and that columns denote features
        per_row = []
        for i in range(x.shape[0]):
            denom = scipy.stats.mstats.gmean(x[i])
            row = np.log(x[i] / denom)
            per_row.append(row)
        assert len(per_row) == x.shape[0]
        retval = np.stack(per_row)
    else:
        raise ValueError(f"Cannot CLR transform array with {len(x.shape)} dims")
    return retval


def main():
    """On the fly testing"""
    # Create the PBMC ATAC merged bins across replicates
    temp1 = utils.sc_read_10x_h5_ft_type(
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep1_filtered_feature_bc_matrix.h5"),
        "Peaks",
    )
    temp2 = utils.sc_read_10x_h5_ft_type(
        os.path.join(TENX_PBMC_DATA_DIR, "pbmc_rep2_filtered_feature_bc_matrix.h5"),
        "Peaks",
    )
    pbmc_merged_atac_bins = harmonize_atac_intervals(temp1.var_names, temp2.var_names)
    logging.info(
        f"Got {len(pbmc_merged_atac_bins)} merged PBMC ATAC bins across 2 replicates"
    )
    with open(TENX_PBMC_ATAC_BINS_FILE, "w") as sink:
        for b in pbmc_merged_atac_bins:
            sink.write(b + "\n")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    # main()
    liftover_atac_adata(utils.sc_read_10x_h5_ft_type(sys.argv[1], "Peaks"))
