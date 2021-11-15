import torch
import logging
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import functools
import copy
import collections

from pathlib import Path
from typing import *
from torch.utils.data import Dataset
from anndata import AnnData
from cached_property import cached_property

from babel import utils, plot_utils, adata_utils
from .loaders import (
    sc_read_mtx,
    MM10_GTF,
    get_indices_to_combine,
    get_indices_to_form_target_intervals,
    get_chrom_from_intervals,
    get_chrom_from_genes,
    combine_by_proximity,
    combine_array_cols_by_idx,
    euclidean_sim_matrix,
    reorder_genes_by_pos,
    shuffle_indices_train_valid_test,
    clr_transform
)


class SingleCellDataset(Dataset):
    """
    Given a sparse matrix file, load in dataset

    can be things like sklearn MaxAbsScaler().fit_transform
    Parameters:
        valid_cluster_id: Only used if data_split_by_cluster is on, default 0
        predefined_split: of type SingleCellDataset

    """

    def __init__(
            self,
            adata: Union[AnnData, None],  # Should be size-factor normalised and logged counts
            mode: str = "all",
            data_split_by_cluster: str = "leiden",
            valid_cluster_id: int = 0,
            test_cluster_id: int = 1,
            data_split_by_cluster_log: bool = True,
            predefined_split=None,
            selfsupervise: bool = True,
            pool_genomic_interval: Union[int, List[str]] = 0,
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
            gtf_file: str = MM10_GTF,  # GTF file mapping genes to chromosomes, unused for ATAC
            cluster_res: float = 2.0,
            cache_prefix: Path = None,
            **kwargs
    ):
        assert mode in [
            "all",
            "skip",
        ], "SingleCellDataset now operates as a full dataset only. Use SingleCellDatasetSplit to define data splits"
        assert y_mode in [
            "size_norm",
            "log_size_norm",
            "raw_count",
            "log_raw_count",
            "x",
        ], f"Unrecognized mode for y output: {y_mode}"
        self.mode = mode
        self.selfsupervise = selfsupervise
        self.x_dropout = x_dropout
        self.y_mode = y_mode
        self.sample_y = sample_y
        self.return_sf = return_sf
        self.return_pbulk = return_pbulk
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

        logging.info(
            f"Got AnnData object {str(adata)}"
        )
        self.adata = adata
        assert isinstance(
            self.adata, AnnData
        ), f"Expected AnnData but got {type(self.adata)}"
        if not isinstance(self.adata.X, scipy.sparse.csr_matrix):
            self.adata.X = scipy.sparse.csr_matrix(
                self.adata.X
            )  # Convert to sparse matrix

        # Filter out undesirable var/obs
        self.adata = adata_utils.filter_adata(
            self.adata, filt_cells=filter_samples, filt_var=filter_features
        )

        # Attach obs/var annotations
        if sort_by_pos:
            fname = f'{self.y_mode}_genes_reordered.npy'
            if cache_prefix and (self.cache_prefix / fname).exists():
                genes_reordered = np.load(self.cache_prefix / fname)
            else:
                genes_reordered, chroms_reordered = reorder_genes_by_pos(
                    self.adata.var_names, gtf_file=gtf_file, return_chrom=True
                )
            if cache_prefix:
                np.save(self.cache_prefix / fname, genes_reordered)
            self.adata = self.adata[:, genes_reordered]

        self.__annotate_chroms(gtf_file)
        if self.autosomes_only:
            autosomal_idx = [
                i
                for i, chrom in enumerate(self.adata.var["chrom"])
                if utils.is_numeric(chrom.strip("chr"))
            ]
            self.adata = self.adata[:, autosomal_idx]

        # Sort by the observation names so we can combine datasets
        sort_order_idx = np.argsort(self.adata.obs_names)
        self.adata = self.adata[sort_order_idx, :]
        # NOTE pooling occurs AFTER feature/observation filtering
        if pool_genomic_interval:
            self.__pool_features(pool_genomic_interval=pool_genomic_interval)
            # Re-annotate because we have lost this information
            self.__annotate_chroms(gtf_file)

        # Preprocess the data now that we're done filtering
        adata_utils.annotate_basic_adata_metrics(self.adata)
        self._size_norm_log_counts = AnnData(
            scipy.sparse.csr_matrix(self.adata.X),
            obs=pd.DataFrame(index=self.adata.obs_names),
            var=pd.DataFrame(index=self.adata.var_names),
            obsm=self.adata.obsm
        )
        size_norm_counts = np.exp(utils.ensure_arr(self.adata.X)) - 1
        self._size_norm_counts = AnnData(
            scipy.sparse.csr_matrix(size_norm_counts),
            obs=pd.DataFrame(index=self.adata.obs_names),
            var=pd.DataFrame(index=self.adata.var_names),
            obsm=self.adata.obsm
        )
        plot_utils.preprocess_anndata(
            self._size_norm_log_counts,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res
        )
        plot_utils.preprocess_anndata(
            self._size_norm_counts,
            louvain_resolution=self.cluster_res,
            leiden_resolution=self.cluster_res
        )
        self.adata.X = utils.ensure_arr(self.adata.X)

        if clip > 0:
            assert isinstance(clip, float) and 0.0 < clip < 50.0
            logging.info(f"Clipping to {clip} percentile")
            clip_low, clip_high = np.percentile(
                self.adata.X.flatten(), [clip, 100.0 - clip]
            )
            if clip_low == clip_high == 0:
                logging.warning("Skipping clipping, as clipping intervals are 0")
            else:
                assert (
                    clip_low < clip_high
                ), f"Got discordant values for clipping ends: {clip_low} {clip_high}"
                self.adata.X = np.clip(self.adata.X, clip_low, clip_high)

        # Make sure the data is a sparse matrix
        if not isinstance(self.adata.X, scipy.sparse.csr_matrix):
            self.adata.X = scipy.sparse.csr_matrix(self.adata.X)

        # Do all normalization before we split to make sure all folds get the same normalization
        self.data_split_to_idx = {}
        if predefined_split is not None:
            logging.info("Got predefined split, ignoring mode")
            # Subset items
            self.adata = self.adata[
                [
                    i
                    for i in predefined_split.adata.obs.index
                    if i in self.adata.obs.index
                ],
            ]
            assert (
                self.adata.n_obs > 0
            ), "No intersected obs names from predefined split"
            # Carry over cluster indexing
            self.data_split_to_idx = copy.copy(predefined_split.data_split_to_idx)
        elif mode != "skip":
            # Create dicts mapping string to list of indices
            if self.data_split_by_cluster:
                self.data_split_to_idx = self.__split_train_valid_test_cluster(
                    clustering_key=self.data_split_by_cluster
                    if isinstance(self.data_split_by_cluster, str)
                    else "leiden",
                    valid_cluster={str(self.valid_cluster_id)},
                    test_cluster={str(self.test_cluster_id)},
                )
            else:
                self.data_split_to_idx = self.__split_train_valid_test()
        else:
            logging.info("Got data split skip, skipping data split")
        self.data_split_to_idx["all"] = np.arange(len(self.adata))

        self.size_factors = (
            torch.from_numpy(self.adata.obs.size_factors.values).type(
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
        # if self.cache_prefix is not None:
        #     self.data_raw_cache_fname = self.cache_prefix / f'{self.y_mode}.adata.h5ad'
        #     logging.info(f"Setting cache at {self.data_raw_cache_fname}")
        #     self.adata.filename = self.data_raw_cache_fname
        #     if hasattr(self, "_size_norm_counts"):
        #         size_norm_cache_name = self.cache_prefix / f'{self.y_mode}.size_norm_counts.h5ad'
        #         logging.info(
        #             f"Setting size norm counts cache at {size_norm_cache_name}"
        #         )
        #         self._size_norm_counts.filename = size_norm_cache_name
        #     if hasattr(self, "_size_norm_log_counts"):
        #         size_norm_log_cache_name = (
        #             self.cache_prefix / f'{self.y_mode}.size_norm_log_counts.h5ad'
        #         )
        #         logging.info(
        #             f"Setting size log norm counts cache at {size_norm_log_cache_name}"
        #         )
        #         self._size_norm_log_counts.filename = size_norm_log_cache_name

    def __annotate_chroms(self, gtf_file: str = "") -> None:
        """Annotates chromosome information on the var field, without the chr prefix"""
        # gtf_file can be empty if we're using atac intervals
        feature_chroms = (
            get_chrom_from_intervals(self.adata.var_names)
            if list(self.adata.var_names)[0].startswith("chr")
            else get_chrom_from_genes(self.adata.var_names, gtf_file)
        )
        self.adata.var["chrom"] = feature_chroms

    def __pool_features(self, pool_genomic_interval: Union[int, List[str]]):
        n_obs = self.adata.n_obs
        if isinstance(pool_genomic_interval, int):
            if pool_genomic_interval > 0:
                # WARNING This will wipe out any existing var information
                idx, names = get_indices_to_combine(
                    list(self.adata.var.index), interval=pool_genomic_interval
                )
                data_raw_aggregated = combine_array_cols_by_idx(  # Returns np ndarray
                    self.adata.X,
                    idx,
                )
                data_raw_aggregated = scipy.sparse.csr_matrix(data_raw_aggregated)
                self.adata = AnnData(
                    data_raw_aggregated,
                    obs=self.adata.obs,
                    var=pd.DataFrame(index=names),
                )
            elif pool_genomic_interval < 0:
                assert (
                    pool_genomic_interval == -1
                ), f"Invalid value: {pool_genomic_interval}"
                # Pool based on proximity to genes
                data_raw_aggregated, names = combine_by_proximity(self.adata)
                self.adata = AnnData(
                    data_raw_aggregated,
                    obs=self.adata.obs,
                    var=pd.DataFrame(index=names),
                )
            else:
                raise ValueError(f"Invalid integer value: {pool_genomic_interval}")
        elif isinstance(pool_genomic_interval, (list, set, tuple)):
            idx = get_indices_to_form_target_intervals(
                self.adata.var.index, target_intervals=pool_genomic_interval
            )
            data_raw_aggregated = scipy.sparse.csr_matrix(
                combine_array_cols_by_idx(
                    self.adata.X,
                    idx,
                )
            )
            self.adata = AnnData(
                data_raw_aggregated,
                obs=self.adata.obs,
                var=pd.DataFrame(index=pool_genomic_interval),
            )
        else:
            raise TypeError(
                f"Unrecognized type for pooling features: {type(pool_genomic_interval)}"
            )
        assert self.adata.n_obs == n_obs

    def __split_train_valid_test(self) -> Dict[str, List[int]]:
        """
        Split the dataset into the appropriate split, returning the indices of split
        """
        logging.warning(
            f"Constructing {self.mode} random data split - not recommended due to potential leakage between data split"
        )
        indices = np.arange(self.adata.n_obs)
        (train_idx, valid_idx, test_idx,) = shuffle_indices_train_valid_test(
            indices, shuffle=True, seed=1234, valid=0.15, test=0.15
        )
        assert train_idx, "Got empty training split"
        assert valid_idx, "Got empty validation split"
        assert test_idx, "Got empty test split"
        data_split_idx = {}
        data_split_idx["train"] = train_idx
        data_split_idx["valid"] = valid_idx
        data_split_idx["test"] = test_idx
        return data_split_idx

    def __split_train_valid_test_cluster(
        self, clustering_key: str = "leiden", valid_cluster={"0"}, test_cluster={"1"}
    ) -> Dict[str, List[int]]:
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
        logging.info(
            f"Constructing {clustering_key} {'log' if self.data_split_by_cluster_log else 'linear'} clustered data split with valid test cluster {valid_cluster} {test_cluster}"
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
        data_split_idx = {}
        data_split_idx["train"] = train_idx
        data_split_idx["valid"] = valid_idx
        data_split_idx["test"] = test_idx
        return data_split_idx

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
            list(set(self.adata.var["chrom"]))
        )  # Sort to guarantee consistent ordering
        chrom_to_idx = collections.OrderedDict()
        for chrom in chromosomes:
            chrom_to_idx[chrom] = np.where(self.adata.var["chrom"] == chrom)
        return chrom_to_idx

    def __get_chrom_split_features(self, i):
        """Given an index, get the features split by chromsome, returning in chromosome-sorted order"""
        if self.x_dropout:
            raise NotImplementedError
        features = torch.from_numpy(
            utils.ensure_arr(self.adata.X[i]).flatten()
        ).type(torch.FloatTensor)
        assert len(features.shape) == 1  # Assumes one dimensional vector of features

        chrom_to_idx = self.__get_chrom_idx()
        retval = tuple([features[indices] for _chrom, indices in chrom_to_idx.items()])
        if self.concat_outputs:
            retval = torch.cat(retval)
        return retval

    def __len__(self):
        """Number of examples"""
        return self.adata.n_obs

    def get_item_data_split(self, idx: int, split: str):
        """Get the i-th item in the split (e.g. train)"""
        assert split in ["train", "valid", "test", "all"]
        if split == "all":
            return self.__getitem__(idx)
        else:
            return self.__getitem__(self.data_split_to_idx[split][idx])

    def __getitem__(self, i):
        # TODO compatibility with slices
        expression_data = (
            torch.from_numpy(utils.ensure_arr(self.adata.X[i]).flatten()).type(
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
            key = self.size_norm_counts.obs_names[i]
            target = torch.from_numpy(
                utils.ensure_arr(self.adata.raw.var_vector(key))
            ).type(torch.FloatTensor)
        elif self.y_mode.endswith("size_norm"):
            key = self.size_norm_counts.obs_names[i]
            target = torch.from_numpy(self.size_norm_counts.var_vector(key)).type(
                torch.FloatTensor
            )
        elif self.y_mode == "x":
            target = torch.from_numpy(
                utils.ensure_arr(self.adata.X[y_idx]).flatten()
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
                ith_cluster = self.adata.obs.iloc[i]["leiden"]
                pbulk = torch.from_numpy(
                    self.get_cluster_psuedobulk().var_vector(ith_cluster)
                ).type(torch.FloatTensor)
                retval.append((target, pbulk))
        elif self.return_pbulk:
            ith_cluster = self.adata.obs.iloc[i]["leiden"]
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
        assert self._size_norm_counts.shape == self.adata.shape
        return self._size_norm_counts

    def _set_size_norm_counts(self) -> AnnData:
        logging.info(f"Setting size normalized counts")
        raw_counts_anndata = AnnData(
            scipy.sparse.csr_matrix(self.adata.raw.X),
            obs=pd.DataFrame(index=self.adata.obs_names),
            var=pd.DataFrame(index=self.adata.var_names),
            obsm=self.adata.obsm
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
        assert self._size_norm_log_counts.shape == self.adata.shape
        return self._size_norm_log_counts

    def _set_size_norm_log_counts(self) -> AnnData:
        retval = self.size_norm_counts.copy()  # Generates a new copy
        logging.info(f"Setting log-normalized size-normalized counts")
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
        assert mode in self.adata.obs.columns
        cluster_labels = sorted(list(set(self.adata.obs[mode])))
        norm_counts = self.get_normalized_counts()
        aggs = []
        for cluster in cluster_labels:
            cluster_cells = np.where(self.adata.obs[mode] == cluster)
            pbulk = norm_counts.X[cluster_cells]
            pbulk_aggregate = np.sum(pbulk, axis=0, keepdims=True)
            if normalize:
                pbulk_aggregate = (
                    pbulk_aggregate
                    / np.sum(pbulk_aggregate)
                    * self.adata.uns["median_counts"]
                )
                assert np.isclose(
                    np.sum(pbulk_aggregate), self.adata.uns["median_counts"]
                )
            aggs.append(pbulk_aggregate)
        retval = AnnData(
            np.vstack(aggs),
            obs={mode: cluster_labels},
            var=self.adata.var,
        )
        return retval


class SingleCellDatasetSplit(Dataset):
    """
    Wraps SingleCellDataset to provide train/valid/test splits
    """

    def __init__(self, sc_dataset: SingleCellDataset, split: str) -> None:
        assert isinstance(sc_dataset, SingleCellDataset)
        self.dset = sc_dataset  # Full dataset
        self.split = split
        assert self.split in self.dset.data_split_to_idx
        logging.info(
            f"Created {split} data split with {len(self.dset.data_split_to_idx[self.split])} examples"
        )

    def __len__(self) -> int:
        return len(self.dset.data_split_to_idx[self.split])

    def __getitem__(self, index: int):
        return self.dset.get_item_data_split(index, self.split)

    # These properties facilitate compatibility with old code by forwarding some properties
    # Note that these are NOT meant to be modified
    @cached_property
    def size_norm_counts(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.size_norm_counts[indices].copy()

    @cached_property
    def adata(self) -> AnnData:
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.adata[indices].copy()

    @cached_property
    def obs_names(self):
        indices = self.dset.data_split_to_idx[self.split]
        return self.dset.adata.obs_names[indices]


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
        # Protein matrices are small anyway
        self.raw_counts.X = utils.ensure_arr(self.raw_counts.X)
        assert np.all(self.raw_counts.X >= 0), "Got negative raw counts"
        assert utils.is_integral_val(self.raw_counts.X), "Got non-integer counts"

        # Subset
        if obs_names is not None:
            logging.info(f"Subsetting protein dataset to {len(obs_names)} cells")
            self.raw_counts = self.raw_counts[list(obs_names)]
        else:
            logging.info("Full protein dataset with no subsetting")
        assert np.sum(self.raw_counts.X) > 0, "Got count matrix of all 0"

        # Normalize
        # Since this normalization is independently done PER CELL we don't have to
        # worry about doing this after we do subsetting
        clr_counts = clr_transform(self.raw_counts.X)
        # Use adata to be more similar to SingleCellDataset
        self.adata = AnnData(
            clr_counts,
            obs=self.raw_counts.obs,
            var=self.raw_counts.var,
        )
        self.adata.raw = self.raw_counts

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, i: int):
        clr_vec = self.adata.X[i].flatten()
        clr_tensor = torch.from_numpy(clr_vec).type(torch.FloatTensor)
        return clr_tensor, clr_tensor


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

        self.adata = sc.read_csv(counts_fname, first_column_names=True)
        if normalize:
            # Note that we normalize the ENTIRE DATASET as a whole
            # We don't subset till later, so all data splits have the same normalization
            self.adata = adata_utils.normalize_count_table(self.adata)

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
                np.arange(self.adata.n_obs),
                test=0,
                valid=0.2,
            )
            if self.mode == "train":
                self.adata = self.adata[indices_train]
                if self.labels is not None:
                    self.labels = self.labels[indices_train]
            elif self.mode == "valid":
                self.adata = self.adata[indices_valid]
                if self.labels is not None:
                    self.labels = self.labels[indices_valid]
            elif self.mode == "test":
                self.adata = self.adata[indices_test]
                if self.labels is not None:
                    self.labels = self.labels[indices_test]
            else:
                raise ValueError(f"Unrecognized mode: {self.mode}")
        assert not np.any(pd.isnull(self.adata))

        self.features_names = self.adata.var_names
        self.sample_names = self.adata.obs_names
        self.data = torch.from_numpy(self.adata.X).type(torch.FloatTensor)
        self.data_counts = torch.from_numpy(self.adata.raw.X).type(torch.FloatTensor)
        self.size_factors = torch.from_numpy(
            self.adata.obs.size_factors.values
        ).type(torch.FloatTensor)
        self.size_norm_counts = self.get_normalized_counts()
        if self.labels is not None:
            self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)
            assert len(self.labels) == len(
                self.adata
            ), f"Got mismatched sizes {len(self.labels)} {len(self.adata)}"

    def get_normalized_counts(self):
        """Return table of normalized counts w/ size factor adjustment and no other normalization"""
        raw_counts = self.adata.raw.X
        raw_counts_anndata = AnnData(
            raw_counts, obs=self.adata.obs, var=self.adata.var
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
