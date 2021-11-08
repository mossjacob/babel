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

from typing import *

import intervaltree

import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.sparse
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

import tqdm

import sortedcontainers

from babel import utils, adata_utils, plot_utils

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
assert os.path.isdir(DATA_DIR)
SNARESEQ_DATA_DIR = os.path.join(DATA_DIR, "snareseq_GSE126074")
assert os.path.isdir(SNARESEQ_DATA_DIR)
MM9_GTF = os.path.join(DATA_DIR, "Mus_musculus.NCBIM37.67.gtf.gz")
assert os.path.isfile(MM9_GTF)
MM10_GTF = os.path.join(DATA_DIR, "gencode.vM7.annotation.gtf.gz")
assert os.path.isfile(MM10_GTF)
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
    "pool_genomic_interval": 0,  # Smaller bin size because we can handle it
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
    "gtf_file": MM10_GTF,
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

TENX_PBMC_ATAC_DATA_KWARGS = {
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
    genes, gtf_file=MM10_GTF, return_genes=False, return_chrom=False
):
    """Reorders list of genes by their genomic coordinate in the given gtf"""
    assert len(genes) > 0, "Got empty set of genes"
    genes_set = set(genes)
    genes_list = list(genes)
    assert len(genes_set) == len(genes), f"Got duplicates in genes"

    genes_to_pos = utils.read_gtf_gene_to_pos(gtf_file)
    genes_intersection = [
        g for g in genes_to_pos if g in genes_set
    ]  # In order of position
    assert genes_intersection, "Got empty list of intersected genes"
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


def get_chrom_from_genes(genes: List[str], gtf_file=MM10_GTF):
    """
    Given a list of genes, return a list of chromosomes that those genes are on
    For missing: NA
    """
    gene_to_pos = utils.read_gtf_gene_to_pos(gtf_file)
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
    """
    Convert the string to tuple
    >>> interval_string_to_tuple("chr1:100-200")
    ('chr1', 100, 200)
    >>> interval_string_to_tuple("chr1:1e+06-1000199")
    ('chr1', 1000000, 1000199)
    """
    tokens = x.split(":")
    assert len(tokens) == 2, f"Malformed interval string: {x}"
    chrom, interval = tokens
    if not chrom.startswith("chr"):
        logging.warn(f"Got noncanonical chromsome in {x}")
    start, stop = map(float, interval.split("-"))
    assert start < stop, f"Got invalid interval span: {x}"
    return (chrom, int(start), int(stop))


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


def get_indices_to_form_target_genes(
    genes: List[str], target_genes: List[str]
) -> List[List[int]]:
    """
    Given a list of genes, and a target set of genes, return list
    of indices to combine to map into target
    While List[List[int]] structure isn't immediately necessary,
    it is useful for compatibility with above
    """
    assert set(genes).intersection(target_genes), "No shared genes"
    source_gene_to_idx = {gene: i for i, gene in enumerate(genes)}

    retval = []
    for target_gene in target_genes:
        if target_gene in source_gene_to_idx:
            retval.append([source_gene_to_idx[target_gene]])
        else:
            retval.append([])
    return retval


def combine_array_cols_by_idx(
    arr, idx: List[List[int]], combine_func: Callable = np.sum
) -> scipy.sparse.csr_matrix:
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
        elif len(indices) == 1:
            next_col = scipy.sparse.csc_matrix(arr.getcol(indices[0]))
        else:  # Multiple indices to combine
            col_set = np.hstack([arr.getcol(i).toarray() for i in indices])
            next_col = scipy.sparse.csc_matrix(
                combine_func(col_set, axis=1, keepdims=True)
            )
        new_cols.append(next_col)
    new_mat_sparse = scipy.sparse.hstack(new_cols).tocsr()
    assert (
        len(new_mat_sparse.shape) == 2
    ), f"Returned matrix is expected to be 2 dimensional, but got shape {new_mat_sparse.shape}"
    # print(arr.shape, new_mat_sparse.shape)
    return new_mat_sparse


def combine_by_proximity(
    arr, gtf_file=MM10_GTF, start_extension: int = 10000, stop_extension: int = 10000
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
        arr = pd.DataFrame(
            d,
            index=arr.obs_names,
            columns=arr.var_names,
        )
    assert isinstance(arr, pd.DataFrame)

    gene_to_pos = utils.read_gtf_gene_to_pos(
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


def _harmonize_atac_intervals(
    intervals_1: List[str], intervals_2: List[str]
) -> List[str]:
    """
    Given two files describing intervals, harmonize them by merging overlapping
    intervals for each chromosome
    """

    def interval_list_to_itree(
        l: List[str],
    ) -> Dict[str, intervaltree.IntervalTree]:
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
        if chrom not in itree2:  # Unique to itree1
            merged_itree[chrom] = itree1[chrom]
        combined = itree1[chrom] | itree2[chrom]
        combined.merge_overlaps()
        merged_itree[chrom] = combined
    for chrom in itree2.keys():  # Unique to itree2
        if chrom not in merged_itree:
            merged_itree[chrom] = itree2[chrom]

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


def harmonize_atac_intervals(*intervals: List[str]) -> List[str]:
    """
    Given multiple intervals, harmonize them
    >>> harmonize_atac_intervals(["chr1:100-200"], ["chr1:150-250"])
    ['chr1:100-250']
    >>> harmonize_atac_intervals(["chr1:100-200"], ["chr1:150-250"], ["chr1:300-350", "chr2:100-1000"])
    ['chr1:100-250', 'chr1:300-350', 'chr2:100-1000']
    """
    assert len(intervals) > 0
    if len(intervals) == 1:
        return intervals
    retval = _harmonize_atac_intervals(intervals[0], intervals[1])
    for i in intervals[2:]:
        retval = _harmonize_atac_intervals(retval, i)
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
    # This already gives a sparse matrix
    data_raw_aggregated = combine_array_cols_by_idx(x.X, idx)
    retval = AnnData(
        data_raw_aggregated,
        obs=x.obs,
        var=pd.DataFrame(index=target_bins),
    )
    return retval


def repool_genes(x: AnnData, target_genes: List[str]) -> AnnData:
    """
    Reorder (insert/drop cols) from x to match given target genes
    """
    idx = get_indices_to_form_target_genes(x.var_names, target_genes=target_genes)
    data_raw_aggregated = combine_array_cols_by_idx(x.X, idx)
    return AnnData(data_raw_aggregated, obs=x.obs, var=pd.DataFrame(index=target_genes))


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
        assert retval.shape == x.shape
    else:
        raise ValueError(f"Cannot CLR transform array with {len(x.shape)} dims")
    return retval


def read_bird_table(fname: str, atac_bins: Iterable[str] = []) -> AnnData:
    """Read the table outputted by BIRD. If atac_bins is given, ignore non-overlapping peaks"""
    # Expect 1361776 lines in file
    # create dict of interval tree from atac_bins
    peaks_itree = collections.defaultdict(intervaltree.IntervalTree)
    for peak in atac_bins:
        chrom, grange = peak.split(":")
        start, stop = (int(i) for i in grange.split("-"))
        peaks_itree[chrom][start:stop] = peak

    opener = gzip.open if fname.endswith(".gz") else open

    rows = []
    atac_intervals = []
    with opener(fname) as source:
        for i, line in tqdm.tqdm(enumerate(source)):
            line = line.decode()
            tokens = line.strip().split("\t")
            if i == 0:
                cell_barcodes = tokens[3:]
                continue
            chrom = tokens[0]
            start = int(float(tokens[1]))
            stop = int(float(tokens[2]))
            # If atac_bins is given, do a check for overlap
            if atac_bins:
                # Check for overlap
                if chrom not in peaks_itree or not peaks_itree[chrom][start:stop]:
                    continue
            interval = f"{chrom}:{start}-{stop}"
            atac_intervals.append(interval)
            values = scipy.sparse.csr_matrix(np.array(tokens[3:]).astype(float))
            rows.append(values)

    # Stack, tranpose to csc matrix, recast as csr matrix
    retval = AnnData(
        scipy.sparse.vstack(rows).T.tocsr(),
        obs=pd.DataFrame(index=cell_barcodes),
        var=pd.DataFrame(index=atac_intervals),
    )
    return retval


def main():
    """On the fly testing"""
    x = read_bird_table(
        sys.argv[1],
        utils.read_delimited_file(
            "/home/wukevin/projects/commonspace_models_final/cv_logsplit_01/atac_bins.txt"
        ),
    )
    logging.info(f"Read in {sys.argv[1]} for {x}")
    logging.info(f"Writing AnnData to {sys.argv[2]}")
    x.write_h5ad(sys.argv[2])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
