import pandas as pd

from typing import *
from anndata import AnnData

from dataclasses import dataclass
from babel import utils, plot_utils, adata_utils


@dataclass
class FilterConfig:
    cell_min_counts: object = None  # All of these are off by default
    cell_max_counts: object = None
    cell_min_genes:  object = None
    cell_max_genes:  object = None
    gene_min_counts: object = None
    gene_max_counts: object = None
    gene_min_cells:  object = None
    gene_max_cells:  object = None


def join_cell_info(adata, cell_info):
    assert isinstance(cell_info, pd.DataFrame)
    if adata.obs is not None and not adata.obs.empty:
        adata.obs = adata.obs.join(
            cell_info, how="left", sort=False
        )
    else:
        adata.obs = cell_info
    assert (
            adata.shape[0] == adata.obs.shape[0]
    ), f"Got discordant shapes for data and obs: {adata.shape} {adata.obs.shape}"


def join_gene_info(adata, gene_info):
    assert isinstance(gene_info, pd.DataFrame)
    if (
            adata.var is not None and not adata.var.empty
    ):  # Is not None and is not empty
        adata.var = adata.var.join(
            gene_info, how="left", sort=False
        )
    else:
        adata.var = gene_info
    assert (
            adata.shape[1] == adata.var.shape[0]
    ), f"Got discordant shapes for data and var: {adata.shape} {adata.var.shape}"
