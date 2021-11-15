import copy
import os
from dataclasses import dataclass

from babel.data.loaders import (
    SNARESEQ_RNA_DATA_KWARGS, SNARESEQ_ATAC_DATA_KWARGS
)
from babel.data.single_cell_dataset import SingleCellDataset, SingleCellDatasetSplit
from babel.data.datasets import PairedDataset


@dataclass
class SnareConfig:
    validcluster:  int = 0
    testcluster:   int = 1
    clustermethod: str = 'leiden'  # leiden | louvain
    linear:        bool = True


def load_or_build_cortex_dataset(config: SnareConfig, save_dir=None, load=True):
    rna_data_kwargs = copy.copy(SNARESEQ_RNA_DATA_KWARGS)
    rna_data_kwargs["data_split_by_cluster_log"] = not config.linear
    rna_data_kwargs["data_split_by_cluster"] = config.clustermethod
    atac_data_kwargs = copy.copy(SNARESEQ_ATAC_DATA_KWARGS)
    atac_data_kwargs["cluster_res"] = 0  # Do not bother clustering ATAC data

    if save_dir is not None:
        if not os.path.isdir(save_dir):
            assert not os.path.exists(save_dir)
            os.makedirs(save_dir)
            load = False
        assert os.path.isdir(save_dir)

    # if load:

    rna_dataset = SingleCellDataset(
        valid_cluster_id=config.validcluster,
        test_cluster_id=config.testcluster,
        **rna_data_kwargs,
    )
    atac_dataset = SingleCellDataset(
        predefined_split=rna_dataset, **atac_data_kwargs
    )
    dual_full_dataset = PairedDataset(
        rna_dataset, atac_dataset, flat_mode=True,
    )
    rna_dataset.size_norm_counts.write_h5ad(
        os.path.join(save_dir, "full_rna.h5ad")
    )
    rna_dataset.size_norm_log_counts.write_h5ad(
        os.path.join(save_dir, "full_rna_log.h5ad")
    )
    atac_dataset.adata.write_h5ad(os.path.join(save_dir, "full_atac.h5ad"))

    dual_subsets = list()
    for subset in ['train', 'valid', 'test']:
        rna_subset = SingleCellDatasetSplit(
            rna_dataset, split=subset,
        )
        atac_subset = SingleCellDatasetSplit(
            atac_dataset, split=subset,
        )
        dual_subset = PairedDataset(
            rna_subset, atac_subset, flat_mode=True,
        )
        dual_subsets.append(dual_subset)
        rna_subset.size_norm_counts.write_h5ad(
            os.path.join(save_dir, f'{subset}_rna.h5ad')
        )
        atac_subset.adata.write_h5ad(
            os.path.join(save_dir, f'{subset}_atac.h5ad')
        )

    with open(os.path.join(save_dir, "rna_genes.txt"), "w") as sink:
        for gene in rna_dataset.adata.var_names:
            sink.write(gene + "\n")
    with open(os.path.join(save_dir, "atac_bins.txt"), "w") as sink:
        for atac_bin in atac_dataset.adata.var_names:
            sink.write(atac_bin + "\n")

    return dual_full_dataset, dual_subsets