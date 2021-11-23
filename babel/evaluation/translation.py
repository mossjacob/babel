import os
from typing import *
import logging
import scipy
import scanpy as sc

from babel import model_utils, plot_utils, utils


def evaluate_rna_from_rna(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    """
    Evaluate the given network on the dataset
    """
    # Do inference and plotting
    ### RNA > RNA
    logging.info("Inferring RNA from RNA...")
    sc_rna_full_preds = spliced_net.translate_1_to_1(sc_dual_full_dataset)
    sc_rna_full_preds_anndata = sc.AnnData(
        sc_rna_full_preds,
        obs=sc_dual_full_dataset.dataset_x.adata.obs,
    )
    sc_rna_full_preds_anndata.var_names = gene_names

    logging.info("Writing RNA from RNA")
    sc_rna_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_rna_rna_adata.h5ad".strip("_"))
    )
    if hasattr(sc_dual_full_dataset.dataset_x, "size_norm_counts") and ext is not None:
        logging.info("Plotting RNA from RNA")
        plot_utils.plot_scatter_with_r(
            sc_dual_full_dataset.dataset_x.size_norm_counts.X,
            sc_rna_full_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title=f"RNA > RNA".strip(),
            fname=os.path.join(outdir, f"{prefix}_rna_rna_log.{ext}".strip("_")),
        )


def evaluate_atac_from_rna(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### RNA > ATAC
    logging.info("Inferring ATAC from RNA")
    sc_rna_atac_full_preds = spliced_net.translate_1_to_2(sc_dual_full_dataset)
    sc_rna_atac_full_preds_anndata = sc.AnnData(
        scipy.sparse.csr_matrix(sc_rna_atac_full_preds),
        obs=sc_dual_full_dataset.dataset_x.adata.obs,
    )
    sc_rna_atac_full_preds_anndata.var_names = atac_names
    logging.info("Writing ATAC from RNA")
    sc_rna_atac_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_rna_atac_adata.h5ad".strip("_"))
    )

    if hasattr(sc_dual_full_dataset.dataset_y, "adata") and ext is not None:
        logging.info("Plotting ATAC from RNA")
        plot_utils.plot_auroc(
            utils.ensure_arr(sc_dual_full_dataset.dataset_y.adata.X).flatten(),
            utils.ensure_arr(sc_rna_atac_full_preds).flatten(),
            title_prefix=f"RNA > ATAC".strip(),
            fname=os.path.join(outdir, f"{prefix}_rna_atac_auroc.{ext}".strip("_")),
        )
        # plot_utils.plot_auprc(
        #     utils.ensure_arr(sc_dual_full_dataset.dataset_y.adata.X).flatten(),
        #     utils.ensure_arr(sc_rna_atac_full_preds),
        #     title_prefix=f"{DATASET_NAME} RNA > ATAC".strip(),
        #     fname=os.path.join(outdir, f"{prefix}_rna_atac_auprc.{ext}".strip("_")),
        # )


def evaluate_atac_from_atac(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### ATAC > ATAC
    logging.info("Inferring ATAC from ATAC")
    sc_atac_full_preds = spliced_net.translate_2_to_2(sc_dual_full_dataset)
    sc_atac_full_preds_anndata = sc.AnnData(
        sc_atac_full_preds,
        obs=sc_dual_full_dataset.dataset_y.adata.obs.copy(deep=True),
    )
    sc_atac_full_preds_anndata.var_names = atac_names
    logging.info("Writing ATAC from ATAC")

    # Infer marker bins
    # logging.info("Getting marker bins for ATAC from ATAC")
    # plot_utils.preprocess_anndata(sc_atac_full_preds_anndata)
    # adata_utils.find_marker_genes(sc_atac_full_preds_anndata)
    # inferred_marker_bins = adata_utils.flatten_marker_genes(
    #     sc_atac_full_preds_anndata.uns["rank_genes_leiden"]
    # )
    # logging.info(f"Found {len(inferred_marker_bins)} marker bins for ATAC from ATAC")
    # with open(
    #     os.path.join(outdir, f"{prefix}_atac_atac_marker_bins.txt".strip("_")), "w"
    # ) as sink:
    #     sink.write("\n".join(inferred_marker_bins) + "\n")

    sc_atac_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_atac_atac_adata.h5ad".strip("_"))
    )
    if hasattr(sc_dual_full_dataset.dataset_y, "adata") and ext is not None:
        logging.info("Plotting ATAC from ATAC")
        plot_utils.plot_auroc(
            utils.ensure_arr(sc_dual_full_dataset.dataset_y.adata.X).flatten(),
            utils.ensure_arr(sc_atac_full_preds).flatten(),
            title_prefix=f"ATAC > ATAC".strip(),
            fname=os.path.join(outdir, f"{prefix}_atac_atac_auroc.{ext}".strip("_")),
        )
        # plot_utils.plot_auprc(
        #     utils.ensure_arr(sc_dual_full_dataset.dataset_y.adata.X).flatten(),
        #     utils.ensure_arr(sc_atac_full_preds).flatten(),
        #     title_prefix=f"{DATASET_NAME} ATAC > ATAC".strip(),
        #     fname=os.path.join(outdir, f"{prefix}_atac_atac_auprc.{ext}".strip("_")),
        # )

    # Remove some objects to free memory
    del sc_atac_full_preds
    del sc_atac_full_preds_anndata


def evaluate_rna_from_atac(
    spliced_net,
    sc_dual_full_dataset,
    gene_names: str,
    atac_names: str,
    outdir: str,
    ext: str,
    marker_genes: List[str],
    prefix: str = "",
):
    ### ATAC > RNA
    logging.info("Inferring RNA from ATAC")
    sc_atac_rna_full_preds = spliced_net.translate_2_to_1(sc_dual_full_dataset)
    # Seurat expects everything to be sparse
    # https://github.com/satijalab/seurat/issues/2228
    sc_atac_rna_full_preds_anndata = sc.AnnData(
        sc_atac_rna_full_preds,
        obs=sc_dual_full_dataset.dataset_y.adata.obs.copy(deep=True),
    )
    sc_atac_rna_full_preds_anndata.var_names = gene_names
    logging.info("Writing RNA from ATAC")

    # Seurat also expects the raw attribute to be populated
    sc_atac_rna_full_preds_anndata.raw = sc_atac_rna_full_preds_anndata.copy()
    sc_atac_rna_full_preds_anndata.write(
        os.path.join(outdir, f"{prefix}_atac_rna_adata.h5ad".strip("_"))
    )
    # sc_atac_rna_full_preds_anndata.write_csvs(
    #     os.path.join(outdir, f"{prefix}_atac_rna_constituent_csv".strip("_")),
    #     skip_data=False,
    # )
    # sc_atac_rna_full_preds_anndata.to_df().to_csv(
    #     os.path.join(outdir, f"{prefix}_atac_rna_table.csv".strip("_"))
    # )

    # If there eixsts a ground truth RNA, do RNA plotting
    if hasattr(sc_dual_full_dataset.dataset_x, "size_norm_counts") and ext is not None:
        logging.info("Plotting RNA from ATAC")
        plot_utils.plot_scatter_with_r(
            sc_dual_full_dataset.dataset_x.size_norm_counts.X,
            sc_atac_rna_full_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title=f"ATAC > RNA".strip(),
            fname=os.path.join(outdir, f"{prefix}_atac_rna_log.{ext}".strip("_")),
        )

    # Remove objects to free memory
    del sc_atac_rna_full_preds
    del sc_atac_rna_full_preds_anndata

