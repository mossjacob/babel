import logging
import scanpy as sc
import os


def evaluate_latent(
    spliced_net, sc_dual_full_dataset, outdir: str, prefix: str = ""
):
    """
    Pull out latent space and write to file
    """
    logging.info("Inferring latent representations")
    encoded_from_rna, encoded_from_atac = spliced_net.get_encoded_layer(
        sc_dual_full_dataset
    )

    if hasattr(sc_dual_full_dataset.dataset_x, "adata"):
        encoded_from_rna_adata = sc.AnnData(
            encoded_from_rna,
            obs=sc_dual_full_dataset.dataset_x.adata.obs.copy(deep=True),
        )
        encoded_from_rna_adata.write(
            os.path.join(outdir, f"{prefix}_rna_encoded_adata.h5ad".strip("_"))
        )
    if hasattr(sc_dual_full_dataset.dataset_y, "adata"):
        encoded_from_atac_adata = sc.AnnData(
            encoded_from_atac,
            obs=sc_dual_full_dataset.dataset_y.adata.obs.copy(deep=True),
        )
        encoded_from_atac_adata.write(
            os.path.join(outdir, f"{prefix}_atac_encoded_adata.h5ad".strip("_"))
        )
