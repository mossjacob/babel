"""
Code to train a model
"""

import os
import sys
import logging
import argparse
import copy
import functools
import itertools
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import skorch

from skorch.helper import predefined_split
from babel import activations, loss_functions
from babel import utils, model_utils, plot_utils
from babel.data import loaders, SingleCellDatasetSplit, SingleCellDataset, PairedDataset, processing
from babel.models import skorch_wrappers, autoencoders
from babel.data.processing import join_cell_info, join_gene_info
from babel.data.loaders import sc_read_mtx, get_filter_config_from_kwargs


torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "babel"
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

MODELS_DIR = os.path.join(SRC_DIR, "models")
assert os.path.isdir(MODELS_DIR)
sys.path.append(MODELS_DIR)


logging.basicConfig(level=logging.INFO)

OPTIMIZER_DICT = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
}


def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", "-d", type=str, nargs="*", help="Data files to train on",
    )
    input_group.add_argument(
        "--snareseq",
        action="store_true",
        help="Data in SNAREseq format, use custom data loading logic for separated RNA ATC files",
    )
    input_group.add_argument(
        "--shareseq",
        nargs="+",
        type=str,
        choices=["lung", "skin", "brain"],
        help="Load in the given SHAREseq datasets",
    )
    parser.add_argument(
        "--nofilter",
        action="store_true",
        help="Whether or not to perform filtering (only applies with --data argument)",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Do clustering data splitting in linear instead of log space",
    )
    parser.add_argument(
        "--clustermethod",
        type=str,
        choices=["leiden", "louvain"],
        default="leiden",
        help="Clustering method to determine data splits",
    )
    parser.add_argument(
        "--validcluster", type=int, default=0, help="Cluster ID to use as valid cluster"
    )
    parser.add_argument(
        "--testcluster", type=int, default=1, help="Cluster ID to use as test cluster"
    )
    parser.add_argument(
        "--outdir", "-o", required=True, type=str, help="Directory to output to"
    )
    parser.add_argument(
        "--naive",
        "-n",
        action="store_true",
        help="Use a naive model instead of lego model",
    )
    parser.add_argument(
        "--hidden", type=int, nargs="*", default=[16], help="Hidden dimensions"
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        default="",
        help="params.pt file to use to warm initialize the model (instead of starting from scratch)",
    )
    parser.add_argument(
        "--lossweight",
        type=float,
        nargs="*",
        default=[1.33],
        help="Relative loss weight",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=OPTIMIZER_DICT.keys(),
        help="Optimizer to use",
    )
    parser.add_argument(
        "--lr", "-l", type=float, default=[0.01], nargs="*", help="Learning rate"
    )
    parser.add_argument(
        "--batchsize", "-b", type=int, nargs="*", default=[512], help="Batch size"
    )
    parser.add_argument(
        "--earlystop", type=int, default=25, help="Early stopping after N epochs"
    )
    parser.add_argument(
        "--seed", type=int, nargs="*", default=[182822], help="Random seed to use"
    )
    parser.add_argument("--device", default=0, type=int, help="Device to train on")
    parser.add_argument(
        "--ext",
        type=str,
        choices=["png", "pdf", "jpg"],
        default="pdf",
        help="Output format for plots",
    )
    return parser


def plot_loss_history(history, fname: str):
    """Create a plot of train valid loss"""
    fig, ax = plt.subplots(dpi=300)
    ax.plot(
        np.arange(len(history)), history[:, "train_loss"], label="Train",
    )
    ax.plot(
        np.arange(len(history)), history[:, "valid_loss"], label="Valid",
    )
    ax.legend()
    ax.set(
        xlabel="Epoch", ylabel="Loss",
    )
    fig.savefig(fname)
    return fig


def main():
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
    args.outdir = os.path.abspath(args.outdir)

    if not os.path.isdir(os.path.dirname(args.outdir)):
        os.makedirs(os.path.dirname(args.outdir))

    # Specify output log file
    logger = logging.getLogger()
    fh = logging.FileHandler(f"{args.outdir}_training.log", "w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Log parameters and pytorch version
    if torch.cuda.is_available():
        logging.info(f"PyTorch CUDA version: {torch.version.cuda}")
    for arg in vars(args):
        logging.info(f"Parameter {arg}: {getattr(args, arg)}")

    # Borrow parameters
    logging.info("Reading RNA data")
    if args.snareseq:
        rna_data_kwargs = copy.copy(loaders.SNARESEQ_RNA_DATA_KWARGS)
        atac_data_kwargs = copy.copy(loaders.SNARESEQ_ATAC_DATA_KWARGS)
        adata_gex = sc_read_mtx(rna_data_kwargs['fname'])
        adata_atac = sc_read_mtx(atac_data_kwargs['fname'])
    elif args.shareseq:
        logging.info(f"Loading in SHAREseq RNA and ATAC data for: {args.shareseq}")
        atac_data_kwargs = copy.copy(loaders.SNARESEQ_ATAC_DATA_KWARGS)
        atac_data_kwargs["reader"] = None
        atac_data_kwargs["fname"] = None
        atac_data_kwargs["cell_info"] = None
        atac_data_kwargs["gene_info"] = None
        atac_data_kwargs["transpose"] = False
        rna_data_kwargs = copy.copy(loaders.SNARESEQ_RNA_DATA_KWARGS)
        rna_data_kwargs["fname"] = None
        rna_data_kwargs["reader"] = None
        rna_data_kwargs["cell_info"] = None
        rna_data_kwargs["gene_info"] = None
        rna_data_kwargs["transpose"] = False
        # Load in the datasets
        shareseq_rna_adatas = []
        atac_adatas = []
        for tissuetype in args.shareseq:
            shareseq_rna_adatas.append(
                processing.load_shareseq_data(
                    tissuetype,
                    dirname="/data/wukevin/commonspace_data/GSE140203_SHAREseq",
                    mode="RNA",
                )
            )
            atac_adatas.append(
                processing.load_shareseq_data(
                    tissuetype,
                    dirname="/data/wukevin/commonspace_data/GSE140203_SHAREseq",
                    mode="ATAC",
                )
            )

        shareseq_rna_adata = shareseq_rna_adatas[0]
        if len(shareseq_rna_adatas) > 1:
            shareseq_rna_adata = shareseq_rna_adata.concatenate(
                *shareseq_rna_adatas[1:],
                join="inner",
                batch_key="tissue",
                batch_categories=args.shareseq,
            )
        adata_gex = shareseq_rna_adata

        atac_bins = [a.var_names for a in atac_adatas]
        if len(atac_adatas) > 1:
            atac_bins_harmonized = loaders.harmonize_atac_intervals(*atac_bins)
            atac_adatas = [
                loaders.repool_atac_bins(a, atac_bins_harmonized)
                for a in atac_adatas
            ]
        shareseq_atac_adata = atac_adatas[0]
        if len(atac_adatas) > 1:
            shareseq_atac_adata = shareseq_atac_adata.concatenate(
                *atac_adatas[1:],
                join="inner",
                batch_key="tissue",
                batch_categories=args.shareseq,
            )
        adata_atac = shareseq_atac_adata

    else:
        rna_data_kwargs = copy.copy(loaders.TENX_PBMC_RNA_DATA_KWARGS)
        rna_data_kwargs["fname"] = args.data
        if args.nofilter:
            rna_data_kwargs = {
                k: v for k, v in rna_data_kwargs.items() if not k.startswith("filt_")
            }
        adata_gex = rna_data_kwargs['reader'](args.data)
        atac_parsed = [
            utils.sc_read_10x_h5_ft_type(fname, "Peaks") for fname in args.data
        ]
        if len(atac_parsed) > 1:
            atac_bins = loaders.harmonize_atac_intervals(
                atac_parsed[0].var_names, atac_parsed[1].var_names
            )
            for bins in atac_parsed[2:]:
                atac_bins = loaders.harmonize_atac_intervals(
                    atac_bins, bins.var_names
                )
            logging.info(f"Aggregated {len(atac_bins)} bins")
        else:
            atac_bins = list(atac_parsed[0].var_names)

        atac_data_kwargs = copy.copy(loaders.TENX_PBMC_ATAC_DATA_KWARGS)
        atac_data_kwargs["fname"] = rna_data_kwargs["fname"]
        atac_data_kwargs["pool_genomic_interval"] = 0  # Do not pool
        reader = functools.partial(
            utils.sc_read_multi_files,
            reader=lambda x: loaders.repool_atac_bins(
                utils.sc_read_10x_h5_ft_type(x, "Peaks"), atac_bins,
            ),
        )
        adata_atac = reader(rna_data_kwargs['fname'])

    if rna_data_kwargs['transpose']:
        adata_gex = adata_gex.T
    if atac_data_kwargs['transpose']:
        adata_atac = adata_atac.T
    rna_data_kwargs["data_split_by_cluster_log"] = not args.linear
    rna_data_kwargs["data_split_by_cluster"] = args.clustermethod


    # Filter and join gene and cell info
    for adata, data_kwargs in zip([adata_gex, adata_atac], [rna_data_kwargs, atac_data_kwargs]):
        processing.annotate_basic_adata_metrics(adata)
        join_gene_info(adata, data_kwargs['gene_info'])
        join_cell_info(adata, data_kwargs['cell_info'])
        config = get_filter_config_from_kwargs(data_kwargs)
        processing.filter_adata_cells_and_genes(adata, config)

    # Normalize counts
    adata_gex = processing.normalize_count_table(  # Normalizes in place
        adata_gex,
        size_factors=rna_data_kwargs['calc_size_factors'],
        normalize=rna_data_kwargs['normalize'],
        log_trans=rna_data_kwargs['log_trans'],
    )

    sc_rna_dataset = SingleCellDataset(
        adata_gex,
        valid_cluster_id=args.validcluster,
        test_cluster_id=args.testcluster,
        **rna_data_kwargs,
    )

    sc_rna_train_dataset = SingleCellDatasetSplit(
        sc_rna_dataset, split="train",
    )
    sc_rna_valid_dataset = SingleCellDatasetSplit(
        sc_rna_dataset, split="valid",
    )
    sc_rna_test_dataset = SingleCellDatasetSplit(
        sc_rna_dataset, split="test",
    )

    # ATAC
    logging.info("Aggregating ATAC clusters")
    atac_data_kwargs["cluster_res"] = 0  # Do not bother clustering ATAC data
    if atac_data_kwargs['binarize']:
        # If we are binarizing data we probably don't care about raw counts
        # self.adata.raw = self.adata.copy()  # Store original counts
        adata_atac.X[adata_atac.X.nonzero()] = 1  # .X here is a csr matrix

    sc_atac_dataset = SingleCellDataset(
        adata_atac,
        predefined_split=sc_rna_dataset, **atac_data_kwargs
    )
    sc_atac_train_dataset = SingleCellDatasetSplit(
        sc_atac_dataset, split="train",
    )
    sc_atac_valid_dataset = SingleCellDatasetSplit(
        sc_atac_dataset, split="valid",
    )
    sc_atac_test_dataset = SingleCellDatasetSplit(
        sc_atac_dataset, split="test",
    )

    sc_dual_train_dataset = PairedDataset(
        sc_rna_train_dataset, sc_atac_train_dataset, flat_mode=True,
    )
    sc_dual_valid_dataset = PairedDataset(
        sc_rna_valid_dataset, sc_atac_valid_dataset, flat_mode=True,
    )
    sc_dual_test_dataset = PairedDataset(
        sc_rna_test_dataset, sc_atac_test_dataset, flat_mode=True,
    )
    sc_dual_full_dataset = PairedDataset(
        sc_rna_dataset, sc_atac_dataset, flat_mode=True,
    )

    # Model
    param_combos = list(
        itertools.product(
            args.hidden, args.lossweight, args.lr, args.batchsize, args.seed
        )
    )
    for h_dim, lw, lr, bs, rand_seed in param_combos:
        outdir_name = (
            f"{args.outdir}_hidden_{h_dim}_lossweight_{lw}_lr_{lr}_batchsize_{bs}_seed_{rand_seed}"
            if len(param_combos) > 1
            else args.outdir
        )
        if not os.path.isdir(outdir_name):
            assert not os.path.exists(outdir_name)
            os.makedirs(outdir_name)
        assert os.path.isdir(outdir_name)
        with open(os.path.join(outdir_name, "rna_genes.txt"), "w") as sink:
            for gene in sc_rna_dataset.adata.var_names:
                sink.write(gene + "\n")
        with open(os.path.join(outdir_name, "atac_bins.txt"), "w") as sink:
            for atac_bin in sc_atac_dataset.adata.var_names:
                sink.write(atac_bin + "\n")

        # Write dataset
        ### Full
        sc_rna_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "full_rna.h5ad")
        )
        sc_rna_dataset.size_norm_log_counts.write_h5ad(
            os.path.join(outdir_name, "full_rna_log.h5ad")
        )
        sc_atac_dataset.adata.write_h5ad(os.path.join(outdir_name, "full_atac.h5ad"))
        ### Train
        sc_rna_train_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "train_rna.h5ad")
        )
        sc_atac_train_dataset.adata.write_h5ad(
            os.path.join(outdir_name, "train_atac.h5ad")
        )
        ### Valid
        sc_rna_valid_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "valid_rna.h5ad")
        )
        sc_atac_valid_dataset.adata.write_h5ad(
            os.path.join(outdir_name, "valid_atac.h5ad")
        )
        ### Test
        sc_rna_test_dataset.size_norm_counts.write_h5ad(
            os.path.join(outdir_name, "truth_rna.h5ad")
        )
        sc_atac_test_dataset.adata.write_h5ad(
            os.path.join(outdir_name, "truth_atac.h5ad")
        )

        # Instantiate and train model
        model_class = (
            autoencoders.NaiveSplicedAutoEncoder
            if args.naive
            else autoencoders.AssymSplicedAutoEncoder
        )
        spliced_net = skorch_wrappers.SplicedAutoEncoderSkorchNet(
            module=model_class,
            module__hidden_dim=h_dim,  # Based on hyperparam tuning
            module__input_dim1=sc_rna_dataset.adata.shape[1],
            module__input_dim2=sc_atac_dataset.get_per_chrom_feature_count(),
            module__final_activations1=[
                activations.Exp(),
                activations.ClippedSoftplus(),
            ],
            module__final_activations2=nn.Sigmoid(),
            module__flat_mode=True,
            module__seed=rand_seed,
            lr=lr,  # Based on hyperparam tuning
            criterion=loss_functions.QuadLoss,
            criterion__loss2=loss_functions.BCELoss,  # handle output of encoded layer
            criterion__loss2_weight=lw,  # numerically balance the two losses with different magnitudes
            criterion__record_history=True,
            optimizer=OPTIMIZER_DICT[args.optim],
            iterator_train__shuffle=True,
            device=utils.get_device(args.device),
            batch_size=bs,  # Based on  hyperparam tuning
            max_epochs=500,
            callbacks=[
                skorch.callbacks.EarlyStopping(patience=args.earlystop),
                skorch.callbacks.LRScheduler(
                    policy=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    **model_utils.REDUCE_LR_ON_PLATEAU_PARAMS,
                ),
                skorch.callbacks.GradientNormClipping(gradient_clip_value=5),
                skorch.callbacks.Checkpoint(
                    dirname=outdir_name, fn_prefix="net_", monitor="valid_loss_best",
                ),
            ],
            train_split=predefined_split(sc_dual_valid_dataset),
            iterator_train__num_workers=8,
            iterator_valid__num_workers=8,
        )
        if args.pretrain:
            # Load in the warm start parameters
            spliced_net.load_params(f_params=args.pretrain)
            spliced_net.partial_fit(sc_dual_train_dataset, y=None)
        else:
            spliced_net.fit(sc_dual_train_dataset, y=None)

        fig = plot_loss_history(
            spliced_net.history, os.path.join(outdir_name, f"loss.{args.ext}")
        )
        plt.close(fig)

        logging.info("Evaluating on test set")
        logging.info("Evaluating RNA > RNA")
        sc_rna_test_preds = spliced_net.translate_1_to_1(sc_dual_test_dataset)
        sc_rna_test_preds_anndata = sc.AnnData(
            sc_rna_test_preds,
            var=sc_rna_test_dataset.adata.var,
            obs=sc_rna_test_dataset.adata.obs,
        )
        sc_rna_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "rna_rna_test_preds.h5ad")
        )
        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_rna_test_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title="RNA > RNA (test set)",
            fname=os.path.join(outdir_name, f"rna_rna_scatter_log.{args.ext}"),
        )
        plt.close(fig)

        logging.info("Evaluating ATAC > ATAC")
        sc_atac_test_preds = spliced_net.translate_2_to_2(sc_dual_test_dataset)
        sc_atac_test_preds_anndata = sc.AnnData(
            sc_atac_test_preds,
            var=sc_atac_test_dataset.adata.var,
            obs=sc_atac_test_dataset.adata.obs,
        )
        sc_atac_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "atac_atac_test_preds.h5ad")
        )
        fig = plot_utils.plot_auroc(
            sc_atac_test_dataset.adata.X,
            sc_atac_test_preds,
            title_prefix="ATAC > ATAC",
            fname=os.path.join(outdir_name, f"atac_atac_auroc.{args.ext}"),
        )
        plt.close(fig)

        logging.info("Evaluating ATAC > RNA")
        sc_atac_rna_test_preds = spliced_net.translate_2_to_1(sc_dual_test_dataset)
        sc_atac_rna_test_preds_anndata = sc.AnnData(
            sc_atac_rna_test_preds,
            var=sc_rna_test_dataset.adata.var,
            obs=sc_rna_test_dataset.adata.obs,
        )
        sc_atac_rna_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "atac_rna_test_preds.h5ad")
        )
        fig = plot_utils.plot_scatter_with_r(
            sc_rna_test_dataset.size_norm_counts.X,
            sc_atac_rna_test_preds,
            one_to_one=True,
            logscale=True,
            density_heatmap=True,
            title="ATAC > RNA (test set)",
            fname=os.path.join(outdir_name, f"atac_rna_scatter_log.{args.ext}"),
        )
        plt.close(fig)

        logging.info("Evaluating RNA > ATAC")
        sc_rna_atac_test_preds = spliced_net.translate_1_to_2(sc_dual_test_dataset)
        sc_rna_atac_test_preds_anndata = sc.AnnData(
            sc_rna_atac_test_preds,
            var=sc_atac_test_dataset.adata.var,
            obs=sc_atac_test_dataset.adata.obs,
        )
        sc_rna_atac_test_preds_anndata.write_h5ad(
            os.path.join(outdir_name, "rna_atac_test_preds.h5ad")
        )
        fig = plot_utils.plot_auroc(
            sc_atac_test_dataset.adata.X,
            sc_rna_atac_test_preds,
            title_prefix="RNA > ATAC",
            fname=os.path.join(outdir_name, f"rna_atac_auroc.{args.ext}"),
        )
        plt.close(fig)

        del spliced_net


if __name__ == "__main__":
    main()
