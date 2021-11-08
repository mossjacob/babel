import skorch
import skorch.utils

from scipy import sparse


class AutoEncoderSkorchNet(skorch.NeuralNet):
    """Subclassed so that we can easily extract the encoded layer"""

    def __init__(self, prop_reg_lambda: float = 0.0, *args, **kwargs):
        assert isinstance(prop_reg_lambda, float)
        super(AutoEncoderSkorchNet, self).__init__(*args, **kwargs)
        self.prop_reg_lambda = prop_reg_lambda

    def get_encoded_layer(self, x):
        """Return the encoded representation"""
        encoded = []
        for output in self.forward_iter(x, training=False):
            assert isinstance(output, tuple)
            e = output[-1]
            encoded.append(skorch.utils.to_numpy(e))
        return np.concatenate(encoded, 0)

    def get_prop_reg(self, y_pred, y_true):
        """
        Compute regularization based on the overall proportion of each gene in the batch
        """
        per_gene_counts = torch.sum(y_true, axis=0)  # Sum across the batch
        per_gene_counts_norm = per_gene_counts / torch.sum(per_gene_counts)

        per_gene_pred_counts = torch.sum(y_pred, axis=0)
        per_gene_pred_counts_norm = per_gene_pred_counts / torch.sum(
            per_gene_pred_counts
        )

        # L2 distance between the two
        # d = F.pairwise_distance(per_gene_pred_counts_norm, per_gene_counts_norm, p=2, eps=1e-6, keepdim=False)
        d2 = torch.pow(per_gene_counts_norm - per_gene_pred_counts_norm, 2).mean()
        # d = torch.sqrt(d2)
        return d2

    def get_prop_reg_pbulk(self, y_pred, pbulk):
        """
        Compute regularization based on pre-computed cluster-aggregated pseudobulk
        """
        return F.mse_loss(y_pred, pbulk)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        """
        Adds in a regularization term
        If y_true is a tuple, then we assume that is it is (cell_truth, cell_cluster_psuedobulk_mean)

        Reference:
        https://github.com/skorch-dev/skorch/blob/4097e90/skorch/net.py
        """
        # print(type(y_pred), type(y_true))
        if isinstance(y_true, tuple) or isinstance(y_true, list):
            assert len(y_true) == 2
            y, y_cluster_pbulk = y_true
            y = skorch.utils.to_tensor(y, device=self.device)
            y_cluster_pbulk = skorch.utils.to_tensor(
                y_cluster_pbulk, device=self.device
            )
        else:
            y = skorch.utils.to_tensor(y_true, device=self.device)
        loss = self.criterion_(y_pred, y)

        # Add regularization term
        if self.prop_reg_lambda != 0.0:
            if isinstance(y_true, tuple) or isinstance(y_true, list):
                loss += self.prop_reg_lambda * self.get_prop_reg_pbulk(
                    y_pred[0], y_cluster_pbulk
                )
            else:
                loss += self.prop_reg_lambda * self.get_prop_reg(y_pred[0], y_true)

        # print(type(y_pred), type(y))
        return loss


class PairedAutoEncoderSkorchNet(skorch.NeuralNet):
    def forward_iter(self, X, training=False, device="cpu"):
        """Subclassed to work with tuples"""
        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for data in iterator:
            Xi = skorch.dataset.unpack_data(data)[0]
            yp = self.evaluation_step(Xi, training=training)
            if isinstance(yp, tuple):
                yield model_utils.recursive_to_device(yp)  # <- modification here
            else:
                yield yp.to(device)

    def predict_proba(self, x):
        """Subclassed so calling predict produces a tuple of outputs"""
        y_probas1, y_probas2 = [], []
        for yp in self.forward_iter(x, training=False):
            assert isinstance(yp, tuple)
            yp1 = yp[0][0]
            yp2 = yp[1][0]
            y_probas1.append(skorch.utils.to_numpy(yp1))
            y_probas2.append(skorch.utils.to_numpy(yp2))
        y_proba1 = np.concatenate(y_probas1, 0)
        y_proba2 = np.concatenate(y_probas2, 0)
        return y_proba1, y_proba2

    def get_encoded_layer(self, x):
        """Get the encoded representation as a TUPLE of two elements"""
        encoded1, encoded2 = [], []
        for out1, out2, *_other in self.forward_iter(x, training=False):
            encoded1.append(out1[-1])
            encoded2.append(out2[-1])
        return np.concatenate(encoded1, axis=0), np.concatenate(encoded2, axis=0)

    def translate_1_to_2(self, x):
        enc1, enc2 = self.get_encoded_layer(x)
        device = next(self.module_.parameters()).device
        enc1_torch = torch.from_numpy(enc1).to(device)
        return self.module_.translate_1_to_2(enc1_torch)[0].detach().cpu().numpy()

    def translate_2_to_1(self, x):
        enc1, enc2 = self.get_encoded_layer(x)
        device = next(self.module_.parameters()).device
        enc2_torch = torch.from_numpy(enc2).to(device)
        return self.module_.translate_2_to_1(enc2_torch)[0].detach().cpu().numpy()


class SplicedAutoEncoderSkorchNet(PairedAutoEncoderSkorchNet):
    """
    Skorch wrapper for the SplicedAutoEncoder above.
    Mostly here to take care of how we calculate loss
    """

    def predict_proba(self, x):
        """
        Subclassed so that calling predict produces a tuple of 4 outputs
        """
        y_probas1, y_probas2, y_probas3, y_probas4 = [], [], [], []
        for yp in self.forward_iter(x, training=False):
            assert isinstance(yp, tuple)
            yp1 = yp[0][0]
            yp2 = yp[1][0]
            yp3 = yp[2][0]
            yp4 = yp[3][0]
            y_probas1.append(skorch.utils.to_numpy(yp1))
            y_probas2.append(skorch.utils.to_numpy(yp2))
            y_probas3.append(skorch.utils.to_numpy(yp3))
            y_probas4.append(skorch.utils.to_numpy(yp4))
        y_proba1 = np.concatenate(y_probas1)
        y_proba2 = np.concatenate(y_probas2)
        y_proba3 = np.concatenate(y_probas3)
        y_proba4 = np.concatenate(y_probas4)
        # Order: 1to1, 1to2, 2to1, 2to2
        return y_proba1, y_proba2, y_proba3, y_proba4

    def get_encoded_layer(self, x):
        """Get the encoded representation as a TUPLE of two elements"""
        encoded1, encoded2 = [], []
        for out11, out12, out21, out22 in self.forward_iter(x, training=False):
            encoded1.append(out11[-1])
            encoded2.append(out22[-1])
        return np.concatenate(encoded1, axis=0), np.concatenate(encoded2, axis=0)

    def translate_1_to_1(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[0][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_1_to_2(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[1][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_2_to_1(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[2][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def translate_2_to_2(self, x) -> sparse.csr_matrix:
        retval = [
            sparse.csr_matrix(skorch.utils.to_numpy(yp[3][0]))
            for yp in self.forward_iter(x, training=False)
        ]
        return sparse.vstack(retval)

    def score(self, true, pred):
        """
        Required for sklearn gridsearch
        Since sklearn uses convention of (true, pred) in its score functions
        We use the same
        https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        """
        return self.get_loss(pred, true)


class PairedInvertibleAutoEncoderSkorchNet(PairedAutoEncoderSkorchNet):
    def translate_1_to_2(self, x):
        enc1, enc2 = self.get_encoded_layer(x)
        device = next(self.module_.parameters()).device
        enc1_torch = torch.from_numpy(enc1).to(device)
        return self.module_.translate_1_to_2(enc1_torch)[0].detach().cpu().numpy()

    def translate_2_to_1(self, x):
        enc1, enc2 = self.get_encoded_layer(x)
        device = next(self.module_.parameters()).device
        enc2_torch = torch.from_numpy(enc2).to(device)
        return self.module_.translate_2_to_1(enc2_torch)[0].detach().cpu().numpy()
