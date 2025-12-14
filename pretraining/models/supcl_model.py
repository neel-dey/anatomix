import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from models.base_model import BaseModel
from . import pretraining_networks as networks

import util.util as util
from collections import OrderedDict
import os


# -------------------------------
# Loss class for a single-layer
# -------------------------------
class SupPatchNCELoss(nn.Module):
    """
    Computes the supervised patch-based contrastive loss for
    representation learning.

    This loss encourages features from the same spatial class (as defined by
    segmentation labels) to be close in the embedding space, while features
    from different classes are pushed apart. It is designed to work with
    patch-level features extracted from 2D or 3D data, and supports both 2D
    and 3D segmentation tasks.

    This implementation is yoinked and modified from:
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py

    Parameters
    ----------
    opt : argparse.Namespace
        Options containing hyperparameters for the loss, including:
            - nce_T: Temperature parameter for contrastive loss.

    Attributes
    ----------
    opt : argparse.Namespace
        The options object.
    mask_dtype : torch.dtype
        Data type for masks (torch.bool).
    temperature : float
        Temperature scaling for contrastive logits.
    _cosine_similarity : torch.nn.CosineSimilarity
        Cosine similarity function along the last dimension.
    """

    def __init__(self, opt):

        super().__init__()
        self.opt = opt
        self.mask_dtype = torch.bool

        self.temperature = self.opt.nce_T
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _compute_similarity(self, x, y):
        assert x.size() == y.size(), f"wrong shape {x.size()} and {y.size()}"
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        return v

    def forward(
        self, features, labels_seg, labels_coords, coords_range, debug=False
    ):
        """
        Compute the supervised patch-based contrastive loss.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor of shape (n_views, num_patches, mlp_op_dim), where
            n_views is the number of views (typically 2), num_patches is the
            number of sampled patches, and mlp_op_dim is the feature dimension.
            Only n_views == 2 supported.
        labels_seg : torch.Tensor
            Segmentation label tensor of shape (bs, 1, H, W, D) or (bs, 1, H, W).
        labels_coords : torch.Tensor
            Patch coordinates tensor of shape (num_patches, 3) or (num_patches, 2).
        coords_range : torch.Size or tuple
            The spatial size of the segmentation label (H, W, D) or (H, W).
        debug : bool, optional
            If True, enables debug mode (default: False).

        Returns
        -------
        loss : torch.Tensor
            The computed supervised contrastive loss (scalar).
        """
        if len(coords_range) == 3:
            w, h, d = coords_range

            labels_seg = F.interpolate(
                labels_seg, size=(w, h, d), mode="nearest"
            )

            labels_seg = labels_seg.squeeze(1)[
                :,
                labels_coords[:, 0],
                labels_coords[:, 1],
                labels_coords[:, 2],
            ]

        elif len(coords_range) == 2:
            w, h = coords_range

            labels_seg = F.interpolate(labels_seg, size=(w, h), mode="nearest")

            labels_seg = labels_seg.squeeze(1)[
                :,
                labels_coords[:, 0],
                labels_coords[:, 1],
            ]
        else:
            raise NotImplementedError

        device = features.device
        ntps, num_patches, nc = features.size()

        # finds matching segs
        mask = torch.eq(labels_seg, labels_seg.T).float().to(device)

        contrast_count = features.shape[0]  # always two views in our work
        contrast_feature = features.view(ntps * num_patches, nc)

        # this corresponds to self.contrast_mode == 'all' in
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L62
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            self._compute_similarity(anchor_feature, contrast_feature),
            self.temperature,
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_patches * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # TODO: replace the 2nd term with torch.logsumexp
        # (no need for exp_logits then but replace with logits*logits_mask)
        # Random note: 2nd term is the denominator. whether to remove the positives
        # from the denominator is an open issue IMO
        # https://github.com/HobbitLong/SupContrast/issues/64#issuecomment-1182845137
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        # TODO: fix an edge case
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L92
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(ntps, num_patches).mean()

        return loss


# -------------------------------
# Trainer class for multi-layer loss computation
# -------------------------------

class SupCLModel(BaseModel):
    """
    Supervised contrastive learning model.

    This class implements a supervised contrastive learning (SupCL) model
    with a patch contrastive loss, supporting 3D data. 2D support is WIP.

    Inherits
    --------
    BaseModel : The base model class for all models in this codebase.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add model-specific options to the command line parser.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser to which new options will be added.
        is_train : bool, optional
            Whether the model is being used for training (default is True).

        Returns
        -------
        argparse.ArgumentParser
            The parser with added options.
        """
        # Add all model-specific arguments
        parser.add_argument(
            "--lambda_NCE",
            type=float,
            default=1.0,
            help="weight for NCE loss: NCE(X,Y)",
        )
        parser.add_argument(
            "--unfreeze_layers",
            type=str,
            default="",
            help="specify partial layers that will be trained",
        )
        parser.add_argument(
            "--nce_layers",
            type=str,
            default="3,6,10,13,17,20,24,27,31,34,38,41,45,48,52,55,59,62",
            help="compute NCE loss on which layers",
        )
        parser.add_argument(
            "--nce_weights",
            type=str,
            default="1",
            help='how to sum all nce losses. If "1", simply use mean. Otherwise, weighted average by specified weights that must have equal length of nce_layers',
        )
        parser.add_argument(
            "--last_id", type=int, default=65, help="id of the last layer"
        )
        parser.add_argument(
            "--netF",
            type=str,
            default="mlp_sample",
            choices=["mlp_sample"],
            help="Networks that projects sampled patches for the contrastive loss",
        )
        parser.add_argument("--netF_nc", type=int, default=256)
        parser.add_argument(
            "--use_mlp",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=True,
            help="whether to use MLP ",
        )
        parser.add_argument(
            "--nce_T", type=float, default=0.07, help="temperature for NCE loss"
        )
        parser.add_argument(
            "--num_patches",
            type=int,
            default=768,
            help="number of patches per layer",
        )
        parser.add_argument(
            "--n_mlps", type=int, default=3, help="number of mlp layers, 2 or 3"
        )

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters
        parser.set_defaults(lambda_NCE=1.0)

        return parser

    def __init__(self, opt):
        """
        Initialize the SupCLModel.

        Parameters
        ----------
        opt : argparse.Namespace
            The options/configuration for the model.
        """
        BaseModel.__init__(self, opt)

        # Specify the training losses to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        if self.opt.lambda_NCE > 0.0:
            self.loss_names += ["NCE"]

        self.grad_accum_iters = opt.grad_accum_iters

        print("Visuals: {}".format(self.visual_names))

        # Parse NCE layers and weights
        if not opt.nce_layers == "":
            self.nce_layers = [int(i) for i in self.opt.nce_layers.split(",")]
            self.last_layer_to_freeze = self.nce_layers[-1]
        else:
            self.nce_layers = []
            self.last_layer_to_freeze = -1

        if self.opt.nce_weights != "1":
            self.nce_weights = [
                float(i) for i in self.opt.nce_weights.split(",")
            ]
            sum_ = np.sum(np.asarray(self.nce_weights))
            self.nce_weights = [i / sum_ for i in self.nce_weights]
        else:
            if len(self.nce_layers) > 0:
                self.nce_weights = [
                    1.0 / len(self.nce_layers) for _ in self.nce_layers
                ]
            else:
                self.nce_weights = []
        assert len(self.nce_weights) == len(self.nce_layers)

        self.use_mask = opt.load_mask

        # Print NCE configuration if enabled
        if self.opt.lambda_NCE > 0:
            print("--------------- NCE configuration -------------")
            print("NCE Layers", self.nce_layers)
            print("NCE weights", self.nce_weights)
            print("Use foreground masking", self.use_mask)

        self.reset_params = False
        if not self.opt.unfreeze_layers == "":
            self.unfreeze_layers = [
                i for i in self.opt.unfreeze_layers.split(",")
            ]
            self.reset_params = True

        self.model_names = ["G"]
        # Define base network
        self.netG = networks.define_G(
            opt.ndims,
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            self.gpu_ids,
            opt,
            final_act="none",
            pooling=opt.pool_type,
            interp=opt.interp_type,
        )

        # Save base network configuration to file if not already present
        if not os.path.exists(os.path.join(self.save_dir, "netG.txt")):
            print("Write network configs into text")
            with open(os.path.join(self.save_dir, "netG.txt"), "w") as f:
                print(self.netG, file=f)
            f.close()

        if self.isTrain:
            # Define MLP networks and NCE loss:
            if self.opt.lambda_NCE > 0.0:
                self.netF = networks.define_F(
                    opt.input_nc,
                    opt.netF,
                    opt.normF,
                    opt.init_type,
                    opt.init_gain,
                    opt.n_mlps,
                    self.gpu_ids,
                    opt,
                    activation=opt.actF,
                    use_mlp=opt.use_mlp,
                )

                if self.opt.use_mlp:
                    self.model_names += ["F"]

                # Define loss functions for each NCE layer
                self.criterionNCE = []
                for nce_layer in self.nce_layers:
                    self.criterionNCE.append(
                        SupPatchNCELoss(opt).to(self.device)
                    )

            # Selectively unfreeze layers if specified
            if self.reset_params:
                paramsG = []
                params_dict_G = dict(self.netG.named_parameters())
                for key, value in params_dict_G.items():
                    grad = False
                    for f in self.unfreeze_layers:
                        if f in key:
                            print("Add %s to optimizer list" % key)
                            grad = True
                            paramsG += [{"params": [value]}]
                            break
                    value.requires_grad = grad
                    print(key, value.requires_grad)
            else:
                paramsG = self.netG.parameters()

            # Define optimizer for base network
            # Paper used Adam only, not AdamW
            self.optimizer_G = torch.optim.AdamW(
                paramsG,
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2),
                eps=opt.eps,
                weight_decay=opt.weight_decay,
            )

            self.optimizers.append(self.optimizer_G)

        else:
            # For inference, set feature names
            self.feat_names = ["feat", "global_feat"]

        self.new_bs = -1
        if len(self.load_model_names) == 0:
            self.load_model_names = self.model_names

    def data_dependent_initialize(self, data):
        """
        Initialize the feature network netF based on the shape of the intermediate
        features of various layers of netG. This is required because netF's
        weights depend on the input data shape. Also, see
        PatchSampleF.create_mlp(), which is called at first forward pass.

        Parameters
        ----------
        data : dict
            Input data batch for initialization.
        """
        if self.opt.isTrain:
            if self.opt.lambda_NCE > 0:
                self.set_input(data, verbose=True)

                # TODO (this block should go into set_input)
                # there's also some unfinished multigpu shenanigans here
                """
                bs_per_gpu = 2 #self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
                self.real_A = self.real_A[:bs_per_gpu]
                if self.use_mask:
                    self.mask_A = self.mask_A[:bs_per_gpu]
                self.label_subjid = self.label_subjid[:bs_per_gpu]
                """

                self.forward(verbose=False)
                self.compute_G_loss(0).backward()
                if (
                    self.opt.lambda_NCE > 0.0
                    and self.opt.use_mlp
                ):
                    # Define optimizer for patch MLP network
                    self.optimizer_F = torch.optim.AdamW(
                        self.netF.parameters(),
                        lr=self.opt.lr,
                        betas=(self.opt.beta1, self.opt.beta2),
                        eps=self.opt.eps,
                        weight_decay=self.opt.weight_decay,
                    )
                    self.optimizers.append(self.optimizer_F)
                # Save patch MLP network configuration to file if not already present
                if not os.path.exists(os.path.join(self.save_dir, "netF.txt")):
                    print("Write network configs into text")
                    with open(
                        os.path.join(self.save_dir, "netF.txt"), "w"
                    ) as f:
                        print(self.netF, file=f)
                    f.close()
                torch.cuda.empty_cache()
                # self.parallelize()

    def optimize_parameters(self, iters):
        """
        Calculate losses, perform backward pass, and update network weights.

        Parameters
        ----------
        iters : int
            The current training iteration (used for gradient accumulation).
        """
        # Forward pass and accumulate gradients
        accum_iter = self.grad_accum_iters  # e.g., 4
        with torch.set_grad_enabled(True):
            self.forward()
            self.loss_G = self.compute_G_loss()
            self.loss_G = self.loss_G / accum_iter
            self.loss_G.backward()

        # Update weights after accum_iter steps
        if (iters) % accum_iter == 0:
            self.optimizer_G.step()

            if (
                self.opt.lambda_NCE > 0
                and self.opt.use_mlp
            ):
                if self.opt.clip_grad:
                    nn.utils.clip_grad_norm_(
                        self.netF.parameters(),
                        max_norm=self.opt.max_norm,
                        norm_type=2,
                        error_if_nonfinite=True,
                    )
                self.optimizer_F.step()
            # Zero gradients
            self.optimizer_G.zero_grad()
            if (
                self.opt.lambda_NCE > 0
                and self.opt.use_mlp
            ):
                self.optimizer_F.zero_grad()

    def set_input(self, input, verbose=False):
        """
        Unpack input data from the dataloader and perform necessary 
        pre-processing steps.

        Parameters
        ----------
        input : dict
            The input data batch, including images and metadata.
        verbose : bool, optional
            Whether to print verbose information (default is False).
        """
        self.verbose = verbose

        if self.isTrain:
            # Load real_A and subject IDs
            self.real_A = input["A"].to(self.device).float()
            if self.opt.lambda_NCE > 0:
                self.subj_A = input["A_id"].to(self.device).float()

            # Load real_B and subject IDs if present
            if "B" in input.keys():
                self.real_B = input["B"].to(self.device).float()
                if self.opt.lambda_NCE > 0:
                    self.subj_B = input["B_id"].to(self.device).float()
                    self.label_subjid = torch.cat(
                        (self.subj_A, self.subj_B), dim=0
                    ).squeeze(-1)
            else:
                self.real_B = None
                if self.opt.lambda_NCE > 0:
                    self.label_subjid = self.subj_A.squeeze(-1)

            # Load segmentation masks for domain A if present
            if "A_seg" in input.keys():
                self.seg_A = input["A_seg"].to(self.device)
                self.visual_names = ["real_A", "pred_seg_A", "seg_A"]
                self.seg_A_gt = True
            else:
                self.seg_A = None
                self.seg_A_gt = False

            # Load segmentation masks for domain B if present
            if "B_seg" in input.keys():
                self.seg_B = input["B_seg"].to(self.device)
                self.visual_names += ["real_B", "pred_seg_B", "seg_B"]
                self.seg_B_gt = True
            else:
                self.seg_B = None
                self.seg_B_gt = False
            del input
            print("A", self.real_A.size())

        else:
            self.real_A = input["A"].to(self.device).float()
            if "A_seg" in input.keys():
                self.seg_A = input["A_seg"].to(self.device).float()
            else:
                self.visual_names = ["real_A", "pred_seg"]

    def forward(self, verbose=False):
        """
        Run forward pass; called by both <optimize_parameters> and <test>.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print verbose information (default is False).
        """
        if self.isTrain:
            # For segmentation: concatenate real_A and real_B if both present
            if self.real_B is not None:
                self.reals = torch.cat((self.real_A, self.real_B), dim=0)
            else:
                self.reals = self.real_A

            # Forward pass through base network with NCE layers
            segs, self.feat_kq = self.netG(
                self.reals, self.nce_layers, False, verbose=verbose
            )

            # Split predictions for A and B
            self.pred_A = segs[: self.real_A.size(0)]
            self.pred_seg_A = (
                torch.argmax(F.softmax(self.pred_A.detach(), dim=1), dim=1)
                .float()
                .unsqueeze(1)
            )

            self.pred_B = segs[self.real_A.size(0) :]
            self.pred_seg_B = (
                torch.argmax(F.softmax(self.pred_B.detach(), dim=1), dim=1)
                .float()
                .unsqueeze(1)
            )

        else:
            # Test time: single input
            crop_size = self.opt.crop_size
            if self.opt.ndims == 2:
                raise NotImplementedError

            elif self.opt.ndims == 3:
                if crop_size == -1:
                    patch = self.real_A
                    self.pred_seg = self.netG(patch, [], False)
                    self.pred_seg = (
                        torch.argmax(F.softmax(self.pred_seg, dim=1), dim=1)
                        .float()
                        .unsqueeze(1)
                    )
                else:
                    raise NotImplementedError

    def compute_G_loss(self, step=0):
        """
        Calculate NCE loss.

        Parameters
        ----------
        step : int, optional
            Training step (default is 0, not used).

        Returns
        -------
        float
            The total NCE loss.
        """
        self.loss_G = 0.0

        if self.opt.lambda_NCE > 0:
            self.loss_NCE, self.nce_dict = self.calculate_NCE_loss(
                self.feat_kq, self.seg_A, vis=False
            )
            self.loss_G += self.loss_NCE * self.opt.lambda_NCE

        return self.loss_G

    def calculate_NCE_loss(self, feat_kq, seg, step=-1, vis=False):
        """
        Calculate the NCE loss for all specified layers.

        Parameters
        ----------
        feat_kq : list of torch.Tensor
            List of feature tensors from the base network.
        seg : torch.Tensor
            Segmentation mask for the input.
        step : int, optional
            Training step (default is -1, not used).
        vis : bool, optional
            Whether to visualize (default is False).

        Returns
        -------
        total_nce_loss : float
            The total NCE loss (weighted sum over layers).
        layer_wise_dict : OrderedDict
            Dictionary of per-layer NCE losses.
        """
        # Pool features and sample patch indices
        feat_kq_pool, sample_ids = self.netF(
            feat_kq, self.opt.num_patches, None, None, False
        )  # verbose=self.verbose)
        total_nce_loss = 0.0
        layer_wise_dict = OrderedDict()

        # Compute NCE loss for each layer
        for f_kq, sample_id, crit, nce_layer, nce_w, feat in zip(
            feat_kq_pool,
            sample_ids,
            self.criterionNCE,
            self.nce_layers,
            self.nce_weights,
            feat_kq,
        ):
            loss = crit(f_kq, seg, sample_id, feat.size()[2:])
            total_nce_loss += loss.mean() * nce_w * self.opt.lambda_NCE
            layer_wise_dict[str(nce_layer)] = float(loss.mean().item())

        return total_nce_loss, layer_wise_dict

    def get_current_losses(self):
        """
        Return current training losses/errors.

        Returns
        -------
        OrderedDict
            Dictionary of current losses (for printing and logging).
        """
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, "loss_" + name))
        if self.opt.lambda_NCE > 0:
            errors_ret.update(self.nce_dict)
        return errors_ret

    def get_current_visuals(self):
        """
        Return current visual outputs for visualization.

        Returns
        -------
        OrderedDict
            Dictionary of current visual outputs (e.g., images, segmentations).
        """
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                tmp = getattr(self, name)
                if tmp is not None:
                    print(name, tmp.size(), tmp.min().item(), tmp.max().item())
                    visual_ret[name] = tmp
        return visual_ret

    def train(self):
        """Make models train mode during train time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                net.train()
                Norm = networks.get_norm_layer(self.opt.ndims, self.opt.normG)
                if name == "G" and len(self.unfreeze_layers) > 0:
                    for layer_id, layer in enumerate(net.model):
                        if str(
                            layer_id
                        ) not in self.unfreeze_layers and isinstance(
                            layer, Norm
                        ):
                            print("Freezing BN stats {}".format(layer_id))
                            layer.eval()
