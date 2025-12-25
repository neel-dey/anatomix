# Reference:
# CUT (https://github.com/taesungp/contrastive-unpaired-translation)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.optim import lr_scheduler

from anatomix.model.network import (
    Unet,
    get_actvn_layer,
    get_norm_layer,
)


# -------------------------------
# Base UNet, just wraps the main anatomix model
# -------------------------------

def define_G(
    dimension,
    input_nc,
    output_nc,
    ngf,
    netG,
    norm="batch",
    use_dropout=False,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    opt=None,
    final_act="none",
    activation="relu",
    pooling="Max",
    interp="nearest",
    num_downs=4,
):
    """
    Create a base network to pretrain.

    Parameters
    ----------
    dimension : int
        Number of spatial dimensions (2 or 3).
    input_nc : int
        Number of channels in input images.
    output_nc : int
        Number of channels in output features.
    ngf : int
        Number of filters in the first conv layer.
    netG : str
        The architecture's name: 'unet'.
    norm : str, optional
        The name of normalization layers used in the network: 'batch' | 'instance' | 'none'.
        Default is 'batch'.
    use_dropout : bool, optional
        If True, use dropout layers. Default is False. Unused in code.
    init_type : str, optional
        The name of the initialization method. Default is 'normal'.
    init_gain : float, optional
        Scaling factor for normal, xavier and orthogonal. Default is 0.02.
    gpu_ids : list of int, optional
        Which GPUs the network runs on. Default is [].
    opt : object, optional
        Additional options (not used here).
    final_act : str, optional
        Final activation function. Default is 'none'.
    activation : str, optional
        Activation function. Default is 'relu'.
    pooling : str, optional
        Pooling type. Default is 'Max'.
    interp : str, optional
        Interpolation type. Default is 'nearest'.
    num_downs : int, optional
        Number of downsamples in encoder. Default is 4.

    Returns
    -------
    net : nn.Module
        Initialized network to pretrain.

    """
    net = None

    if netG == "unet":
        net = Unet(
            dimension,
            input_nc,
            output_nc,
            num_downs=num_downs,
            ngf=ngf,
            norm=norm,
            activation=activation,
            final_act=final_act,
            pad_type="reflect",
            doubleconv=True,
            residual_connection=False,
            use_skip_connection=True,
            pooling=pooling,
            interp=interp,
        )
    else:
        raise NotImplementedError(
            "Base network name [%s] is not recognized" % netG
        )
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=True)


# -------------------------------
# Patch Sampling MLP
# -------------------------------

def define_F(
    input_nc,
    netF,
    norm="batch",
    init_type="normal",
    init_gain=0.02,
    n_mlps=2,
    gpu_ids=[],
    opt=None,
    activation="relu",
    use_mlp=True,
):
    """
    Create a patch sampling network (feature projector).

    Parameters
    ----------
    input_nc : int
        Number of input channels.
    netF : str
        The architecture's name: 'mlp_sample'.
    norm : str, optional
        Normalization type. Default is 'batch'.
    init_type : str, optional
        Initialization method. Default is 'normal'.
    init_gain : float, optional
        Initialization gain. Default is 0.02.
    n_mlps : int, optional
        Number of MLP layers. Default is 2.
    gpu_ids : list of int, optional
        List of GPU ids. Default is [].
    opt : object, optional
        Options object, must have attribute 'netF_nc'.
    activation : str, optional
        Activation function. Default is 'relu'.
    use_mlp : bool, optional
        Whether to use MLP. Default is True.

    Returns
    -------
    net : nn.Module
        Initialized patch sampling network.

    Raises
    ------
    NotImplementedError
        If netF is not recognized.
    """
    if netF == "mlp_sample":
        net = PatchSampleF(
            use_mlp=use_mlp,
            init_type=init_type,
            init_gain=init_gain,
            gpu_ids=gpu_ids,
            nc=opt.netF_nc,
            n_mlps=n_mlps,
            activation=activation,
            norm=norm,
        )
    else:
        raise NotImplementedError(
            "projection model name [%s] is not recognized" % netF
        )
    return init_net(net, init_type, init_gain, gpu_ids)


class Normalize(nn.Module):
    """
    Lp normalization layer.

    Parameters
    ----------
    power : int, optional
        The power for the norm (default is 2 for L2 norm).
    eps : float, optional
        Small value to avoid division by zero (default is 1e-7).
    """
    def __init__(self, power=2, eps=1e-7):
        super(Normalize, self).__init__()
        self.power = power
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, C).

        Returns
        -------
        out : torch.Tensor
            Normalized tensor of same shape as input.
        """
        assert len(x.size()) == 2, "wrong shape {} for L2-Norm".format(x.size())
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm + self.eps)
        return out


class PatchSampleF(nn.Module):
    """
    Patch sampling and feature projection module.

    Parameters
    ----------
    use_mlp : bool, optional
        Whether to use MLP for feature projection. Default is False.
    init_type : str, optional
        Initialization method. Default is 'normal'.
    init_gain : float, optional
        Initialization gain. Default is 0.02.
    nc : int, optional
        Feature dimension. Default is 256.
    gpu_ids : list of int, optional
        List of GPU ids. Default is [].
    n_mlps : int, optional
        Number of MLP layers. Default is 2.
    activation : str, optional
        Activation function. Default is 'relu'.
    norm : str, optional
        Normalization type. Default is 'batch'.
    """
    def __init__(
        self,
        use_mlp=False,
        init_type="normal",
        init_gain=0.02,
        nc=256,
        gpu_ids=[],
        n_mlps=2,
        activation="relu",
        norm="batch",
    ):
        # use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        print("Use MLP: {}".format(use_mlp))
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids
        self.n_mlps = n_mlps
        self.activation = activation
        self.normtype = norm

    def create_mlp(self, feats):
        """
        Create MLP layers for each feature map.

        Parameters
        ----------
        feats : list of torch.Tensor
            List of feature maps to determine input dimensions.
        """
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            norm = get_norm_layer(1, self.normtype)
            Activation = get_actvn_layer(self.activation)

            if self.n_mlps == 2:
                mlp = nn.Sequential(
                    *[
                        nn.Linear(input_nc, self.nc, bias=False),
                        norm(self.nc),
                        Activation,
                        nn.Linear(self.nc, self.nc, bias=False),
                        norm(self.nc, affine=False),
                    ]
                )
            elif self.n_mlps == 3:
                mlp = nn.Sequential(
                    *[
                        nn.Linear(input_nc, self.nc, bias=False),
                        norm(self.nc),
                        Activation,
                        nn.Linear(self.nc, self.nc, bias=False),
                        norm(self.nc),
                        Activation,
                        nn.Linear(self.nc, self.nc, bias=False),
                        norm(self.nc, affine=False),
                    ]
                )
            else:
                raise NotImplementedError

            if len(self.gpu_ids) > 0:
                mlp.cuda()

            setattr(self, "mlp_%d" % mlp_id, mlp)
            print("mlp_%d created, input nc %d" % (mlp_id, input_nc))

        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(
        self, feats, num_patches=64, patch_ids=None, mask=None, verbose=False
    ):
        """
        Forward pass for patch sampling and feature projection.

        Parameters
        ----------
        feats : list of torch.Tensor
            List of feature maps to sample from.
        num_patches : int, optional
            Number of patches to sample per feature map. Default is 64.
        patch_ids : list of torch.Tensor, optional
            List of patch indices to use for sampling. If None, random sampling is used.
        mask : torch.Tensor, optional
            Foreground mask for sampling. If None, all locations are considered.
        verbose : bool, optional
            If True, print debug information. Default is False.

        Returns
        -------
        return_feats : list of torch.Tensor
            List of sampled and projected features for each feature map.
        return_ids : list
            List of coordinates of sampled patches for each feature map.
        """
        return_ids = []
        return_feats = []

        if verbose:
            print(f"Net F forward pass: # features: {len(feats)}")

        ndims = len(feats[0].size()[2:])
        if mask is not None:
            if verbose:
                print(f"Using foreground mask {mask.size()}")
            masks = [
                F.interpolate(mask, size=f.size()[2:], mode="nearest").cuda()
                for f in feats
            ]  # TODO: should deal with devices and not just .cuda()
        else:
            masks = [
                torch.ones(f.size()[2:]).unsqueeze(0).unsqueeze(0).cuda()
                for f in feats
            ]  # TODO: should deal with devices and not just .cuda()

        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)

        for feat_id, feat in enumerate(feats):
            if verbose:
                print(feat_id, "input -> {}".format(feat.size()))

            if num_patches > 0:
                if patch_ids is not None:  # sample based on given index
                    patch_id = patch_ids[feat_id]
                    # patch_id is a torch tensor (share idx across batch)
                    if ndims == 3:
                        x_sample = feat[
                            :, :, patch_id[:, 0], patch_id[:, 1], patch_id[:, 2]
                        ]
                    elif ndims == 2:
                        x_sample = feat[:, :, patch_id[:, 0], patch_id[:, 1]]
                    else:
                        raise NotImplementedError

                    if verbose:
                        print(
                            "Sample basd on given {} idx w/o mask: sample shape: {}".format(
                                len(patch_id), x_sample.size()
                            )
                        )

                else:  # sample patch index
                    mask_i = masks[feat_id]

                    # TODO: this is very inefficient and unneccessary because
                    # for a mask_i of (1, 1, 16, 16, 16), it'll store a tuple of
                    # shape (4096, 4096, 4096, 4096, 4096). Should just remove all
                    # foreground masking, we dont need it for this paper.
                    fg_coords = torch.where(mask_i > 0)
                    if ndims == 3:
                        (_, _, fg_x, fg_y, fg_z) = fg_coords
                    elif ndims == 2:
                        (_, _, fg_x, fg_y) = fg_coords
                    else:
                        raise NotImplementedError

                    # Rand selection algo: Just randomly permute all fg indices
                    # and then take first num_patches
                    patch_id = torch.randperm(
                        fg_x.shape[0], device=feats[0].device
                    )
                    patch_id = patch_id[
                        : int(min(num_patches, patch_id.shape[0]))
                    ]

                    select_x, select_y = fg_x[patch_id], fg_y[patch_id]
                    if ndims == 3:
                        select_z = fg_z[patch_id]
                        coords = torch.cat(
                            (
                                select_x.unsqueeze(1),
                                select_y.unsqueeze(1),
                                select_z.unsqueeze(1),
                            ),
                            dim=1,
                        )
                        x_sample = feat[:, :, select_x, select_y, select_z]

                    elif ndims == 2:
                        coords = torch.cat(
                            (select_x.unsqueeze(1), select_y.unsqueeze(1)),
                            dim=1,
                        )
                        x_sample = feat[:, :, select_x, select_y]

                    else:
                        raise NotImplementedError

                    if verbose:
                        print(
                            "Masked sampling, patch_id: {} sample shape: {}".format(
                                len(patch_id), x_sample.size()
                            )
                        )

            else:
                x_sample = feat
                coords = []

            nviews, nc, nsample = x_sample.size()
            x_sample = x_sample.permute(0, 2, 1).flatten(
                0, 1
            )  # nviews*nsample, nc
            # print(x_sample.size())
            return_ids.append(coords)

            if self.use_mlp:
                mlp = getattr(self, "mlp_%d" % feat_id)

                x_sample = mlp(x_sample)
                x_sample = x_sample.view(nviews, nsample, -1)

            if verbose:
                print("MLP + reshape: {}".format(x_sample.size()))
                print(
                    "feature range ",
                    feat_id,
                    x_sample.min().item(),
                    x_sample.max().item(),
                )

                print("\n\n")
            return_feats.append(x_sample)

        return return_feats, return_ids


# -------------------------------
# LR Scheduling
# -------------------------------

def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Specify the optimizer to use.
    opt : object
        Stores all the experiment flags; needs to be a subclass of BaseOptions.
        opt.lr_policy is the name of learning rate policy: 
        'const_linear' | 'linear' | 'step' | 'plateau' | 'cosine'.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler._LRScheduler or ReduceLROnPlateau
        Learning rate scheduler.

    Notes
    -----
    For 'linear', the learning rate is kept constant for the first <opt.n_epochs> epochs
    and linearly decays to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), the default PyTorch schedulers are used.
    See https://pytorch.org/docs/stable/optim.html for more details.

    Raises
    ------
    NotImplementedError
        If the learning rate policy is not implemented.
    """
    if opt.lr_policy == "const_linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(
                opt.n_epochs_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "linear":
        scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=5e-2,
            total_iters=opt.n_epochs + opt.n_epochs_decay,
            last_epoch=-1,
        )
    elif opt.lr_policy == "exponential":
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.99)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.5
        )  # 0.1
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            threshold=1e-4,
            patience=5,
            min_lr=1e-7,
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.n_epochs, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    a certain number of epochs.

    Parameters
    ----------
    patience : int, optional
        How many epochs to wait before stopping when loss is not improving.
        Default is 5.
    min_delta : float, optional
        Minimum difference between new loss and old loss for new loss to be considered as an improvement.
        Default is 0.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        Initialize EarlyStopping.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait for improvement. Default is 5.
        min_delta : float, optional
            Minimum change to qualify as improvement. Default is 0.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call method to check if early stopping condition is met.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        None
        """
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}"
            )
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


# -------------------------------
# Network initialization
# -------------------------------

def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """
    Initialize network weights.

    Parameters
    ----------
    net : nn.Module
        Network to be initialized.
    init_type : str, optional
        The name of an initialization method: 'normal' | 'xavier' | 'kaiming' | 'orthogonal'.
        Default is 'normal'.
    init_gain : float, optional
        Scaling factor for normal, xavier and orthogonal. Default is 0.02.
    debug : bool, optional
        If True, print debug information. Default is False.

    Returns
    -------
    None
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
            or classname.find("BatchNorm3d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    debug=False,
    initialize_weights=True,
):
    """
    Initialize a network: 
    1. register CPU/GPU device (with multi-GPU support); 
    2. initialize the network weights.

    Parameters
    ----------
    net : nn.Module
        The network to be initialized.
    init_type : str, optional
        The name of an initialization method: 'normal' | 'xavier' | 'kaiming' | 'orthogonal'.
        Default is 'normal'.
    init_gain : float, optional
        Scaling factor for normal, xavier and orthogonal. Default is 0.02.
    gpu_ids : list of int, optional
        Which GPUs the network runs on. Default is [].
    debug : bool, optional
        If True, print debug information. Default is False.
    initialize_weights : bool, optional
        If True, initialize network weights. Default is True.

    Returns
    -------
    net : nn.Module
        Initialized network.
    """
    if len(gpu_ids) > 0:
        #import pdb; pdb.set_trace()
        assert torch.cuda.is_available()
        net.to("cuda")
        # if not amp:
        # net = torch.nn.DataParallel(net, "gpu_ids")  # multi-GPUs for non-AMP training
        #net = torch.nn.DataParallel(net, gpu_ids)
        #net = net.cuda()
    #    net = torch.nn.DataParallel(net, "cuda")
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net
