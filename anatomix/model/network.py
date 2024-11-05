# Taken and lightly modified from:
# CUT (https://github.com/taesungp/contrastive-unpaired-translation)

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Building blocks

class ConvBlock(nn.Module):
    """
    A convolutional block with optional normalization and activation.

    This block performs a convolution, followed by optional normalization 
    and activation. The block supports 1D, 2D, and 3D convolutions.

    Parameters
    ----------
    ndims : int
        Number of dimensions (1, 2, or 3) for the convolution.
    input_dim : int
        Number of channels in the input.
    output_dim : int
        Number of channels in the output.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple
        Stride of the convolution.
    bias : bool
        Whether to use a bias term in the convolution.
    padding : int or tuple, optional
        Amount of padding to add to the input, by default 0.
    norm : str, optional
        Type of normalization to apply ('batch', 'instance', or 'none'), 
        by default 'none'.
    activation : str, optional
        Activation function to use ('relu', 'lrelu', 'elu', 'prelu', 'selu', 
        'tanh', or 'none'), by default 'relu'.
    pad_type : str, optional
        Type of padding to use ('zeros', 'reflect', etc.), by default 'zeros'.

    """

    def __init__(
        self, ndims, input_dim, output_dim, kernel_size, stride, bias,
        padding=0, norm='none', activation='relu', pad_type='zeros',
    ):
        """
        Initialize the ConvBlock with convolution, normalization, and 
        activation layers.

        Parameters are described in the class docstring.
        """
        super(ConvBlock, self).__init__()
        self.use_bias = bias
        assert ndims in [1, 2, 3], 'ndims in 1--3. found: %d' % ndims
        Conv = getattr(nn, 'Conv%dd' % ndims)

        # initialize convolution
        self.conv = Conv(
            input_dim, output_dim, kernel_size, stride, bias=self.use_bias,
            padding=padding, padding_mode=pad_type
        )

        # initialize normalization
        norm_dim = output_dim
        if norm == 'batch':
            self.norm = getattr(nn, 'BatchNorm%dd'%ndims)(norm_dim)
        elif norm == 'instance':
            self.norm = getattr(
                nn, 'InstanceNorm%dd'%ndims
            )(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)


    def forward(self, x):
        """
        Perform the forward pass through the ConvBlock.

        Applies convolution, followed by optional normalization and 
        activation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the block.

        Returns
        -------
        torch.Tensor
            The output tensor after applying convolution, normalization, 
            and activation.
        """

        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


def get_norm_layer(ndims, norm='batch'):
    """
    Get the normalization layer based on the number of dimensions and type of
    normalization.

    Parameters
    ----------
    ndims : int
        The number of dimensions for the normalization layer (1--3).
    norm : str, optional
        The type of normalization to use. 
        Options are 'batch', 'instance', or 'none'. 
        Default is 'batch'.

    Returns
    -------
    Norm : torch.nn.Module or None
        The corresponding PyTorch normalization layer, or None if 'none'.
    """

    if norm == 'batch':
        Norm = getattr(nn, 'BatchNorm%dd' % ndims)
    elif norm == 'instance':
        Norm = getattr(nn, 'InstanceNorm%dd' % ndims)
    elif norm == 'none':
        Norm = None
    else:
        assert 0, "Unsupported normalization: {}".format(norm)
    return Norm


def get_actvn_layer(activation='relu'):
    """
    Get the activation function layer based on the provided activation type.

    Parameters
    ----------
    activation : str, optional
        The type of activation function to use. 
        Options are 'relu', 'lrelu', 'elu',  'prelu', 'selu', 'tanh', or 'none'
        Default is 'relu'.

    Returns
    -------
    Activation : torch.nn.Module or None
        The corresponding PyTorch activation layer, or None if 'none'.
    """

    if activation == 'relu':
        Activation = nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        Activation = nn.LeakyReLU(0.3, inplace=True)
    elif activation == 'elu':
        Activation = nn.ELU()
    elif activation == 'prelu':
        Activation = nn.PReLU()
    elif activation == 'selu':
        Activation = nn.SELU(inplace=True)
    elif activation == 'tanh':
        Activation = nn.Tanh()
    elif activation == 'none':
        Activation = None
    else:
        assert 0, "Unsupported activation: {}".format(activation)
    return Activation

################
# Network
################
class Unet(nn.Module):
    """
    U-Net architecture for image-to-image translation.

    This class constructs a U-Net with configurable depth, filter sizes, 
    normalization, and activation layers.

    Parameters
    ----------
    dimension : int
        The number of dimensions (1, 2, or 3) for the input and convolution 
        operations.
    input_nc : int
        Number of channels in the input image.
    output_nc : int
        Number of channels in the output image.
    num_downs : int
        Number of downsampling operations in the U-Net architecture.
        For example, if `num_downs == 7`, an input image of size 128x128 
        becomes 1x1 at the bottleneck.
    ngf : int, optional
        Number of filters in the last convolutional layer, by default 24.
    norm : str, optional
        Type of normalization to use ('batch', 'instance', or 'none'), 
        by default 'batch'.
    final_act : str, optional
        Activation function to apply at the output layer, by default 'none'.
    activation : str, optional
        Activation function to use in hidden layers ('relu', 'lrelu', 
        'elu', etc.), by default 'relu'.
    pad_type : str, optional
        Padding type to use in convolution layers ('reflect', 'zero', etc.), 
        by default 'reflect'.
    doubleconv : bool, optional
        Whether to apply double convolution in each block, default True.
    residual_connection : bool, optional
        Whether to add residual connections within the network, default False.
    pooling : str, optional
        Pooling type to use ('Max' or 'Avg'), by default 'Max'.
    interp : str, optional
        Upsampling method for the decoder ('nearest' or 'trilinear'), 
        by default 'nearest'.
    use_skip_connection : bool, optional
        Whether to use skip connections between corresponding encoder and 
        decoder layers, by default True.

    """

    def __init__(
        self, dimension, input_nc, output_nc, num_downs, ngf=24, norm='batch',
        final_act='none', activation='relu', pad_type='reflect', 
        doubleconv=True, residual_connection=False, 
        pooling='Max', interp='nearest', use_skip_connection=True,
    ):
        """
        Initialize the U-Net model by constructing the architecture from the
        innermost to the outermost layers.

        Parameters are described in the class docstring.
        """
        super(Unet, self).__init__()
        # Check dims
        ndims = dimension
        assert ndims in [1, 2, 3], 'ndims should be 1--3. found: %d' % ndims
        
        # Decide whether to use bias based on normalization type
        use_bias = norm == 'instance'
        self.use_bias = use_bias

        # Get the appropriate convolution and pooling layers for the given dim
        Conv = getattr(nn, 'Conv%dd' % ndims)
        Pool = getattr(nn, '%sPool%dd' % (pooling,ndims))

        # Initialize normalization, activation, and final activation layers
        Norm = get_norm_layer(ndims, norm)
        Activation = get_actvn_layer(activation)
        FinalActivation = get_actvn_layer(final_act)

        self.residual_connection = residual_connection
        self.res_dest = []  # List to track destination layers for residuals
        self.res_source  = []  # List to track source layers for residuals

        # Start creating the model
        model = [
            Conv(
                input_nc,
                ngf,
                3,
                stride=1,
                bias=use_bias,
                padding='same',
                padding_mode=pad_type,
            )
        ]
        self.res_source += [len(model)-1]
        if Norm is not None:
            model += [Norm(ngf)]
            
        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        # Initialize encoder-related variables
        self.use_skip_connection = use_skip_connection
        self.encoder_idx = []
        in_ngf = ngf
        
        # Create the downsampling (encoder) blocks
        for i in range(num_downs):
            if i == 0:
                mult = 1
            else:
                mult = 2
            model += [
                Conv(
                    in_ngf, in_ngf * mult, kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * mult)]

            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                model += [
                    Conv(
                        in_ngf * mult, in_ngf * mult, kernel_size=3, stride=1,
                        bias=use_bias, padding='same', padding_mode=pad_type,
                    )
                ]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(in_ngf * mult)]
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            self.encoder_idx += [len(model) - 1]
            model += [Pool(2)]
            in_ngf = in_ngf * mult


        model += [
            Conv(
                in_ngf, in_ngf * 2, kernel_size=3, stride=1, bias=use_bias,
                padding='same', padding_mode=pad_type,
            )
        ]
        self.res_source += [len(model) - 1]
        if Norm is not None:
            model += [Norm(in_ngf * 2)]

        if Activation is not None:
            model += [Activation]
        self.res_dest += [len(model) - 1]

        if doubleconv:
            #self.conv_id += [len(model)]
            model += [
                Conv(
                    in_ngf * 2, in_ngf * 2, kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(in_ngf * 2)]
    
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

        # Create the upsampling (decoder) blocks
        self.decoder_idx = []
        mult = 2 ** (num_downs)
        for i in range(num_downs):
            self.decoder_idx += [len(model)]
            model += [nn.Upsample(scale_factor=2, mode=interp)]
            if self.use_skip_connection:  # concatenate encoder/decoder feature
                m = mult + mult // 2
            else:
                m = mult
            model += [
                Conv(
                    ngf * m, ngf * (mult // 2), kernel_size=3, stride=1,
                    bias=use_bias, padding='same', padding_mode=pad_type,
                )
            ]
            self.res_source += [len(model) - 1]
            if Norm is not None:
                model += [Norm(ngf * (mult // 2))]
            if Activation is not None:
                model += [Activation]
            self.res_dest += [len(model) - 1]

            if doubleconv:
                model += [
                    Conv(
                        ngf * (mult // 2),
                        ngf * (mult // 2),
                        kernel_size=3,
                        stride=1,
                        bias=use_bias,
                        padding='same',
                        padding_mode=pad_type,
                    )
                ]
                self.res_source += [len(model) - 1]
                if Norm is not None:
                    model += [Norm(ngf * (mult // 2))]
   
                if Activation is not None:
                    model += [Activation]
                self.res_dest += [len(model) - 1]

            mult = mult // 2

        print('Encoder skip connect id', self.encoder_idx)
        print('Decoder skip connect id', self.decoder_idx)

        Conv = getattr(nn, 'Conv%dd' % ndims)
        # final conv w/o normalization layer
        model += [
            Conv(
                ngf * mult,
                output_nc,
                kernel_size=3,
                stride=1,
                bias=use_bias,
                padding='same',
                padding_mode=pad_type,
            )
        ]
        if FinalActivation is not None:
            model += [FinalActivation]
        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False, verbose=False):
        if len(layers) == 0:
            """Standard forward"""
            enc_feats = []
            feat = input
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer.__class__.__name__)
                feat = layer(feat)
                if self.residual_connection and layer_id in self.res_source:
                    feat_tmp = feat
                if self.residual_connection and layer_id in self.res_dest:
                    assert feat_tmp.size() == feat.size()
                    feat = feat + 0.1 * feat_tmp
                
                if self.use_skip_connection:
                    if layer_id in self.decoder_idx:
                        feat = torch.cat((enc_feats.pop(), feat), dim=1)
                    if layer_id in self.encoder_idx:
                        enc_feats.append(feat)
            return feat
        else:
            raise NotImplementedError
