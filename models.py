import torch
import constants
import numpy as np


class PointNetModule(torch.nn.Module):
    """Implementation of a basic PointNet module.
    Implemented as a 2D Convolution over the point cloud sequence cube with (1,1) kernels
    (where point features can be seen as image channels), can be seen as a simple one layer MLP
    applied to each point separately, using a single set of shared parameters (i.e. the convolutional
    filter learnable parameters).
    This module also applies Batch Norm and activation function.

    - Input: (b_size, in_channels, seq_length, n_points)
    - Output: (b_size, out_chs, seq_length, n_points)
    """

    def __init__(self, in_chs, out_chs, activation=torch.nn.ELU()):
        super(PointNetModule, self).__init__()
        self.module = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chs,
                out_chs,
                (1, 1),
                stride=1,
                padding="valid",
                dilation=1,
            ),
            torch.nn.BatchNorm2d(num_features=out_chs),
            activation,
        )

    def forward(self, x):
        return self.module(x)


class DilTempConv1d(torch.nn.Module):
    """Dilated Temporal Convolution module, implemented by normally adding `n` padding elements,
    then removing last n outputs of the classical 1d convolution.
    Nice explanation at:
    https://programs.wiki/wiki/the-difference-between-conventional-convolution-and-causal-convolution.html

    This module also applies Batch Norm and activation function.
    """

    def __init__(
        self,
        in_chs,
        out_chs,
        dilation,
        kernel_size=3,
        stride=1,
        use_bias=True,
        activation=torch.nn.ELU(),
    ):

        super(DilTempConv1d, self).__init__()

        self.padding = int(np.floor((kernel_size - 1) * dilation))
        self.conv1d = torch.nn.Conv1d(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=dilation,
            bias=True,
        )

        self.activation = activation
        self.batch_norm = torch.nn.BatchNorm1d(out_chs)

    def forward(self, x):
        """Input x is of shape (batch_size, channels, time)"""
        x = self.conv1d(x)
        x = x[:, :, : -self.padding]
        x = self.batch_norm(x)
        out = self.activation(x)
        return out


class PointNetBlock(torch.nn.Module):
    def __init__(self):
        super(PointNetBlock, self).__init__()

        self.pointnet1 = PointNetModule(
            in_chs=constants.NFEATURES, out_chs=constants.POINTNET_OUT_DIM // 2
        )
        self.pointnet2 = PointNetModule(
            in_chs=constants.POINTNET_OUT_DIM // 2,
            out_chs=constants.POINTNET_OUT_DIM // 2,
        )
        self.pointnet3 = PointNetModule(
            in_chs=constants.POINTNET_OUT_DIM // 2, out_chs=constants.POINTNET_OUT_DIM
        )
        self.pointnet4 = PointNetModule(
            in_chs=constants.POINTNET_OUT_DIM, out_chs=constants.POINTNET_OUT_DIM
        )

    def forward(self, x):
        x = self.pointnet1(x)
        x = self.pointnet2(x)
        x = self.pointnet3(x)
        out = self.pointnet4(x)
        return out


class TemporalConvolutionBlock(torch.nn.Module):
    def __init__(self):
        super(TemporalConvolutionBlock, self).__init__()
        self.dtc1 = DilTempConv1d(
            in_chs=constants.POINTNET_OUT_DIM,
            out_chs=constants.DTC_FILTERS[0],
            dilation=1,
            kernel_size=3,
        )

        self.dtc2 = DilTempConv1d(
            in_chs=constants.DTC_FILTERS[0],
            out_chs=constants.DTC_FILTERS[1],
            dilation=2,
            kernel_size=3,
        )

        self.dtc3 = DilTempConv1d(
            in_chs=constants.DTC_FILTERS[1],
            out_chs=constants.DTC_FILTERS[2],
            dilation=4,
            kernel_size=3,
        )

        self.dtc4 = DilTempConv1d(
            in_chs=constants.DTC_FILTERS[2],
            out_chs=constants.DTC_FILTERS[3],
            dilation=1,
            kernel_size=3,
        )

        self.dtc5 = DilTempConv1d(
            in_chs=constants.DTC_FILTERS[3],
            out_chs=constants.DTC_FILTERS[4],
            dilation=2,
            kernel_size=3,
        )

        self.dtc6 = DilTempConv1d(
            in_chs=constants.DTC_FILTERS[4],
            out_chs=constants.DTC_FILTERS[5],
            dilation=4,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.dtc1(x)
        x = self.dtc2(x)
        x = self.dtc3(x)
        x = self.dtc4(x)
        x = self.dtc5(x)
        out = self.dtc6(x)
        return out


class Encoder(torch.nn.Module):
    def __init__(self, n_out_labels):
        super(Encoder, self).__init__()

        # Point Cloud block
        self.pc_block = PointNetBlock()

        # Global Avarage pooling on points dimension (NMAX)
        self.glob_avg_pool1 = torch.nn.AvgPool2d(
            kernel_size=(1, constants.NMAX))

        # Dilated temporal convolution (or some time dependence modeling network)
        self.tc_block = TemporalConvolutionBlock()

        # Global Average pooling on time dimension (NSTEPS) -> outsize = (batch, channels)
        self.glob_avg_pool2 = torch.nn.AvgPool1d(kernel_size=constants.NSTEPS)

        # Final Dense Layers
        self.MLP_sup1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.DTC_FILTERS[-1],
                out_features=constants.SUP_LATENT_DIM,
            ),
            torch.nn.Dropout(p=0.2),
            torch.nn.ELU(),
        )

        self.MLP_sup2 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.SUP_LATENT_DIM,
                out_features=n_out_labels,
            ),
            torch.nn.Dropout(p=0.2),
            torch.nn.ELU(),
        )

        self.MLP_unsupervised = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.DTC_FILTERS[-1],
                out_features=constants.UNSUP_LATENT_DIM // 4,
            ),
            torch.nn.Dropout(p=0.2),
            torch.nn.ELU(),
            torch.nn.Linear(
                in_features=constants.UNSUP_LATENT_DIM // 4,
                out_features=constants.UNSUP_LATENT_DIM // 2,
            ),
            torch.nn.Dropout(p=0.2),
            torch.nn.ELU(),
            torch.nn.Linear(
                in_features=constants.UNSUP_LATENT_DIM // 2,
                out_features=constants.UNSUP_LATENT_DIM,
            ),
            torch.nn.ELU(),
        )

    def forward(self, x):
        x1 = self.pc_block(x)
        # Squeeze is to remove dimensions
        x2 = torch.squeeze(self.glob_avg_pool1(x1), dim=-1)
        x3 = self.tc_block(x2)
        x4 = torch.squeeze(self.glob_avg_pool2(x3), dim=-1)
        sup_fv = self.MLP_sup1(x4)
        out_classes = self.MLP_sup2(sup_fv)
        unsup_fv = self.MLP_unsupervised(x4)

        return out_classes, sup_fv, unsup_fv


class CGEncoder(torch.nn.Module):
    def __init__(self, n_out_labels, nmax_points=constants.NMAX, use_projection_head=False):
        super(CGEncoder, self).__init__()

        self.use_projection_head = use_projection_head

        # Point Cloud block
        self.pc_block = PointNetBlock()

        # Global Avarage pooling on points dimension (NMAX)
        self.glob_avg_pool1 = torch.nn.AvgPool2d(
            kernel_size=(1, nmax_points))

        # Dilated temporal convolution
        self.tc_block = TemporalConvolutionBlock()

        # Global Average pooling on time dimension (NSTEPS) -> outsize = (batch, channels)
        self.glob_avg_pool2 = torch.nn.AvgPool1d(kernel_size=constants.NSTEPS)

        # Final Dense Layers
        self.MLP_sup1 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.DTC_FILTERS[-1],
                out_features=constants.SUP_LATENT_DIM,
            ),
            torch.nn.ELU(),
        )

        head_out_dimension = constants.SUP_LATENT_DIM if not use_projection_head else constants.SUP_LATENT_DIM // 2

        if self.use_projection_head:
            self.MLP_head = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=constants.SUP_LATENT_DIM,
                    out_features=head_out_dimension,
                ),
                torch.nn.ELU(),
            )

        self.MLP_sup2 = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=head_out_dimension,
                out_features=n_out_labels,
            ),
            torch.nn.ELU(),
        )

    def forward(self, x):
        x1 = self.pc_block(x)
        # Squeeze is to remove dimensions
        x2 = torch.squeeze(self.glob_avg_pool1(x1), dim=-1)
        x3 = self.tc_block(x2)
        x4 = torch.squeeze(self.glob_avg_pool2(x3), dim=-1)
        sup_fv = self.MLP_sup1(x4)
        if self.use_projection_head:
            proj_sup_fv = self.MLP_head(sup_fv)
            out_classes = self.MLP_sup2(proj_sup_fv)
        else:
            out_classes = self.MLP_sup2(sup_fv)

        return out_classes, sup_fv


class Decoder(torch.nn.Module):
    def __init__(self, n_in_labels):
        super(Decoder, self).__init__()

        self.activation = torch.nn.ELU()
        self.dense1 = torch.nn.Linear(
            in_features=constants.UNSUP_LATENT_DIM + n_in_labels,
            out_features=constants.DEC_MLP_SIZE // 16,
        )
        self.bn1 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 16)
        self.dense2 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 16,
            out_features=constants.DEC_MLP_SIZE // 8,
        )
        self.bn2 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 8)
        self.dense3 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 8,
            out_features=constants.DEC_MLP_SIZE // 4,
        )
        self.bn3 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 4)
        self.dense4 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 4,
            out_features=constants.DEC_MLP_SIZE // 2,
        )
        self.bn4 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 2)
        self.dense5 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 2, out_features=constants.DEC_MLP_SIZE
        )
        self.bn5 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 16)

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)
        x = self.activation(x)
        x = self.dense5(x)
        out = x.view(-1, constants.NFEATURES, constants.NSTEPS, constants.NMAX)
        out = torch.nn.Tanh()(out)
        return out


class CGDecoder(torch.nn.Module):
    def __init__(self, input_dim=constants.SUP_LATENT_DIM, nmax_points=constants.NMAX):
        super(CGDecoder, self).__init__()

        self.decoder_mlp_size = constants.NSTEPS * constants.NFEATURES * nmax_points

        self.nmax_points = nmax_points

        self.activation = torch.nn.ELU()
        self.dense1 = torch.nn.Linear(
            in_features=input_dim,
            out_features=self.decoder_mlp_size // 16,
        )
        self.bn1 = torch.nn.BatchNorm1d(self.decoder_mlp_size // 16)
        self.dense2 = torch.nn.Linear(
            in_features=self.decoder_mlp_size // 16,
            out_features=self.decoder_mlp_size // 8,
        )
        self.bn2 = torch.nn.BatchNorm1d(self.decoder_mlp_size // 8)
        self.dense3 = torch.nn.Linear(
            in_features=self.decoder_mlp_size // 8,
            out_features=self.decoder_mlp_size // 4,
        )
        self.bn3 = torch.nn.BatchNorm1d(self.decoder_mlp_size // 4)
        self.dense4 = torch.nn.Linear(
            in_features=self.decoder_mlp_size // 4,
            out_features=self.decoder_mlp_size // 2,
        )
        self.bn4 = torch.nn.BatchNorm1d(self.decoder_mlp_size // 2)
        self.dense5 = torch.nn.Linear(
            in_features=self.decoder_mlp_size // 2, out_features=self.decoder_mlp_size
        )

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)
        x = self.activation(x)
        x = self.dense5(x)
        out = x.view(-1, constants.NFEATURES,
                     constants.NSTEPS, self.nmax_points)
        return out


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(constants.UNSUP_LATENT_DIM, 64, bias=True),
            torch.nn.ELU(),
            torch.nn.Linear(64, 32, bias=True),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class CGDiscriminator(torch.nn.Module):
    def __init__(self, n_in_labels):
        super(CGDiscriminator, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(constants.SUP_LATENT_DIM +
                            n_in_labels, 64, bias=True),
            torch.nn.ELU(),
            torch.nn.Linear(64, 32, bias=True),
            torch.nn.ELU(),
            torch.nn.Linear(32, 1, bias=True),
        )

    def forward(self, x, label):
        x = torch.cat([x, label], dim=-1)
        out = self.model(x)
        return out


class GaussianMeanLearner(torch.nn.Module):
    def __init__(self, n_in_labels):
        super(GaussianMeanLearner, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_in_labels, 16, bias=True),
            torch.nn.BatchNorm1d(16),
            torch.nn.ELU(),
            torch.nn.Linear(16, 32, bias=True),
            torch.nn.BatchNorm1d(32),
            torch.nn.ELU(),
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(),
            torch.nn.Linear(64, constants.SUP_LATENT_DIM, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class ORCEDEncoder(torch.nn.Module):
    def __init__(self, n_out_labels):
        super(ORCEDEncoder, self).__init__()

        # Point Cloud block
        self.pc_block = PointNetBlock()

        # Global Avarage pooling on points dimension (NMAX)
        self.glob_avg_pool1 = torch.nn.AvgPool2d(
            kernel_size=(1, constants.NMAX))

        # Dilated temporal convolution
        self.tc_block = TemporalConvolutionBlock()

        # Global Average pooling on time dimension (NSTEPS) -> outsize = (batch, channels)
        self.glob_avg_pool2 = torch.nn.AvgPool1d(kernel_size=constants.NSTEPS)

        # 2 final layers to mu and std
        self.MLP_mu = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.DTC_FILTERS[-1],
                out_features=constants.SUP_LATENT_DIM,
            ),
        )

        self.MLP_logvar = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.DTC_FILTERS[-1],
                out_features=constants.SUP_LATENT_DIM,
            ),
        )

        self.MLP_classification = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=constants.SUP_LATENT_DIM,
                out_features=n_out_labels,
            ),
        )

    def forward(self, x):
        x1 = self.pc_block(x)
        # Squeeze is to remove dimensions
        x2 = torch.squeeze(self.glob_avg_pool1(x1), dim=-1)
        x3 = self.tc_block(x2)
        x4 = torch.squeeze(self.glob_avg_pool2(x3), dim=-1)
        vae_mu = self.MLP_mu(x4)
        vae_logvar = self.MLP_logvar(x4)

        # Reparametrization trick
        eps = torch.randn_like(vae_logvar)
        sup_fv = vae_mu + eps * torch.exp(0.5*vae_logvar)

        out_classes = self.MLP_classification(sup_fv)

        return out_classes, sup_fv, vae_mu, vae_logvar


class ORCEDDecoder(torch.nn.Module):
    def __init__(self, nmax_points=constants.NMAX):
        super(ORCEDDecoder, self).__init__()
        self.nmax_points = nmax_points

        self.activation = torch.nn.ELU()
        self.dense1 = torch.nn.Linear(
            in_features=constants.SUP_LATENT_DIM,
            out_features=constants.DEC_MLP_SIZE // 16,
        )
        self.bn1 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 16)
        self.dense2 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 16,
            out_features=constants.DEC_MLP_SIZE // 8,
        )
        self.bn2 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 8)
        self.dense3 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 8,
            out_features=constants.DEC_MLP_SIZE // 4,
        )
        self.bn3 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 4)
        self.dense4 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 4,
            out_features=constants.DEC_MLP_SIZE // 2,
        )
        self.bn4 = torch.nn.BatchNorm1d(constants.DEC_MLP_SIZE // 2)
        self.dense5 = torch.nn.Linear(
            in_features=constants.DEC_MLP_SIZE // 2, out_features=constants.DEC_MLP_SIZE
        )

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dense3(x)
        x = self.activation(x)
        x = self.dense4(x)
        x = self.activation(x)
        x = self.dense5(x)
        out = x.view(-1, constants.NFEATURES,
                     constants.NSTEPS, self.nmax_points)
        return out
