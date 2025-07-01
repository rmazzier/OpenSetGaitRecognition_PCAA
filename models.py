import torch
import constants
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import utils as ut

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
    def __init__(self, n_out_labels, nmax_points=constants.NMAX, use_projection_head=False, latent_dim=constants.SUP_LATENT_DIM):
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
                out_features=latent_dim,
            ),
            torch.nn.ELU(),
        )

        head_out_dimension = latent_dim if not use_projection_head else latent_dim // 2

        if self.use_projection_head:
            self.MLP_head = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=latent_dim,
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


class LGM_Net(nn.Module):
    """
    This is the new baseline to check against, as per reviewer request.
    Code draft to refine for model trained with LGM Loss, from this paper:
    https://ieeexplore.ieee.org/document/9750082 
    """
    def __init__(self, embedding_dim=16):
        super(LGM_Net, self).__init__()

        # 1. Pretrained ResNet-18 as feature extractor
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # remove last FC layer
        self.feature_extractor = nn.Sequential(*modules)  # outputs [B, 512, 1, 1]


        # 2. Fully connected layer for embedding
        self.fc = nn.Linear(512, embedding_dim)  # [B, 512] -> [B, embedding_dim]

    def forward(self, x):
        """
        x: input tensor [B, 3, H, W]
        return_embedding: if True, skip classification layer and return only embedding
        """
        x = self.feature_extractor(x)  # [B, 512, 1, 1]
        # print(f"ResNet Output: {x}")
        x = x.view(x.size(0), -1)      # [B, 512]
        embedding = self.fc(x)        # [B, embedding_dim]

        return embedding

class LGM_PCNet(nn.Module):
    """
    This is the adaptation of the LGM_Net architecture for point clouds
    """
    def __init__(self, enc_embedding_dim=128, final_embedding_dim=32, num_classes=20):
        super(LGM_PCNet, self).__init__()

        # 1. My CGEncoder feature extractor for point cloud sequences
        self.feature_extractor = CGEncoder(n_out_labels=num_classes, latent_dim=enc_embedding_dim)

        # 2. Fully connected layer for final embedding dim
        self.fc = nn.Linear(enc_embedding_dim, final_embedding_dim)

    def forward(self, x):
        """
        x: input tensor is a point cloud sequence of shape [B, SeqLength, Npoints, NFeatures]
        """
        _, x = self.feature_extractor(x)  
        # x = x.view(x.size(0), -1)     
        embeddings = self.fc(x)

        # logits = self.fc2(embedding)   # [B, num_classes]
        return embeddings


# Note also that in the training setup the class_means must be specified in the ooptimized as a trainable parameter
# --- Example Setup ---
# model = OpenSetGaitNet(embedding_dim=128, num_classes=20).to(device)
# criterion = LGM_Loss(feature_dim=128, num_classes=20, alpha=0.5, lambda_reg=0.1, margin=0.5).to(device)
# optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=1e-4)

class LGM_Loss(nn.Module):
    def __init__(self, feature_dim, num_classes, lambda_reg=0.5, margin=1.6):
        super(LGM_Loss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg  # likelihood regularization strength
        self.margin = margin          # margin value

        # Learnable class means (centroids)
        self.class_means = nn.Parameter(torch.randn(num_classes, feature_dim), requires_grad=True)

        # Shared log-variance (scalar), initialized to 0 (=> exp(0) = 1)
        # self.log_sigma2 = nn.Parameter(torch.zeros(1))
        # self.log_sigma2 = torch.zeros(1)
    
    def compute_posteriors(self, features, margin=1):
        """
        Given a tensor of features of shape [B, D], returns a tensor of posterior probabilities of shape [B, K]. 
        i.e. element at index (i,j) = P(class_j | features[i])
        """
        B, D = features.shape
        K = self.num_classes
        
        # 1. Compute squared Mahalanobis distances between features and class means in parallel for each element in the batch
        # Outputs tensor of shape [B, K]

        # ||x - mu||^2 = ||x||^2 + ||mu||^2 - 2x·mu
        x_norm = (features ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        mu_norm = (self.class_means ** 2).sum(dim=1, keepdim=True).T  # [1, K]
        cross_term = features @ self.class_means.T  # [B, K]

        # First addition here uses pytorch broadcasting
        dists_squared = x_norm + mu_norm - 2 * cross_term  # [B, K]

        # Add division by 2 like in the paper (why?)
        dists_squared *= 0.5

        # 2. Compute tensor of log posteriors P(y|x) with margin then CE Loss term (shape [B, K])
        posteriors = (1/K * torch.exp(-margin * dists_squared)) / ((1/K * torch.exp(-margin * dists_squared)).sum(dim=1)+ 1e-8).unsqueeze(1)
        # posteriors = F.softmax(-margin * dists_squared, dim=1)

        return posteriors

    def forward(self, features, labels):
        """
        features: [B, D]
        labels: [B]
        """
        B, D = features.shape
        K = self.num_classes
        
        posteriors = self.compute_posteriors(features, self.margin)

        index = torch.stack([torch.arange(B).to(constants.DEVICE), labels], dim=0).T
        # Here I choose for each sample in the batch, the poserior corresponding to its label 
        pys = posteriors[index[:,0], index[:,1]]

        log_py = torch.log(pys + 1e-32)

        ce_loss = -log_py.mean()  

        # 3. Likelihood regularization
        # Here I have to use a for loop, I don't think I can sample a tensor of distributions in pytorch
        log_lkd_sum = 0
        for b_i in range(B):
            mvn = torch.distributions.MultivariateNormal(self.class_means[labels[b_i]], covariance_matrix=torch.eye(D).to(constants.DEVICE))
            log_lkd = mvn.log_prob(features[b_i])
            log_lkd_sum += log_lkd
        
        loss_lkd = -log_lkd_sum

        return ce_loss + self.lambda_reg * loss_lkd, ce_loss, loss_lkd
    

## ------- Original OR-CED model for microdopplers, reference code taken from LVAE paper and adapted --------------------
## This implementation is monolithic, it has the whole loss function implemented as a method of the module
# I will adapt it to have minimal changes compared to the original source code, so it will remain similar in style

from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import DotProductSimilarity

class CONV(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, padding, stride, flat_dim, latent_dim):
        """
        Convolution + Flatten + 2 branches for mu and std.
        Used during upward pass.
        """

        super(CONV, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.flat_dim = flat_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, stride=stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(out_ch, affine=False),
            nn.PReLU()
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(out_ch*flat_dim*flat_dim, latent_dim)
        )

    def encode(self, x):
        h = self.net(x)
        h_flat = h.view(-1, self.out_ch*self.flat_dim*self.flat_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class TCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride, out_dim, t_latent_dim):
        super(TCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding
        self.out_dim = out_dim
        self.t_latent_dim = t_latent_dim

        self.fc = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.net = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            nn.BatchNorm2d(t_out_ch, affine=False),
        )
        self.mean_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )
        self.var_layer = nn.Sequential(
            nn.Linear(t_out_ch*out_dim*out_dim, t_latent_dim)
        )

    def decode(self, x):
        x = self.fc(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        h = self.net(x)
        h_flat = h.view(-1, self.t_out_ch * self.out_dim * self.out_dim)
        mu, var = self.mean_layer(h_flat), self.var_layer(h_flat)
        var = F.softplus(var) + 1e-8
        # mu, var = ut.gaussian_parameters(h, dim=1)
        return h, mu, var

class FCONV(nn.Module):
    def __init__(self, in_size, unflat_dim, t_in_ch, t_out_ch, t_kernel, t_padding, t_stride):
        super(FCONV, self).__init__()
        self.in_size = in_size
        self.unflat_dim = unflat_dim
        self.t_in_ch = t_in_ch
        self.t_out_ch = t_out_ch
        self.t_kernel = t_kernel
        self.t_stride = t_stride
        self.t_padding = t_padding

        self.fc_final = nn.Linear(in_size, t_in_ch * unflat_dim * unflat_dim)
        self.final = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose2d(t_in_ch, t_out_ch, kernel_size=t_kernel, padding=t_padding, stride=t_stride),  # (w-k+2p)/s+1
            #nn.Sigmoid()
            nn.Tanh()
        )

    def final_decode(self,x):
        x = self.fc_final(x)
        x = x.view(-1, self.t_in_ch, self.unflat_dim, self.unflat_dim)
        x_re = self.final(x)
        return x_re

class ORCED_MD(nn.Module):
    def __init__(self, alpha=100, beta=0, gamma=1, xi=1, tau=0.07,
                 in_ch=3,
                 out_ch64=64, out_ch128=128, out_ch256=256, out_ch512=512,
                 kernel1=1, kernel2=2, kernel3=3, padding0=0, padding1=1, padding2=2, stride1=1, stride2=2,
                 flat_dim32=32, flat_dim16=16, flat_dim8=8, flat_dim4=4, flat_dim2=2, flat_dim1=1,
                 latent_dim512=512, latent_dim256=256, latent_dim128=128, latent_dim64=64, latent_dim32=32, num_class =15):
        super().__init__()
        
        # Loss weights hyperparameters
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.xi=xi
        self.tau=tau

        # Define loss functions and miner for contrastive loss
        self.reconstruction_function = nn.MSELoss()
        self.reconstruction_function.size_average = False
        self.nllloss = nn.NLLLoss()
        
        ## Original ORCED doesnt specify any mining strategy so I don't use a miner
        # self.triplet_miner = miners.MultiSimilarityMiner()
        self.contrastive_loss = losses.SupConLoss(temperature=self.tau, distance=DotProductSimilarity())


        self.in_ch = in_ch
        self.out_ch64 = out_ch64
        self.out_ch128 = out_ch128
        self.out_ch256 = out_ch256
        self.out_ch512 = out_ch512
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.padding0 = padding0
        self.padding1 = padding1
        self.padding2 = padding2
        self.stride1 = stride1
        self.stride2 = stride2
        self.flat_dim32 = flat_dim32
        self.flat_dim16 = flat_dim16
        self.flat_dim8 = flat_dim8
        self.flat_dim4 = flat_dim4
        self.flat_dim2 = flat_dim2
        self.flat_dim1 = flat_dim1
        self.latent_dim512 = latent_dim512
        self.latent_dim256 = latent_dim256
        self.latent_dim128 = latent_dim128
        self.latent_dim64 = latent_dim64
        self.latent_dim32 = latent_dim32
        self.num_class = num_class

        # initialize required CONVs
        self.CONV1_1 = CONV(self.in_ch, self.out_ch64, self.kernel1, self.padding2, self.stride1, self.flat_dim32,
                            self.latent_dim512)
        self.CONV1_2 = CONV(self.out_ch64, self.out_ch64, self.kernel3, self.padding1, self.stride2, self.flat_dim16,
                            self.latent_dim512)

        self.CONV2_1 = CONV(self.out_ch64, self.out_ch128, self.kernel3, self.padding1, self.stride1, self.flat_dim16, self.latent_dim256)
        self.CONV2_2 = CONV(self.out_ch128, self.out_ch128, self.kernel3, self.padding1, self.stride2, self.flat_dim8, self.latent_dim256)

        self.CONV3_1 = CONV(self.out_ch128, self.out_ch256, self.kernel3, self.padding1, self.stride1, self.flat_dim8,
                            self.latent_dim128)
        self.CONV3_2 = CONV(self.out_ch256, self.out_ch256, self.kernel3, self.padding1, self.stride2, self.flat_dim4,
                            self.latent_dim128)

        self.CONV4_1 = CONV(self.out_ch256, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim4,
                            self.latent_dim64)
        self.CONV4_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim2,
                            self.latent_dim64)

        self.CONV5_1 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride1, self.flat_dim2,
                            self.latent_dim32)
        self.CONV5_2 = CONV(self.out_ch512, self.out_ch512, self.kernel3, self.padding1, self.stride2, self.flat_dim1,
                            self.latent_dim32)

        # initialize required TCONVs
        self.TCONV5_2 = TCONV(self.latent_dim32, self.flat_dim1, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim2, self.latent_dim32)
        self.TCONV5_1 = TCONV(self.latent_dim32, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim2, self.latent_dim64)

        self.TCONV4_2 = TCONV(self.latent_dim64, self.flat_dim2, self.out_ch512, self.out_ch512, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim4, self.latent_dim64)
        self.TCONV4_1 = TCONV(self.latent_dim64, self.flat_dim4, self.out_ch512, self.out_ch256, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim4, self.latent_dim128)

        self.TCONV3_2 = TCONV(self.latent_dim128, self.flat_dim4, self.out_ch256, self.out_ch256, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim8, self.latent_dim128)
        self.TCONV3_1 = TCONV(self.latent_dim128, self.flat_dim8, self.out_ch256, self.out_ch128, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim8, self.latent_dim256)

        self.TCONV2_2 = TCONV(self.latent_dim256, self.flat_dim8, self.out_ch128, self.out_ch128, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim16, self.latent_dim256)
        self.TCONV2_1 = TCONV(self.latent_dim256, self.flat_dim16, self.out_ch128, self.out_ch64, self.kernel1,
                              self.padding0, self.stride1, self.flat_dim16, self.latent_dim512)

        self.TCONV1_2 = TCONV(self.latent_dim512, self.flat_dim16, self.out_ch64, self.out_ch64, self.kernel2,
                              self.padding0, self.stride2, self.flat_dim32, self.latent_dim512)
        self.TCONV1_1 = FCONV(self.latent_dim512, self.flat_dim32, self.out_ch64, self.in_ch, self.kernel1,
                              self.padding2, self.stride1)

        self.classifier = nn.Linear(self.latent_dim32, self.num_class)

        # This is the layer that takes the one-hot encoded label and returns the centroid for that class
        self.one_hot = nn.Linear(self.num_class, self.latent_dim32)

    def lnet(self, x, y_de):
        # ---deterministic upward pass
        # upwards
        enc1_1, mu_up1_1, var_up1_1 = self.CONV1_1.encode(x)
        enc1_2, mu_up1_2, var_up1_2 = self.CONV1_2.encode(enc1_1)

        enc2_1, mu_up2_1, var_up2_1 = self.CONV2_1.encode(enc1_2)
        enc2_2, mu_up2_2, var_up2_2 = self.CONV2_2.encode(enc2_1)

        enc3_1, mu_up3_1, var_up3_1 = self.CONV3_1.encode(enc2_2)
        enc3_2, mu_up3_2, var_up3_2 = self.CONV3_2.encode(enc3_1)

        enc4_1, mu_up4_1, var_up4_1 = self.CONV4_1.encode(enc3_2)
        enc4_2, mu_up4_2, var_up4_2 = self.CONV4_2.encode(enc4_1)

        enc5_1, mu_up5_1, var_up5_1 = self.CONV5_1.encode(enc4_2)
        enc5_2, mu_latent, var_latent = self.CONV5_2.encode(enc5_1)

        ## "latent" is z, the feature from which the other branches start!
        latent = ut.sample_gaussian(mu_latent, var_latent)
        
        # This implementation uses nlll loss instead of cross entropy loss
        # predict = F.log_softmax(self.classifier(latent), dim=1)

        # yh is the class centroid computed by the network
        yh = self.one_hot(y_de)

        # partially downwards
        dec5_1, mu_dn5_1, var_dn5_1 = self.TCONV5_2.decode(latent)
        prec_up5_1 = var_up5_1 ** (-1)
        prec_dn5_1 = var_dn5_1 ** (-1)
        qmu5_1 = (mu_up5_1 * prec_up5_1 + mu_dn5_1 * prec_dn5_1) / (prec_up5_1 + prec_dn5_1)
        qvar5_1 = (prec_up5_1 + prec_dn5_1) ** (-1)
        de_latent5_1 = ut.sample_gaussian(qmu5_1, qvar5_1)

        dec4_2, mu_dn4_2, var_dn4_2 = self.TCONV5_1.decode(de_latent5_1)
        prec_up4_2 = var_up4_2 ** (-1)
        prec_dn4_2 = var_dn4_2 ** (-1)
        qmu4_2 = (mu_up4_2 * prec_up4_2 + mu_dn4_2 * prec_dn4_2) / (prec_up4_2 + prec_dn4_2)
        qvar4_2 = (prec_up4_2 + prec_dn4_2) ** (-1)
        de_latent4_2 = ut.sample_gaussian(qmu4_2, qvar4_2)

        dec4_1, mu_dn4_1, var_dn4_1 = self.TCONV4_2.decode(de_latent4_2)
        prec_up4_1 = var_up4_1 ** (-1)
        prec_dn4_1 = var_dn4_1 ** (-1)
        qmu4_1 = (mu_up4_1 * prec_up4_1 + mu_dn4_1 * prec_dn4_1) / (prec_up4_1 + prec_dn4_1)
        qvar4_1 = (prec_up4_1 + prec_dn4_1) ** (-1)
        de_latent4_1 = ut.sample_gaussian(qmu4_1, qvar4_1)

        dec3_2, mu_dn3_2, var_dn3_2 = self.TCONV4_1.decode(de_latent4_1)
        prec_up3_2 = var_up3_2 ** (-1)
        prec_dn3_2 = var_dn3_2 ** (-1)
        qmu3_2 = (mu_up3_2 * prec_up3_2 + mu_dn3_2 * prec_dn3_2) / (prec_up3_2 + prec_dn3_2)
        qvar3_2 = (prec_up3_2 + prec_dn3_2) ** (-1)
        de_latent3_2 = ut.sample_gaussian(qmu3_2, qvar3_2)

        dec3_1, mu_dn3_1, var_dn3_1 = self.TCONV3_2.decode(de_latent3_2)
        prec_up3_1 = var_up3_1 ** (-1)
        prec_dn3_1 = var_dn3_1 ** (-1)
        qmu3_1 = (mu_up3_1 * prec_up3_1 + mu_dn3_1 * prec_dn3_1) / (prec_up3_1 + prec_dn3_1)
        qvar3_1 = (prec_up3_1 + prec_dn3_1) ** (-1)
        de_latent3_1 = ut.sample_gaussian(qmu3_1, qvar3_1)

        dec2_2, mu_dn2_2, var_dn2_2 = self.TCONV3_1.decode(de_latent3_1)
        prec_up2_2 = var_up2_2 ** (-1)
        prec_dn2_2 = var_dn2_2 ** (-1)
        qmu2_2 = (mu_up2_2 * prec_up2_2 + mu_dn2_2 * prec_dn2_2) / (prec_up2_2 + prec_dn2_2)
        qvar2_2 = (prec_up2_2 + prec_dn2_2) ** (-1)
        de_latent2_2 = ut.sample_gaussian(qmu2_2, qvar2_2)

        dec2_1, mu_dn2_1, var_dn2_1 = self.TCONV2_2.decode(de_latent2_2)
        prec_up2_1 = var_up2_1 ** (-1)
        prec_dn2_1 = var_dn2_1 ** (-1)
        qmu2_1 = (mu_up2_1 * prec_up2_1 + mu_dn2_1 * prec_dn2_1) / (prec_up2_1 + prec_dn2_1)
        qvar2_1 = (prec_up2_1 + prec_dn2_1) ** (-1)
        de_latent2_1 = ut.sample_gaussian(qmu2_1, qvar2_1)

        dec1_2, mu_dn1_2, var_dn1_2 = self.TCONV2_1.decode(de_latent2_1)
        prec_up1_2 = var_up1_2 ** (-1)
        prec_dn1_2 = var_dn1_2 ** (-1)
        qmu1_2 = (mu_up1_2 * prec_up1_2 + mu_dn1_2 * prec_dn1_2) / (prec_up1_2 + prec_dn1_2)
        qvar1_2 = (prec_up1_2 + prec_dn1_2) ** (-1)
        de_latent1_2 = ut.sample_gaussian(qmu1_2, qvar1_2)

        dec1_1, mu_dn1_1, var_dn1_1 = self.TCONV1_2.decode(de_latent1_2)
        prec_up1_1 = var_up1_1 ** (-1)
        prec_dn1_1 = var_dn1_1 ** (-1)
        qmu1_1 = (mu_up1_1 * prec_up1_1 + mu_dn1_1 * prec_dn1_1) / (prec_up1_1 + prec_dn1_1)
        qvar1_1 = (prec_up1_1 + prec_dn1_1) ** (-1)
        de_latent1_1 = ut.sample_gaussian(qmu1_1, qvar1_1)

        x_re = self.TCONV1_1.final_decode(de_latent1_1)

        return latent, mu_latent, var_latent, \
               qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
               qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
               yh, \
               x_re, \
               mu_dn5_1, var_dn5_1, mu_dn4_2, var_dn4_2, mu_dn4_1, var_dn4_1, mu_dn3_2, var_dn3_2, mu_dn3_1, var_dn3_1, \
               mu_dn2_2, var_dn2_2, mu_dn2_1, var_dn2_1, mu_dn1_2, var_dn1_2, mu_dn1_1, var_dn1_1

    def loss(self, x, y, y_de):

        latent, mu_latent, var_latent, \
        qmu5_1, qvar5_1, qmu4_2, qvar4_2, qmu4_1, qvar4_1, qmu3_2, qvar3_2, qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        yh, \
        x_re, \
        pmu5_1, pvar5_1,pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de)


        # pm, pv = torch.zeros(mu_latent.shape).cuda(), torch.ones(var_latent.shape).cuda()
        pm, pv = torch.zeros(mu_latent.shape).to(constants.DEVICE), torch.ones(var_latent.shape).to(constants.DEVICE)
        # print("mu1", mu1)
        kl_latent = ut.kl_normal(mu_latent, var_latent, pm, pv, yh)
        kl5_1 = ut.kl_normal(qmu5_1, qvar5_1, pmu5_1, pvar5_1, 0)
        kl4_2 = ut.kl_normal(qmu4_2, qvar4_2, pmu4_2, pvar4_2, 0)
        kl4_1 = ut.kl_normal(qmu4_1, qvar4_1, pmu4_1, pvar4_1, 0)
        kl3_2 = ut.kl_normal(qmu3_2, qvar3_2, pmu3_2, pvar3_2, 0)
        kl3_1 = ut.kl_normal(qmu3_1, qvar3_1, pmu3_1, pvar3_1, 0)
        kl2_2 = ut.kl_normal(qmu2_2, qvar2_2, pmu2_2, pvar2_2, 0)
        kl2_1 = ut.kl_normal(qmu2_1, qvar2_1, pmu2_1, pvar2_1, 0)
        kl1_2 = ut.kl_normal(qmu1_2, qvar1_2, pmu1_2, pvar1_2, 0)
        kl1_1 = ut.kl_normal(qmu1_1, qvar1_1, pmu1_1, pvar1_1, 0)

        ## Unified L_KL term, weighted by beta
        L_kl = torch.mean(kl_latent + kl5_1 + kl4_2 + kl4_1 + kl3_2 + kl3_1 + kl2_2 + kl2_1 + kl1_2 + kl1_1)

        ## L_CE term, weighted by alpha
        # NB: This implementation uses nlll loss instead of cross entropy loss, so we explicitly use log_softmax
        predict = F.log_softmax(self.classifier(latent), dim=1)
        L_ce = self.nllloss(predict, y)

        ## L_REC term, weighted by xi
        L_rec = self.reconstruction_function(x_re, x)

        ## L_SC Contrastive loss term weighted by gamma
        normalized_z = F.normalize(latent, p=2, dim=1)
        L_sc = self.contrastive_loss(normalized_z, y)

        total_loss = self.alpha * L_ce + self.beta * L_kl + self.xi * L_rec + self.gamma * L_sc
        # total_loss = L_rec
        
        return total_loss, mu_latent, predict, x_re, L_ce, L_kl, L_rec, L_sc

    def test(self, x, y_de):
        ## I probably have to change this
        _, mu_latent, _, \
        _, _, _, _, _, _, _, _, _, _,\
        _, _, _, _, _, _, _, _, \
        _ ,\
        x_re, \
        pmu5_1, pvar5_1, pmu4_2, pvar4_2, pmu4_1, pvar4_1, pmu3_2, pvar3_2, pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.lnet(x, y_de)

        # TODO: understand what this term is for.
        predict_test = F.log_softmax(self.classifier(mu_latent), dim=1)
        return mu_latent, predict_test, x_re