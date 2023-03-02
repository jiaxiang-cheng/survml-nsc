from torch.autograd import grad
import numpy as np
import torch.nn as nn
import torch


class PositiveLinear(nn.Module):
    """
    Neural network with positive weights
    """

    def __init__(self, in_features, out_features, bias=False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, self.log_weight ** 2, self.bias)
        else:
            return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, activation, dropout=0):
    """
    Create a simple multi layer neural network of positive layers
    With final SoftPlus
    """

    modules = []
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'Tanh':
        act = nn.Tanh()
    else:
        raise ValueError("Unknown {} activation".format(activation))

    prevdim = inputdim
    for hidden in layers:
        modules.append(PositiveLinear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(act)
        prevdim = hidden

    # Need all values positive
    modules[-1] = nn.Softplus()

    return nn.Sequential(*modules)


def create_representation(inputdim, layers, activation, dropout=0.5):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p=dropout))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)


class NeuralSurvivalClusterTorch(nn.Module):

    def __init__(self,
                 inputdim,
                 layers=None,
                 act='ReLU6',
                 layers_surv=None,
                 representation=50,
                 act_surv='Tanh',
                 weight_balance=1.,
                 risks=1,
                 k=3,
                 dropout=0.,
                 optimizer="Adam"):

        super(NeuralSurvivalClusterTorch, self).__init__()
        if layers_surv is None:
            layers_surv = [100]
        if layers is None:
            layers = [100, 100, 100]
        self.input_dim = inputdim
        self.weight_balance = weight_balance  # Used for balancing the loss between censored and uncensored
        self.risks = risks  # Competing risks
        self.k = k  # Number mixture
        self.representation = representation  # Latent input for clusters (centroid representation)
        self.dropout = dropout
        self.optimizer = optimizer

        self.profile = create_representation(inputdim, layers + [self.k], act, self.dropout)
        # this one is create the MLP to infer the weightage of each cluster's distribution

        self.latent = nn.ParameterList(
            [
                nn.Parameter(torch.randn((1, self.representation)))
                for _ in range(self.k)
            ]
        )

        self.outcome = nn.ModuleList(
            [
                create_representation_positive(1 + self.representation, layers_surv + [risks], act_surv, self.dropout)
                for _ in range(self.k)
            ]
        )
        # create_representation_positive(inputdim, layers, activation, dropout=0)
        # inputs are number of latent cluster representation + the time/horizon
        # outputs will be the cumulative hazard, and the number of outputs corresponding to number of risks

        self.competing = create_representation(inputdim, layers + [risks], act, self.dropout)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, horizon, gradient=False):
        if self.risks == 1:
            betas = 1
        else:
            betas = self.soft(self.competing(x))

        if self.k == 1:
            alphas = torch.ones((len(x), self.k), requires_grad=True).float().to(x.device)
        else:
            alphas = self.profile(x)  # the weightage of each cluster distribution

        # Compute intensity and cumulative function
        cumulative, intensity = [], []
        for latent, outcome_competing in zip(self.latent, self.outcome):
            # latent is the latent cluster representation of the k-th cluster
            # outcome_competing is the neural network for the k-th cluster

            # for each feature we need a set of representations
            latent = latent.repeat(len(x), 1)

            tau_outcome = horizon.clone().detach().requires_grad_(gradient)  # Copy with independent gradient

            # inputs into the neural network is the latent cluster representations + the time/horizon t
            outcome = outcome_competing(torch.cat((latent, tau_outcome.unsqueeze(1)), 1))
            # so the outcome will be the cumulative hazard function

            # and the outcome has to subtract the value when t is 0
            # "Finally, the additional constraint of being null at time t = 0 for the cumulative hazard must be
            # enforced. Therefore, the neural network value at the origin time is subtracted from each component.
            # This ensures that each component returns the well defined Λk."
            outcome = outcome - outcome_competing(torch.cat((latent, torch.zeros_like(tau_outcome.unsqueeze(1))), 1))

            # betas are the weightages for different risks,
            # and generated with another neural network followed by a softmax layer
            outcome = betas * outcome
            # so this outcome will be the final cumulative hazard function

            cumulative.append(outcome.unsqueeze(-1))

            if gradient:
                # this is to get the intensity function from the cumulative hazard function
                # the integral of intensity is the cumulative hazard function
                int = []
                for r in range(self.risks):
                    int.append(grad(outcome[:, r].sum(), tau_outcome, create_graph=True)[0].unsqueeze(1))
                int = torch.cat(int, -1).unsqueeze(-1)
                intensity.append(int)

        cumulative = torch.cat(cumulative, -1)
        intensity = torch.cat(intensity, -1) if gradient else None

        return cumulative, intensity, alphas

    def predict(self, x, horizon):
        # the inputs are the features and time/horizon
        cumulative, _, alphas = self.forward(x, horizon)
        alphas = nn.Softmax(dim=1)(alphas).unsqueeze(1).repeat(1, self.risks, 1)  # Repeat alpha for each risk

        # An individual survival function is then a weighted sum of these neural distributions
        return torch.sum(alphas * torch.exp(-cumulative), dim=2), alphas, torch.exp(-cumulative)
