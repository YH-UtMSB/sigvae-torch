import torch
import torch.nn.modules.loss
import torch.nn.functional as F
import numpy as np


def loss_function(preds, labels, mu, logvar, emb, eps, n_nodes, norm, pos_weight):
    """
    Computing the negative ELBO for SIGVAE:
        loss = - \E_{h(z)} \log \frac{p(x|z)p(z)}{h(z)}.

    Parameters
    ----------
    preds : torch.Tensor of shape [J, N, N],
        Reconsurcted graph probability with J samples drawn from h(z).
    labels : torch.Tensor of shape [N, N],
        the ground truth connectivity between nodes in the adjacency matrix.
    mu : torch.Tensor of shape [K+J, N, zdim],
        the gaussian mean of q(z|psi).
    logvar : torch.Tensor of shape [K+J, N, zdim],
        the gaussian logvar of q(z|psi).
    emb: torch.Tensor of shape [J, N, zdim],
        the node embeddings that generate preds.
    eps: torch.Tensor of shape [J, N, zdim],
        the random noise drawn from N(0,1) to construct emb.
    n_nodes : int,
        the number of nodes in the dataset.
    norm : float,
        normalizing constant for re-balanced dataset.
    pos_weight : torch.Tensor of shape [1],
        stands for "positive weight", used for re-balancing +/- trainning samples.

    Returns
        reconstruction loss and kl regularizer.
    -------
    TYPE
        DESCRIPTION.

    """
    def get_rec(pred):
        # pred = torch.sigmoid(pred)
        log_lik = norm * (pos_weight * labels * torch.log(pred) + (1 - labels) * torch.log(1 - pred))  # N * N
        rec = -log_lik.mean()
        return rec

    
    # There are some problem with bce function when running bp models. Causes are under investigation.
    # def get_rec(pred):
    #     return norm * F.binary_cross_entropy_with_logits(pred, labels, pos_weight=pos_weight)


    # The objective is made up of 3 components,
    # loss = rec_cost + beta * (log_posterior - log_prior), where
    # rec_cost = -mean(log p(A|Z[1]), log p(A|Z[2]), ... log p(A|Z[J])),
    # log_prior = mean(log p(Z[1]), log p(Z[2]), ..., log p(Z[J])),
    # log_posterior = mean(log post[1], log post[2], ..., log post[J]), where
    # log post[j] = 1/(K+1) {q(Z[j]\psi[j]) + [q(Z[j]|psi^[1]) + ... + q(Z[j]|psi^[k])]}.
    # In practice, the loss is computed as
    # loss = rec_lost + log_posterior_ker - log_prior_ker.


    SMALL = 1e-6
    std = torch.exp(0.5 * logvar)
    J, N, zdim = emb.shape
    K = mu.shape[0] - J

    mu_mix, mu_emb = mu[:K, :], mu[K:, :]
    std_mix, std_emb = std[:K, :], std[K:, :]

    preds = torch.clamp(preds, min=SMALL, max=1-SMALL)

    # compute rec_cost
    rec_costs = torch.stack(
            [get_rec(pred) for pred in torch.unbind(preds, dim=0)],
            dim=0)
    # average over J * N * N items
    rec_cost = rec_costs.mean()



    # compute log_prior_ker, the constant 1/sqrt(2*pi) is cancelled out.
    # average over J items
    log_prior_ker = torch.sum(- 0.5 * emb.pow(2), dim=[1,2]).mean()


    # compute log_posterior
    # Z.shape = [J, 1, N, zdim]
    Z = emb.view(J, 1, N, zdim)

    # mu_mix.shape = std_mix.shape = [1, K, N, zdim]
    mu_mix = mu_mix.view(1, K, N, zdim)
    std_mix = std_mix.view(1, K, N, zdim)
    
    # compute -log std[k] - (Z[j] - mu[k])^2 / 2*std[k]^2 for all (j,k)
    # the shape of result tensor log_post_ker_JK is [J,K]
    log_post_ker_JK = - torch.sum(
        0.5 * ((Z - mu_mix) / (std_mix + SMALL)).pow(2), dim=[-2,-1]
    )

    log_post_ker_JK += - torch.sum(
        (std_mix + SMALL).log(), dim=[-2,-1]
    )

    # compute -log std[j] - (Z[j] - mu[j])^2 / 2*std[j]^2 for j = 1,2,...,J
    # the shape of result tensor log_post_ker_J is [J, 1]
    log_post_ker_J = - torch.sum(
        0.5 * eps.pow(2), dim=[-2,-1]
    )
    log_post_ker_J += - torch.sum(
        (std_emb + SMALL).log(), dim = [-2,-1]
    )
    log_post_ker_J = log_post_ker_J.view(-1,1)


    # bind up log_post_ker_JK and log_post_ker_J into log_post_ker, the shape of result tensor is [J, K+1].
    log_post_ker = torch.cat([log_post_ker_JK, log_post_ker_J], dim=-1)

    # apply "log-mean-exp" to the above tensor
    log_post_ker -= np.log(K + 1.) / J
    # average over J items.
    log_posterior_ker = torch.logsumexp(log_post_ker, dim=-1).mean()

    


    return rec_cost, log_prior_ker, log_posterior_ker 


    
