import torch
import torch.nn.functional as F


def _off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix[~torch.eye(n, dtype=torch.bool, device=matrix.device)]


def _correlation(left, right):
    left, right = left.reshape(-1), right.reshape(-1)
    left, right = left - left.mean(), right - right.mean()
    return (left * right).mean() / (left.std(unbiased=False) * right.std(unbiased=False)).clamp_min(1e-8)


def linear_hsic(left, right):
    if left.shape[0] < 2:
        return left.new_zeros(())
    left = left - left.mean(0, keepdim=True)
    right = right - right.mean(0, keepdim=True)
    cross = left.t() @ right / max(left.shape[0] - 1, 1)
    return cross.square().mean()


def similarity_correlation(representation, environment_q):
    if representation.shape[0] < 3:
        return representation.new_zeros(())
    rep = F.normalize(representation, dim=-1)
    return _correlation(_off_diagonal(rep @ rep.t()), _off_diagonal(environment_q @ environment_q.t()))


def contrastive_constraint(z_inv, z_var, q, temperature=.2, invariant_weight=1.0):
    n = z_inv.shape[0]
    if n < 2:
        zero = z_inv.new_zeros(())
        return zero, {"domain_var_con": zero, "domain_inv_con": zero}
    var = F.normalize(z_var, dim=-1)
    similarity = var @ var.t() / temperature
    mask = ~torch.eye(n, dtype=torch.bool, device=z_inv.device)
    exp_similarity = torch.exp(similarity - similarity.max(1, keepdim=True).values) * mask
    env_similarity = (q @ q.t()) * mask
    numerator = (env_similarity * exp_similarity).sum(1)
    denominator = exp_similarity.sum(1).clamp_min(1e-8)
    variant_loss = -torch.log((numerator / denominator).clamp_min(1e-8)).mean()
    invariant_corr = similarity_correlation(z_inv, q)
    invariant_loss = invariant_corr.square()
    return variant_loss + invariant_weight * invariant_loss, {
        "domain_var_con": variant_loss, "domain_inv_con": invariant_loss,
    }


def mutual_info_constraint(z_inv, z_var, q, q_logits, weights=(1.0, 1.0, 1.0)):
    inv_hsic = linear_hsic(z_inv, q)
    var_reconstruction = F.kl_div(F.log_softmax(q_logits, -1), q, reduction="batchmean")
    separation = linear_hsic(z_inv, z_var)
    total = weights[0] * inv_hsic + weights[1] * var_reconstruction + weights[2] * separation
    return total, {
        "domain_inv_hsic": inv_hsic,
        "domain_var_kl": var_reconstruction,
        "domain_inv_var_hsic": separation,
    }


def domain_constraint_loss(mode, z_inv, z_var, q, q_logits, temperature=.2,
                           invariant_weight=1.0, mi_weights=(1.0, 1.0, 1.0)):
    if mode == "none":
        return z_inv.new_zeros(()), {}
    if mode == "contrastive":
        return contrastive_constraint(z_inv, z_var, q, temperature, invariant_weight)
    if mode == "mutual_info":
        return mutual_info_constraint(z_inv, z_var, q, q_logits, mi_weights)
    raise ValueError("domain_constraint must be none, contrastive, or mutual_info")


@torch.no_grad()
def representation_diagnostics(z_inv, z_var, q):
    env_similarity = _off_diagonal(q @ q.t()) if q.shape[0] > 1 else q.new_zeros(1)
    return {
        "env_similarity_mean": env_similarity.mean(),
        "env_similarity_std": env_similarity.std(unbiased=False),
        "corr_inv_env": similarity_correlation(z_inv, q),
        "corr_var_env": similarity_correlation(z_var, q),
        "hsic_inv_env": linear_hsic(z_inv, q),
        "hsic_var_env": linear_hsic(z_var, q),
        "hsic_inv_var": linear_hsic(z_inv, z_var),
    }
