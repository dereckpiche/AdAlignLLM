"""
File: mllm/training/credit_methods.py
Summary: Holds credit-assignment routines for reinforcement learning updates.
"""

import torch


def whiten_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """
    Normalize a vector of advantages to zero mean / unit variance (global).

    Useful for variance reduction before computing gradients.
    """
    whitened_advantages = (advantages - torch.mean(advantages)) / (
        torch.std(advantages) + 1e-9
    )
    return whitened_advantages


def whiten_advantages_time_step_wise(
    advantages: torch.Tensor,  # (B, T)
) -> torch.Tensor:
    """
    Whiten advantages independently per timestep (column-wise mean/std).

    Helps when rollout lengths differ or certain positions have very different scales.
    """
    assert advantages.dim() == 2, "Wrong dimensions."
    whitened_advantages_time_step_wise = (
        advantages - advantages.mean(dim=0, keepdim=True)
    ) / (advantages.std(dim=0, keepdim=True) + 1e-9)
    return whitened_advantages_time_step_wise


def get_discounted_state_visitation_credits(
    credits: torch.Tensor, discount_factor: float  # (B, T)
) -> torch.Tensor:
    """
    Apply geometric discounting to credits so earlier visits count less.

    Equivalent to per-timestep multiplication by ``gamma^t``.
    """
    return credits * (
        discount_factor ** torch.arange(credits.shape[1], device=credits.device)
    )


def get_discounted_returns(
    rewards: torch.Tensor,  # (B, T)
    discount_factor: float,
) -> torch.Tensor:
    """
    Computes Monte Carlo discounted returns for a sequence of rewards.

    Args:
        rewards (torch.Tensor): Array of rewards for each timestep.

    Returns:
        torch.Tensor: Array of discounted returns.
    """
    assert rewards.dim() == 2, "Wrong dimensions."
    B, T = rewards.shape
    discounted_returns = torch.zeros_like(rewards)
    accumulator = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        accumulator = rewards[:, t] + discount_factor * accumulator
        discounted_returns[:, t] = accumulator
    return discounted_returns


def get_rloo_credits(credits: torch.Tensor):  # (B, S)
    """Compute leave-one-out baselines for a batch of credits."""
    assert credits.dim() == 2, "Wrong dimensions."
    rloo_baselines = torch.zeros_like(credits)
    n = credits.shape[0]
    if n == 1:
        return credits, rloo_baselines
    rloo_baselines = (torch.sum(credits, dim=0, keepdim=True) - credits) / (n - 1)
    rloo_credits = credits - rloo_baselines
    return rloo_credits, rloo_baselines


def get_generalized_advantage_estimates(
    rewards: torch.Tensor,  # (B, T)
    value_estimates: torch.Tensor,  # (B, T+1)
    discount_factor: float,
    lambda_coef: float,
) -> torch.Tensor:
    """
    Compute Generalized Advantage Estimates (GAE).

    See https://arxiv.org/pdf/1506.02438 for derivation.
    """
    assert rewards.dim() == value_estimates.dim() == 2, "Wrong dimensions."

    assert (
        rewards.shape[0] == value_estimates.shape[0]
    ), f"Got shapes {rewards.shape} and {value_estimates.shape} of rewards and value estimates."
    assert (
        rewards.shape[1] == value_estimates.shape[1] - 1
    ), f"Got shapes {rewards.shape} and {value_estimates.shape} of rewards and value estimates."

    T = rewards.shape[1]
    tds = rewards + discount_factor * value_estimates[:, 1:] - value_estimates[:, :-1]
    gaes = torch.zeros_like(tds)
    acc = 0.0
    for t in reversed(range(T)):
        acc = tds[:, t] + lambda_coef * discount_factor * acc
        gaes[:, t] = acc
    return gaes


def get_advantage_alignment_weights(
    advantages: torch.Tensor,  # (B, T)
    exclude_k_equals_t: bool,
    gamma: float,
    discount_t: bool,
) -> torch.Tensor:
    """
    The advantage alignment credit is calculated as

    \[
        A^*(s_t, a_t, b_t) = A^1(s_t, a_t, b_t) + \beta \cdot
        \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \right)
        A^2(s_t, a_t, b_t)
    \]

    Here, the weights are defined as \( \beta \cdot
        \left( \sum_{k < t} \gamma^{t-k} A^1(s_k, a_k, b_k) \)
    """
    T = advantages.shape[1]
    discounted_advantages = advantages * (
        gamma * torch.ones((1, T), device=advantages.device)
    ) ** (-torch.arange(0, T, 1, device=advantages.device))
    if exclude_k_equals_t:
        sub = torch.eye(T, device=advantages.device)
    else:
        sub = torch.zeros((T, T), device=advantages.device)
    # Identity is for \( k < t \), remove for \( k \leq t \)
    ad_align_weights = discounted_advantages @ (
        torch.triu(torch.ones((T, T), device=advantages.device)) - sub
    )
    t_discounts = (gamma * torch.ones((1, T), device=advantages.device)) ** (
        torch.arange(0, T, 1, device=advantages.device)
    )
    ad_align_weights = t_discounts * ad_align_weights
    if discount_t:
        time_discounted_advantages = advantages * (
            gamma * torch.ones((1, T), device=advantages.device)
        ) ** (torch.arange(0, T, 1, device=advantages.device))
        ad_align_weights = ad_align_weights - advantages + time_discounted_advantages
    return ad_align_weights


def get_advantage_alignment_credits(
    a1: torch.Tensor,  # (B, S)
    a1_alternative: torch.Tensor,  # (B, S, A)
    a2: torch.Tensor,  # (B, S)
    exclude_k_equals_t: bool,
    beta: float,
    gamma: float = 1.0,
    use_old_ad_align: bool = False,
    use_sign: bool = False,
    clipping: float | None = None,
    use_time_regularization: bool = False,
    force_coop_first_step: bool = False,
    use_variance_regularization: bool = False,
    rloo_branch: bool = False,
    reuse_baseline: bool = False,
    mean_normalize_ad_align: bool = False,
    whiten_adalign_advantages: bool = False,
    whiten_adalign_advantages_time_step_wise: bool = False,
    discount_t: bool = False,
) -> torch.Tensor:
    """
    Calculate the advantage alignment credits with vectorization, as described in https://arxiv.org/abs/2406.14662.

    Recall that the advantage opponent shaping term of the AdAlign policy gradient is:
    \[
        \beta \mathbb{E}_{\substack{
        \tau \sim \text{Pr}_{\mu}^{\pi^1, \pi^2} \\
        a_t' \sim \pi^1(\cdot \mid s_t)
        }}
        \left[\sum_{t=0}^\infty  \gamma^{t}\left( \sum_{k\leq t} A^1(s_k,a^{\prime}_k,b_k) \right) A^{2}(s_t,a_t, b_t)\nabla_{\theta^1}\text{log } \pi^1(a_t|s_t) \right]
    \]

    This method computes the following:
    \[
        Credit(s_t, a_t, b_t) = \gamma^t \left[ A^1(s_t, a_t, b_t) + \beta \left( \sum_{k\leq t} A^1(s_k,a^{\prime}_k,b_k) \right) A^{2}(s_t,a_t, b_t) \right]
    \]

    Args:
        a1: Advantages of the main trajectories for the current agent.
        a1_alternative: Advantages of the alternative trajectories for the current agent.
        a2: Advantages of the main trajectories for the other agent.
        discount_factor: Discount factor for the advantage alignment.
        beta: Beta parameter for the advantage alignment.
        gamma: Gamma parameter for the advantage alignment.
        use_sign_in_ad_align: Whether to use sign in the advantage alignment.

    Returns:
        torch.Tensor: The advantage alignment credits.
    """

    assert a1.dim() == a2.dim() == 2, "Advantages must be of shape (B, S)"
    if a1_alternative is not None:
        assert (
            a1_alternative.dim() == 3
        ), "Alternative advantages must be of shape (B, S, A)"
        B, T, A = a1_alternative.shape
    else:
        B, T = a1.shape
    assert a1.shape == a2.shape, "Not the same shape"

    sub_tensors = {}

    if use_old_ad_align:
        ad_align_weights = get_advantage_alignment_weights(
            advantages=a1,
            exclude_k_equals_t=exclude_k_equals_t,
            gamma=gamma,
            discount_t=discount_t,
        )
        sub_tensors["ad_align_weights_prev"] = ad_align_weights
        if exclude_k_equals_t:
            ad_align_weights = gamma * ad_align_weights
    else:
        assert a1_alternative is not None, "Alternative advantages must be provided"
        if rloo_branch:
            a1_alternative = torch.cat([a1.unsqueeze(2), a1_alternative], dim=2)
            a1_alternative = a1_alternative.mean(dim=2)
            a1, baseline = get_rloo_credits(a1)
            if reuse_baseline:
                a1_alternative = a1_alternative - baseline
            else:
                a1_alternative, _ = get_rloo_credits(a1_alternative)
        assert a1.shape == a1_alternative.shape, "Not the same shape"
        ad_align_weights = get_advantage_alignment_weights(
            advantages=a1_alternative,
            exclude_k_equals_t=exclude_k_equals_t,
            gamma=gamma,
        )
        sub_tensors["ad_align_weights"] = ad_align_weights

    # Use sign
    if use_sign:
        assert beta == 1.0, "beta should be 1.0 when using sign"
        positive_signs = ad_align_weights > 0
        negative_signs = ad_align_weights < 0
        ad_align_weights[positive_signs] = 1
        ad_align_weights[negative_signs] = -1
        sub_tensors["ad_align_weights_sign"] = ad_align_weights
        # (rest are 0)

    ###################
    # Process weights
    ###################

    # Use clipping
    if clipping not in [0.0, None]:
        upper_mask = ad_align_weights > 1
        lower_mask = ad_align_weights < -1

        ad_align_weights = torch.clip(
            ad_align_weights,
            -clipping,
            clipping,
        )
        clipping_ratio = (
            torch.sum(upper_mask) + torch.sum(lower_mask)
        ) / upper_mask.size
        sub_tensors["clipped_ad_align_weights"] = ad_align_weights

    # 1/1+t Regularization
    if use_time_regularization:
        t_values = torch.arange(1, T + 1).to(ad_align_weights.device)
        ad_align_weights = ad_align_weights / t_values
        sub_tensors["time_regularized_ad_align_weights"] = ad_align_weights

    # Use coop on t=0
    if force_coop_first_step:
        ad_align_weights[:, 0] = 1
        sub_tensors["coop_first_step_ad_align_weights"] = ad_align_weights

    ####################################
    # Compose elements together
    ####################################

    opp_shaping_terms = beta * ad_align_weights * a2
    sub_tensors["ad_align_opp_shaping_terms"] = opp_shaping_terms

    credits = a1 + opp_shaping_terms
    if mean_normalize_ad_align:
        credits = credits - credits.mean(dim=0)
        sub_tensors["mean_normalized_ad_align_credits"] = credits
    if whiten_adalign_advantages:
        credits = (credits - credits.mean()) / (credits.std() + 1e-9)
        sub_tensors["whitened_ad_align_credits"] = credits
    if whiten_adalign_advantages_time_step_wise:
        credits = (credits - credits.mean(dim=0, keepdim=True)) / (
            credits.std(dim=0, keepdim=True) + 1e-9
        )
        sub_tensors["whitened_ad_align_credits_time_step_wise"] = credits
    sub_tensors["final_ad_align_credits"] = credits

    return credits, sub_tensors
