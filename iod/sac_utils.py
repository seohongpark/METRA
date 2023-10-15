import torch
from torch.nn import functional as F


def _clip_actions(algo, actions):
    epsilon = 1e-6
    lower = torch.from_numpy(algo._env_spec.action_space.low).to(algo.device) + epsilon
    upper = torch.from_numpy(algo._env_spec.action_space.high).to(algo.device) - epsilon

    clip_up = (actions > upper).float()
    clip_down = (actions < lower).float()
    with torch.no_grad():
        clip = ((upper - actions) * clip_up + (lower - actions) * clip_down)

    return actions + clip


def update_loss_qf(
        algo, tensors, v,
        obs,
        actions,
        next_obs,
        dones,
        rewards,
        policy,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    q1_pred = algo.qf1(obs, actions).flatten()
    q2_pred = algo.qf2(obs, actions).flatten()

    next_action_dists, *_ = policy(next_obs)
    if hasattr(next_action_dists, 'rsample_with_pre_tanh_value'):
        new_next_actions_pre_tanh, new_next_actions = next_action_dists.rsample_with_pre_tanh_value()
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions, pre_tanh_value=new_next_actions_pre_tanh)
    else:
        new_next_actions = next_action_dists.rsample()
        new_next_actions = _clip_actions(algo, new_next_actions)
        new_next_action_log_probs = next_action_dists.log_prob(new_next_actions)

    target_q_values = torch.min(
        algo.target_qf1(next_obs, new_next_actions).flatten(),
        algo.target_qf2(next_obs, new_next_actions).flatten(),
    )
    target_q_values = target_q_values - alpha * new_next_action_log_probs
    target_q_values = target_q_values * algo.discount

    with torch.no_grad():
        q_target = rewards + target_q_values * (1. - dones)

    # critic loss weight: 0.5
    loss_qf1 = F.mse_loss(q1_pred, q_target) * 0.5
    loss_qf2 = F.mse_loss(q2_pred, q_target) * 0.5

    tensors.update({
        'QTargetsMean': q_target.mean(),
        'QTdErrsMean': ((q_target - q1_pred).mean() + (q_target - q2_pred).mean()) / 2,
        'LossQf1': loss_qf1,
        'LossQf2': loss_qf2,
    })


def update_loss_sacp(
        algo, tensors, v,
        obs,
        policy,
):
    with torch.no_grad():
        alpha = algo.log_alpha.param.exp()

    action_dists, *_ = policy(obs)
    if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
        new_actions_pre_tanh, new_actions = action_dists.rsample_with_pre_tanh_value()
        new_action_log_probs = action_dists.log_prob(new_actions, pre_tanh_value=new_actions_pre_tanh)
    else:
        new_actions = action_dists.rsample()
        new_actions = _clip_actions(algo, new_actions)
        new_action_log_probs = action_dists.log_prob(new_actions)

    min_q_values = torch.min(
        algo.qf1(obs, new_actions).flatten(),
        algo.qf2(obs, new_actions).flatten(),
    )

    loss_sacp = (alpha * new_action_log_probs - min_q_values).mean()

    tensors.update({
        'SacpNewActionLogProbMean': new_action_log_probs.mean(),
        'LossSacp': loss_sacp,
    })

    v.update({
        'new_action_log_probs': new_action_log_probs,
    })


def update_loss_alpha(
        algo, tensors, v,
):
    loss_alpha = (-algo.log_alpha.param * (
            v['new_action_log_probs'].detach() + algo._target_entropy
    )).mean()

    tensors.update({
        'Alpha': algo.log_alpha.param.exp(),
        'LossAlpha': loss_alpha,
    })


def update_targets(algo):
    """Update parameters in the target q-functions."""
    target_qfs = [algo.target_qf1, algo.target_qf2]
    qfs = [algo.qf1, algo.qf2]
    for target_qf, qf in zip(target_qfs, qfs):
        for t_param, param in zip(target_qf.parameters(), qf.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - algo.tau) +
                               param.data * algo.tau)
