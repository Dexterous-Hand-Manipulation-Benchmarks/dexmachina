"""Conformance check for vectorised envs with the Gymnasium vector API and torch tensors."""

from typing import Any

import torch

from dexmachina.envs.vec_env import AUTORESET_SAME_STEP

DEFAULT_CONFORMANCE_SEED = 0


class VecEnvContractError(Exception):
    """A vector env violated the interface contract that `check_vec_env` tests."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VecEnvContractError(message)


def _check_tensor(name: str, tensor: Any, shape: tuple[int, ...], dtype: torch.dtype, device: str) -> None:
    _require(isinstance(tensor, torch.Tensor), f"{name} is a {type(tensor).__name__}, expected a torch.Tensor")
    _require(tuple(tensor.shape) == shape, f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    _require(tensor.dtype == dtype, f"{name} has dtype {tensor.dtype}, expected {dtype}")
    _require(
        tensor.device.type == torch.device(device).type,
        f"{name} is on {tensor.device}, expected the env's device {device}",
    )


def check_vec_env(env: Any, steps: int) -> None:
    """Steps `env` with random actions and raises `VecEnvContractError` on the first contract violation.

    Checks shapes, dtypes and device of everything returned, `info["final_obs"]` and its mask (rows of unflagged envs
    equal the returned observation), that flags appear only where `_final_obs` is set, the autoreset mode, the step
    counter, and that returned tensors are copies: stepping again must not change tensors returned earlier.
    """
    n = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    action_space = env.single_action_space
    _require(
        env.metadata.get("autoreset_mode") == AUTORESET_SAME_STEP,
        f"metadata['autoreset_mode'] is {env.metadata.get('autoreset_mode')!r}, expected {AUTORESET_SAME_STEP!r}",
    )
    generator = torch.Generator().manual_seed(DEFAULT_CONFORMANCE_SEED)
    low, high = float(action_space.low[0]), float(action_space.high[0])

    obs, info = env.reset()
    _check_tensor("reset obs", obs, (n, obs_dim), torch.float32, env.device)
    _require(isinstance(info, dict), f"reset info is a {type(info).__name__}, expected a dict")

    snapshots: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    previous_info: dict[str, Any] | None = None
    steps_before = env.total_env_steps
    for step in range(steps):
        actions = (torch.rand((n, action_space.shape[0]), generator=generator) * (high - low) + low).to(env.device)
        obs, reward, terminated, truncated, info = env.step(actions)
        for name, snapshot, tensor in snapshots:
            _require(torch.equal(snapshot, tensor), f"{name} returned at step {step - 1} changed when stepping again")
        _check_tensor("obs", obs, (n, obs_dim), torch.float32, env.device)
        _check_tensor("reward", reward, (n,), torch.float32, env.device)
        _check_tensor("terminated", terminated, (n,), torch.bool, env.device)
        _check_tensor("truncated", truncated, (n,), torch.bool, env.device)
        _require(isinstance(info, dict), f"info is a {type(info).__name__}, expected a dict")
        _require(info is not previous_info, "info is the same dict object as in the previous step")
        _check_tensor("info['final_obs']", info["final_obs"], (n, obs_dim), torch.float32, env.device)
        _check_tensor("info['_final_obs']", info["_final_obs"], (n,), torch.bool, env.device)
        ended = info["_final_obs"]
        _require(
            torch.equal(terminated | truncated, ended),
            f"terminated | truncated differs from info['_final_obs'] at step {step}",
        )
        _require(
            torch.equal(info["final_obs"][~ended], obs[~ended]),
            f"info['final_obs'] differs from obs on envs that did not end at step {step}",
        )
        _require(
            bool((obs.abs() <= float(env.single_observation_space.high[0])).all()),
            f"obs leaves the observation space at step {step}",
        )
        expected_steps = steps_before + (step + 1) * n
        _require(
            env.total_env_steps == expected_steps,
            f"total_env_steps is {env.total_env_steps} after {step + 1} steps, expected {expected_steps}",
        )
        previous_info = info
        returned = {
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "final_obs": info["final_obs"],
            "_final_obs": info["_final_obs"],
        }
        snapshots = [(name, tensor.clone(), tensor) for name, tensor in returned.items()]
