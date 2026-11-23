"""DexMachina as a vectorised env with the Gymnasium vector API (duck-typed, torch tensors on the env's device).

`DexMachinaVecEnv` wraps a built `BaseEnv` and changes nothing about which envs reset or when. It adds what the
BaseEnv API lacks: the observation an env had before `step` reset it (`info["final_obs"]`), the reason each episode
ended (`info["end_cause"]`), terminated and truncated flags that follow from those reasons, and returned tensors that
the env does not overwrite on the next step.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Any, Protocol, cast

import torch

from dexmachina.envs.factory import make_env
from dexmachina.envs.maniptrans_curr import ManipTransCurriculum
from dexmachina.envs.spec import EnvSpec

# gymnasium.vector.AutoresetMode.SAME_STEP.value; the enum does not exist in gymnasium 0.29 (the fasttd3 extra).
AUTORESET_SAME_STEP = "SameStep"
POLICY_OBS_KEY = "policy"


class EnvInternals(Protocol):
    """The parts of `BaseEnv` that `DexMachinaVecEnv` reads (BaseEnv itself is untyped)."""

    num_envs: int
    device: str
    obs_dim: int
    action_dim: int
    obs_clip: float
    action_clip: float
    use_rl_games: bool
    chunk_ep_length: int
    demo_length: int
    n_objects: int
    curriculum: object | None
    nan_envs: torch.Tensor
    fell_off_mask: torch.Tensor
    early_reset_overrides: torch.Tensor
    early_reset_mask: torch.Tensor

    def get_observations(self) -> dict[str, torch.Tensor]: ...

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]: ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...

    def reset_idx(self, env_idxs: torch.Tensor, *args: Any, **kwargs: Any) -> None: ...

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class EndCauses:
    """Why each env's episode ended in one step: one bool tensor of shape (num_envs,) per cause.

    Causes overlap (an object can drop in the same step as the demo ends). Only envs that BaseEnv reset are True.
    """

    demo_end: torch.Tensor
    dropped: torch.Tensor
    early_reset: torch.Tensor
    maniptrans: torch.Tensor
    nan: torch.Tensor
    chunk_timeout: torch.Tensor

    @property
    def terminated(self) -> torch.Tensor:
        return self.demo_end | self.dropped | self.early_reset | self.maniptrans | self.nan

    @property
    def truncated(self) -> torch.Tensor:
        return self.chunk_timeout

    def masked(self, mask: torch.Tensor) -> "EndCauses":
        return EndCauses(*(cause & mask for cause in self._tensors()))

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        return (self.demo_end, self.dropped, self.early_reset, self.maniptrans, self.nan, self.chunk_timeout)


def end_causes_from_dones(env: EnvInternals, timeout: torch.Tensor) -> EndCauses:
    """The causes behind what BaseEnv._get_dones just returned, before the ManipTrans term and the reset mask.

    `timeout` is `_get_dones`' second output; it includes NaN envs, which are reported as NaN only.
    """
    no_cause = torch.zeros_like(env.nan_envs)
    time_end = timeout & ~env.nan_envs
    chunked = env.chunk_ep_length > 0
    return EndCauses(
        demo_end=no_cause if chunked else time_end,
        # A drop only resets when an early-reset check did not overwrite need_reset.
        dropped=env.fell_off_mask & ~env.early_reset_overrides,
        early_reset=env.early_reset_mask,
        maniptrans=no_cause,
        nan=env.nan_envs.clone(),
        chunk_timeout=time_end if chunked else no_cause,
    )


@dataclass(slots=True)
class _StepCapture:
    """What the hooks on one `BaseEnv.step` call recorded."""

    final_obs: torch.Tensor
    reset_mask: torch.Tensor
    causes: EndCauses | None = None
    dones_reset: torch.Tensor | None = None


class DexMachinaVecEnv:
    """Gymnasium-style vector env over a built BaseEnv (built with `use_rl_games=True`).

    Resets happen inside `step`, in the same step the episode ends (autoreset mode "SameStep"): the returned
    observation belongs to the new episode and `info["final_obs"]` holds the one the ended episode had. Actions are
    clamped to `[-action_clip, action_clip]` and scaled by the env, as before. A caller's `reset()` is outside `step`
    and reports no flags.
    """

    metadata: dict[str, Any] = {"autoreset_mode": AUTORESET_SAME_STEP}
    # What `from_spec` built the env from, so a run can save it; None for an env wrapped directly.
    spec: EnvSpec | None = None
    reward_overrides: Mapping[str, Any] | None = None

    def __init__(self, env: EnvInternals) -> None:
        if not env.use_rl_games:
            raise ValueError(
                "DexMachinaVecEnv needs a BaseEnv built with use_rl_games=True: its step then returns the "
                "observation dict and the separate time-out flag. Set use_rl_games=True in the EnvSpec."
            )
        chunk = env.chunk_ep_length
        if chunk > 0 and 2 * chunk - 1 > env.demo_length:
            raise ValueError(
                f"chunk_ep_length={chunk} is too long for a demo of {env.demo_length} steps: a chunked episode can "
                "then run past the demo end, and that real end would be reported as truncated (time-out) instead of "
                f"terminated. Use chunk_ep_length <= {(env.demo_length + 1) // 2}."
            )
        self.env = env
        self.num_envs: int = env.num_envs
        self.device: str = env.device
        self.total_env_steps = 0
        self._capture: _StepCapture | None = None
        self._install_hooks()

    @classmethod
    def from_spec(cls, spec: EnvSpec, device: str, reward_overrides: Mapping[str, Any]) -> "DexMachinaVecEnv":
        vec_env = cls(cast(EnvInternals, make_env(spec, device, reward_overrides)))
        vec_env.spec = spec
        vec_env.reward_overrides = dict(reward_overrides)
        return vec_env

    @cached_property
    def single_observation_space(self) -> Any:
        spaces = _gymnasium_spaces()
        clip = float(self.env.obs_clip)
        return spaces.Box(-clip, clip, (self.env.obs_dim,), dtype="float32")

    @cached_property
    def single_action_space(self) -> Any:
        spaces = _gymnasium_spaces()
        clip = float(self.env.action_clip)
        return spaces.Box(-clip, clip, (self.env.action_dim,), dtype="float32")

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if seed is not None or options is not None:
            raise ValueError(
                "DexMachinaVecEnv.reset takes no seed or options: the env is seeded when it is built "
                "(EnvSpec.seed). Call reset() without arguments."
            )
        observations, _ = self.env.reset()
        self._capture = None
        return observations[POLICY_OBS_KEY].clone(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        capture = _StepCapture(
            final_obs=torch.zeros((self.num_envs, self.env.obs_dim), device=self.device),
            reset_mask=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        self._capture = capture
        try:
            observations, reward, _, _, extras = self.env.step(actions)
        finally:
            self._capture = None
        causes = self._causes_of_resets(capture)

        obs = observations[POLICY_OBS_KEY].clone()
        final_obs = torch.where(capture.reset_mask[:, None], capture.final_obs, obs)
        info: dict[str, Any] = {
            "final_obs": final_obs,
            "_final_obs": capture.reset_mask,
            "end_cause": causes,
            "log": {key: _copied(value) for key, value in extras.get("log", {}).items()},
        }
        self.total_env_steps += self.num_envs
        return obs, reward.clone(), causes.terminated, causes.truncated, info

    def close(self) -> None:
        pass

    def _causes_of_resets(self, capture: _StepCapture) -> EndCauses:
        if capture.causes is None or capture.dones_reset is None:
            raise RuntimeError(
                "BaseEnv.step did not call _get_dones, so the end causes are unknown. "
                "DexMachinaVecEnv relies on BaseEnv.step calling self._get_dones() and self.reset_idx()."
            )
        # BaseEnv.step ORs the ManipTrans early termination into reset_buf after _get_dones, so no other cause shows it.
        after_dones = capture.reset_mask & ~capture.dones_reset
        maniptrans = after_dones if _can_terminate_early(self.env) else torch.zeros_like(after_dones)
        causes = replace(capture.causes, maniptrans=maniptrans).masked(capture.reset_mask)
        unexplained = capture.reset_mask & ~(causes.terminated | causes.truncated)
        if unexplained.any():
            unexplained_ids = [int(i) for i in unexplained.nonzero().flatten()]
            raise RuntimeError(
                f"Envs {unexplained_ids} were reset by BaseEnv without a known end cause. "
                "dexmachina's reset rules and DexMachinaVecEnv's cause computation have diverged; compare "
                "BaseEnv._get_dones and BaseEnv.step with end_causes_from_dones."
            )
        return causes

    def _install_hooks(self) -> None:
        """Wraps `_get_dones` and `reset_idx` of this env instance; both still do exactly what they did."""
        original_get_dones = self.env._get_dones  # pyright: ignore[reportPrivateUsage]
        original_reset_idx = self.env.reset_idx

        def get_dones() -> tuple[torch.Tensor, torch.Tensor]:
            need_reset, timeout = original_get_dones()
            if self._capture is not None:
                self._capture.causes = end_causes_from_dones(self.env, timeout)
                self._capture.dones_reset = need_reset | timeout
            return need_reset, timeout

        def reset_idx(env_idxs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            if self._capture is not None and len(env_idxs) > 0:
                before_reset = self.env.get_observations()[POLICY_OBS_KEY]
                self._capture.final_obs[env_idxs] = before_reset[env_idxs]
                self._capture.reset_mask[env_idxs] = True
            return original_reset_idx(env_idxs, *args, **kwargs)

        setattr(self.env, "_get_dones", get_dones)
        setattr(self.env, "reset_idx", reset_idx)


def _can_terminate_early(env: EnvInternals) -> bool:
    """Whether BaseEnv.step may reset envs through the ManipTrans curriculum's early termination."""
    return isinstance(env.curriculum, ManipTransCurriculum) and env.n_objects == 1


def _copied(value: Any) -> Any:
    return value.clone() if isinstance(value, torch.Tensor) else value


def _gymnasium_spaces() -> Any:
    try:
        import gymnasium.spaces as spaces  # noqa: PLC0415  (gymnasium is optional: not every extra installs it)
    except ImportError as error:
        raise ImportError(
            "The observation and action spaces need gymnasium, which this environment does not install. "
            "Install gymnasium (e.g. `uv add gymnasium`) or do not read the spaces."
        ) from error
    return spaces
