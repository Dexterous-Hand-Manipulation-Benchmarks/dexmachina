"""Build a DexMachina env from an `EnvSpec` and check that the built env uses what the spec says."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Protocol, cast

from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.constructors import get_all_env_cfg
from dexmachina.envs.spec import EnvSpec

CONTACT_REW_WEIGHT_KEY = "contact_rew_weight"
IMI_REW_WEIGHT_KEY = "imi_rew_weight"
BC_REW_WEIGHT_KEY = "bc_rew_weight"


@dataclass(frozen=True, slots=True)
class AppliedEnvConfig:
    """What a built env uses, to compare with what the spec asked for."""

    use_contact_reward: bool
    object_actuated: bool
    use_curriculum: bool
    zero_epoch: int | None
    reward_weights: Mapping[str, Any]
    use_imi_rew: bool
    use_bc_rew: bool
    hand: str

    @classmethod
    def read_from(cls, env: BaseEnv, reward_keys: Iterable[str]) -> "AppliedEnvConfig":
        """Reads the values from a built env; reward_keys are the overrides its RewardModule caches."""
        curriculum = env.curriculum
        return cls(
            use_contact_reward=env.use_contact_reward,
            object_actuated=env.object is not None and env.object.actuated,
            use_curriculum=env.use_curriculum,
            zero_epoch=None if curriculum is None else curriculum.zero_epoch,
            reward_weights={key: getattr(env.reward_module, key) for key in reward_keys},
            use_imi_rew=env.reward_module.use_imi_rew,
            use_bc_rew=env.reward_module.use_bc_rew,
            hand=env.robots["left"].name,
        )

    @classmethod
    def expected(
        cls, spec: EnvSpec, reward_cfg: Mapping[str, Any], reward_keys: Iterable[str]
    ) -> "AppliedEnvConfig":
        """The values the env should end up with, from the spec and the final reward config."""
        object_actuated = spec.actuate_object and not spec.no_object
        # The curriculum runs on an actuated object (base_env.py) or, as the ManipTrans ablation, on a fixed one.
        use_curriculum = object_actuated or spec.maniptrans
        return cls(
            use_contact_reward=reward_cfg[CONTACT_REW_WEIGHT_KEY] > 0,
            object_actuated=object_actuated,
            use_curriculum=use_curriculum,
            # get_all_env_cfg sets the curriculum's zero_epoch this way (envs/constructors.py).
            zero_epoch=max(spec.max_epochs - spec.num_zero_epoch, 0) if use_curriculum else None,
            reward_weights={key: reward_cfg[key] for key in reward_keys},
            use_imi_rew=reward_cfg[IMI_REW_WEIGHT_KEY] > 0,
            use_bc_rew=reward_cfg[BC_REW_WEIGHT_KEY] > 0,
            # A robot is named "<hand>_<side>" (get_default_robot_cfg in envs/robot.py).
            hand=f"{spec.hand}_left",
        )

    def mismatches(self, applied: "AppliedEnvConfig") -> list[str]:
        """One "name: configured X, applied Y" line per differing value, reward weights per key."""
        pairs: list[tuple[str, object, object]] = [
            (field.name, getattr(self, field.name), getattr(applied, field.name))
            for field in fields(self)
            if field.name != "reward_weights"
        ]
        pairs += [
            (f"reward_weights.{key}", value, applied.reward_weights[key])
            for key, value in self.reward_weights.items()
        ]
        return [
            f"{name}: configured {configured}, applied {actual}"
            for name, configured, actual in pairs
            if configured != actual
        ]


def verify_applied_config(env: BaseEnv, expected: AppliedEnvConfig) -> None:
    """Raises if the built env does not use the values the spec asked for."""
    mismatches = expected.mismatches(AppliedEnvConfig.read_from(env, expected.reward_weights))
    if mismatches:
        raise ValueError("dexmachina env does not use the configured values: " + "; ".join(mismatches))


def build_env_kwargs(spec: EnvSpec, device: str, reward_overrides: Mapping[str, Any]) -> dict[str, Any]:
    """The BaseEnv keyword arguments for the spec; reward_overrides set reward keys that have no command line flag."""
    env_kwargs = get_all_env_cfg(spec.to_args(), device=device)
    env_kwargs["reward_cfg"].update(reward_overrides)
    return env_kwargs


def make_env(spec: EnvSpec, device: str, reward_overrides: Mapping[str, Any]) -> BaseEnv:
    """Builds the env for the spec; reward_overrides set reward keys that have no command line flag.

    A key that is also a spec field (e.g. contact_rew_weight) belongs in the spec: dexmachina derives other settings
    from the spec value (e.g. env_cfg["use_contact_reward"]), so an override that disagrees with the spec can make
    the read-back raise.
    """
    env_kwargs = build_env_kwargs(spec, device, reward_overrides)
    env = BaseEnv(**env_kwargs)
    cached_reward_keys = [key for key in reward_overrides if hasattr(env.reward_module, key)]
    verify_applied_config(env, AppliedEnvConfig.expected(spec, env_kwargs["reward_cfg"], cached_reward_keys))
    return env


class ObjectInternals(Protocol):
    actuated: bool


class AddMetricInternals(Protocol):
    ready: bool


class EvalEnvInternals(Protocol):
    """The parts of `BaseEnv` that the eval read-back reads (BaseEnv itself is untyped)."""

    is_eval: bool
    early_reset_threshold: float
    rand_init_ratio: float
    rand_cfg: Mapping[str, Any]
    num_envs: int
    object: ObjectInternals | None
    add_metric: AddMetricInternals | None
    record_curriculum: bool


@dataclass(frozen=True, slots=True)
class AppliedEvalConfig:
    """The settings of a built env that the evaluation protocol depends on."""

    is_eval: bool
    early_reset_threshold: float
    rand_init_ratio: float
    randomized: bool
    num_envs: int
    object_actuated: bool
    add_metric_ready: bool
    record_curriculum: bool

    @classmethod
    def read_from(cls, env: EvalEnvInternals) -> "AppliedEvalConfig":
        return cls(
            is_eval=env.is_eval,
            early_reset_threshold=env.early_reset_threshold,
            rand_init_ratio=env.rand_init_ratio,
            randomized=env.rand_cfg["randomize"],
            num_envs=env.num_envs,
            object_actuated=env.object is not None and env.object.actuated,
            add_metric_ready=env.add_metric is not None and env.add_metric.ready,
            record_curriculum=env.record_curriculum,
        )

    def mismatches(self, applied: "AppliedEvalConfig") -> list[str]:
        """One "name: expected X, applied Y" line per differing value."""
        return [
            f"{field.name}: expected {getattr(self, field.name)}, applied {getattr(applied, field.name)}"
            for field in fields(self)
            if getattr(self, field.name) != getattr(applied, field.name)
        ]


EVAL_CONFIG = AppliedEvalConfig(
    is_eval=True,
    early_reset_threshold=0.0,
    rand_init_ratio=0.0,
    randomized=False,
    num_envs=1,
    object_actuated=False,
    add_metric_ready=True,
    record_curriculum=False,
)


# Spec fields that cannot change the eval rollout, left out of `task_hash` so that every seed and method trained on
# one task shares its hash: display and recording, the dead `is_eval` field, and the seed and object-gain curriculum,
# which only the curriculum reads; `make_eval_env` builds no curriculum because it forces actuate_object=False and
# refuses maniptrans. Exclude-list on purpose: a field missing here gives one task two hashes (two groups in the eval
# table); a field wrongly listed here would merge two tasks silently.
NOT_IN_TASK_HASH = frozenset(
    {
        "vis", "overlay", "n_envs_per_row", "show_markers", "show_kpts", "record_video", "render_camera", "raytrace",
        "record_interval", "show_fps", "color_object", "vis_contact", "is_eval", "seed",
        "interval", "first_ratio", "first_stop_iter", "second_stop_iter", "kp_init", "kv_init", "force_range_init",
        "max_epochs", "num_zero_epoch", "curr_schedule", "fixed_mode", "decay_rew", "wait_epochs", "decrease_rew",
        "curr_rew_thres", "deque_freq", "grad_threshold", "gain_mode", "uniform_mode", "upper_ratios",
        "lower_ratios", "deque_len", "resample_every_epoch", "skip_grad", "dialback_ep_len", "dialback_min_epochs",
        "dialback_ratios", "budget_env_steps", "fixed_start_fraction", "fixed_zero_fraction",
    }
)  # fmt: skip
TASK_HASH_HEX_DIGITS = 12


def _eval_spec(spec: EnvSpec) -> EnvSpec:
    return replace(spec, num_envs=1, actuate_object=False, rand_init_ratio=0.0, early_reset_threshold=0.0)


def task_hash(spec: EnvSpec) -> str:
    """Names the eval task a training spec is evaluated on: equal for specs whose eval rollouts can only be equal."""
    task = {name: _as_float(value) for name, value in asdict(_eval_spec(spec)).items() if name not in NOT_IN_TASK_HASH}
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:TASK_HASH_HEX_DIGITS]


def _as_float(value: Any) -> Any:
    """Numbers as floats: specs from code and from argv differ in int vs float (10 vs 10.0) but are equal."""
    if isinstance(value, tuple):
        return [_as_float(item) for item in cast(tuple[Any, ...], value)]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def make_eval_env(spec: EnvSpec, device: str) -> BaseEnv:
    """Builds the single-env, unassisted, reset-free env that evaluation rolls out, from the training spec.

    The episode always runs to the last demo frame (`is_eval`), the object has no assistive gains, the online ADD
    metric is attached, and the env never feeds the curriculum. Call `env.reset()` before every episode.
    """
    if spec.maniptrans:
        raise ValueError(
            "make_eval_env does not support ManipTrans specs: its curriculum terminates episodes early even under "
            "is_eval, so an episode would not run to the last demo frame. Evaluate with a spec without maniptrans."
        )
    if not spec.use_rl_games:
        raise ValueError(
            "make_eval_env needs a spec with use_rl_games=True: the eval rollout reads the policy observation from the "
            "observation dict that only the rl_games step returns. Set use_rl_games=True in the EnvSpec."
        )
    eval_spec = _eval_spec(spec)
    env_kwargs = build_env_kwargs(eval_spec, device, {})
    # is_eval is a dead field of the spec and no spec field reaches the ADD metric or randomisation, so set them here.
    env_kwargs["env_cfg"]["is_eval"] = True
    env_kwargs["env_cfg"]["log_add_metric"] = True
    env_kwargs["rand_cfg"]["randomize"] = False
    env = BaseEnv(**env_kwargs)
    verify_applied_config(env, AppliedEnvConfig.expected(eval_spec, env_kwargs["reward_cfg"], []))
    env.record_curriculum = False
    mismatches = EVAL_CONFIG.mismatches(AppliedEvalConfig.read_from(cast(EvalEnvInternals, env)))
    if mismatches:
        raise ValueError(
            "the eval env does not have the settings the evaluation protocol needs: " + "; ".join(mismatches) + ". "
            "Without them the rollout would be assisted, reset early or randomised, and its scores not comparable. "
            "Likely cause: dexmachina changed how these settings are read; update make_eval_env to match."
        )
    return env
