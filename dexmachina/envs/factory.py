"""Build a DexMachina env from an `EnvSpec` and check that the built env uses what the spec says."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from typing import Any

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


def make_env(spec: EnvSpec, device: str, reward_overrides: Mapping[str, Any]) -> BaseEnv:
    """Builds the env for the spec; reward_overrides set reward keys that have no command line flag.

    A key that is also a spec field (e.g. contact_rew_weight) belongs in the spec: dexmachina derives other settings
    from the spec value (e.g. env_cfg["use_contact_reward"]), so an override that disagrees with the spec can make
    the read-back raise.
    """
    env_kwargs = get_all_env_cfg(spec.to_args(), device=device)
    reward_cfg: dict[str, Any] = env_kwargs["reward_cfg"]
    reward_cfg.update(reward_overrides)
    env = BaseEnv(**env_kwargs)
    cached_reward_keys = [key for key in reward_overrides if hasattr(env.reward_module, key)]
    verify_applied_config(env, AppliedEnvConfig.expected(spec, reward_cfg, cached_reward_keys))
    return env
