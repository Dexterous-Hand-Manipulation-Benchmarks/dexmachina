"""Record an evaluation rollout and score it: one recorder and one scorer for every method.

The recorder never reads an env. An adapter builds one `StepRecord` per env step from the env's online ADD, taken
after the step's time counter advanced and before the reset (the state after step t meets demo frame t + 1; the last
step is scored before its reset). The scorer turns the finished `Trajectory` into every eval metric.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch

from dexmachina.envs.add_metric import ADD_AUC_THRESHOLDS, ADD_PARTS, compute_auc

if TYPE_CHECKING:
    from dexmachina.envs.base_env import BaseEnv

# The per-env ADD of a part in `extras["log"]` after a step (base_env.py).
ADD_LOG_KEY = "add_{part}"
SUCCESS_2CM_M = 0.02
SUCCESS_5CM_M = 0.05
# Marks an env whose NaN mask was never set in `Trajectory.nan_from`.
NEVER_NAN = -1
# A trajectory that is scored in memory and never stored carries no spec hash or dexmachina commit.
NOT_STORED = ""
# The default seed of every method's evaluation, set after the eval env is built (building it reseeds numpy and torch
# with the training seed). Nothing random reaches the eval rollout today (one env, no randomisation, deterministic
# actions); a fixed seed keeps it so if something random is added, instead of varying with the training seed.
EVAL_SEED = 0
# BaseEnv's reset and step with use_rl_games (its default): the observation dict comes first.
EnvReset = Callable[[], "tuple[dict[str, torch.Tensor], dict[str, Any]]"]
EnvStep = Callable[
    ["torch.Tensor"], "tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]"
]

FloatArray = npt.NDArray[np.floating[Any]]
IntArray = npt.NDArray[np.integer[Any]]
BoolArray = npt.NDArray[np.bool_]
# Turns the AUC of each part into one value. `auc_add` (the mean) stays fixed: it is the number methods are
# compared on.
PartAggregation = Callable[[Mapping[str, float]], float]
PART_AGGREGATIONS: Mapping[str, PartAggregation] = {
    "median": lambda auc: float(np.median(list(auc.values()))),
    "min": lambda auc: float(min(auc.values())),
    "max": lambda auc: float(max(auc.values())),
    "std": lambda auc: float(np.std(list(auc.values()))),
}


def paired_demo_frame(step: int, num_demo_frames: int) -> int:
    """The demo frame the env pairs with the state after `step` (0-based): t + 1, clamped like `ADDMetric.compute`.

    A clip of F frames has episode length F, so the last step pairs with frame F - 1 for the second time and frame 0
    is never scored (the team's scoring convention, kept as the env defines it).
    """
    return min(step + 1, num_demo_frames - 1)


@dataclass(frozen=True, eq=False)
class StepRecord:
    """What an adapter reports for one env step; arrays have one row per env."""

    object_root_pos: FloatArray  # (num_envs, 3)
    object_root_quat: FloatArray  # (num_envs, 4)
    object_joint_pos: FloatArray  # (num_envs, num_joints)
    add: FloatArray  # (num_envs, len(ADD_PARTS)), metres, columns in ADD_PARTS order
    nan_mask: BoolArray  # (num_envs,), the env's NaN flag before the step's reset
    reset_mask: BoolArray  # (num_envs,), the env was reset at the end of this step
    dropped: BoolArray  # (num_envs,), object below the table; a record only, it does not reset
    demo_frame: IntArray  # (num_envs,), the demo frame the env paired with this step


@dataclass(frozen=True, eq=False)
class Trajectory:
    """A finished rollout of `num_steps` steps; NaN envs hold NaN from `nan_from` on."""

    hand: str
    clip: str
    spec_hash: str
    dexmachina_sha: str
    object_root_pos: FloatArray  # (num_steps, num_envs, 3)
    object_root_quat: FloatArray  # (num_steps, num_envs, 4)
    object_joint_pos: FloatArray  # (num_steps, num_envs, num_joints)
    add: FloatArray  # (num_steps, num_envs, len(ADD_PARTS))
    dropped: BoolArray  # (num_steps, num_envs)
    demo_frame: IntArray  # (num_steps, num_envs)
    nan_from: IntArray  # (num_envs,), first NaN step or NEVER_NAN


_TRAJECTORY_TEXT_FIELDS = ("hand", "clip", "spec_hash", "dexmachina_sha")


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    """Writes the trajectory as one .npz file that `load_trajectory` reads back; an existing file is never replaced."""
    if path.exists():
        raise FileExistsError(
            f"{path} exists already. Replacing it would overwrite the trajectory of an earlier evaluation that its "
            "scores were computed from. Evaluate into a new directory, or delete the file if it is stale."
        )
    np.savez(path, **{f.name: getattr(trajectory, f.name) for f in fields(trajectory)})


def load_trajectory(path: Path) -> Trajectory:
    with np.load(path, allow_pickle=False) as stored:
        values: dict[str, Any] = {name: stored[name] for name in stored.files}
    missing = {f.name for f in fields(Trajectory)} - values.keys()
    if missing:
        raise ValueError(
            f"{path} is not a saved Trajectory: it lacks {sorted(missing)}. It was not written by save_trajectory "
            "or by another version of it. Evaluate again to write it."
        )
    for name in _TRAJECTORY_TEXT_FIELDS:
        values[name] = str(values[name])
    return Trajectory(**values)


class EvalRecorder:
    """Collects `StepRecord`s of one episode of `num_steps` steps and checks they follow the eval protocol."""

    def __init__(self, num_steps: int, num_envs: int, num_demo_frames: int) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_demo_frames = num_demo_frames
        self._records: list[StepRecord] = []
        self._nan_from: IntArray = np.full(num_envs, NEVER_NAN, dtype=np.int64)

    def record(self, record: StepRecord) -> None:
        step = len(self._records)
        if step >= self.num_steps:
            raise ValueError(
                f"step {step} was recorded but the episode has {self.num_steps} steps. The env did not reset at "
                "the end of the episode, or the recorder was reused: build a new EvalRecorder for each episode."
            )
        record = self._copied(record)
        self._set_nan_from(step, record.nan_mask)
        self._check_reset(step, record.reset_mask)
        self._check_demo_frame(step, record.demo_frame)
        self._records.append(self._without_nan_envs(record))

    def trajectory(self, hand: str, clip: str, spec_hash: str, dexmachina_sha: str) -> Trajectory:
        if len(self._records) != self.num_steps:
            raise ValueError(
                f"{len(self._records)} of {self.num_steps} steps were recorded. Record every step of the episode "
                "before building the trajectory; a shorter one would be scored as if it were a full episode."
            )
        return Trajectory(
            hand=hand,
            clip=clip,
            spec_hash=spec_hash,
            dexmachina_sha=dexmachina_sha,
            object_root_pos=np.stack([r.object_root_pos for r in self._records]),
            object_root_quat=np.stack([r.object_root_quat for r in self._records]),
            object_joint_pos=np.stack([r.object_joint_pos for r in self._records]),
            add=np.stack([r.add for r in self._records]),
            dropped=np.stack([r.dropped for r in self._records]),
            demo_frame=np.stack([r.demo_frame for r in self._records]),
            nan_from=self._nan_from.copy(),
        )

    def _set_nan_from(self, step: int, nan_mask: BoolArray) -> None:
        self._nan_from[nan_mask & (self._nan_from == NEVER_NAN)] = step

    def _nan_from_before(self, step: int) -> BoolArray:
        """Envs that are not NaN yet or turn NaN at this step; a NaN env's frame counter restarts after its reset."""
        return (self._nan_from == NEVER_NAN) | (self._nan_from == step)

    def _check_reset(self, step: int, reset_mask: BoolArray) -> None:
        expected = step == self.num_steps - 1
        wrong = (reset_mask != expected) & (self._nan_from == NEVER_NAN)
        if wrong.any():
            env = int(np.flatnonzero(wrong)[0])
            raise ValueError(
                f"env {env} had reset={bool(reset_mask[env])} at step {step}, expected {expected} (resets are "
                f"expected only at the last step, {self.num_steps - 1}). Likely cause: the eval env still has "
                "early resets (early_reset_threshold > 0), is_eval is off, or the episode length differs from the "
                "recorder's num_steps."
            )

    def _check_demo_frame(self, step: int, demo_frame: IntArray) -> None:
        expected = paired_demo_frame(step, self.num_demo_frames)
        wrong = (demo_frame != expected) & self._nan_from_before(step)
        if wrong.any():
            env = int(np.flatnonzero(wrong)[0])
            raise ValueError(
                f"env {env} paired demo frame {int(demo_frame[env])} with step {step}, expected {expected} (state "
                "after step t meets frame t + 1). Likely cause: the adapter read the frame counter before the "
                "step instead of after it, or after the reset."
            )

    def _copied(self, record: StepRecord) -> StepRecord:
        """A private copy: dexmachina updates its buffers in place, so stored views would change later."""
        for field in fields(record):
            values: npt.NDArray[Any] = getattr(record, field.name)
            if values.shape[0] != self.num_envs:
                raise ValueError(
                    f"StepRecord.{field.name} has {values.shape[0]} rows, expected one per env ({self.num_envs}). "
                    "Likely cause: the adapter built the record from a different env than the recorder's num_envs."
                )
        return replace(record, **{field.name: getattr(record, field.name).copy() for field in fields(record)})

    def _without_nan_envs(self, record: StepRecord) -> StepRecord:
        """NaN envs hold NaN from their first NaN step on, so a later cleared flag changes nothing."""
        nan_env = self._nan_from != NEVER_NAN
        record.add[nan_env] = np.nan
        record.object_root_pos[nan_env] = np.nan
        record.object_root_quat[nan_env] = np.nan
        record.object_joint_pos[nan_env] = np.nan
        return record


class _PreResetTap:
    """Copies the object state and the env's counters inside a step, before the reset overwrites them.

    `BaseEnv.step` computes the ADD right after `_compute_intermediate_values`, so the state read there is the one
    that was scored.
    """

    def __init__(self, env: "BaseEnv") -> None:
        self.env = env
        self.root_pos: FloatArray = np.empty(0)
        self.root_quat: FloatArray = np.empty(0)
        self.joint_pos: FloatArray = np.empty(0)
        self.dropped: BoolArray = np.empty(0, dtype=np.bool_)
        self.episode_length: IntArray = np.empty(0, dtype=np.int64)
        self.nan_mask: BoolArray = np.empty(0, dtype=np.bool_)
        self._compute_intermediate_values: Callable[[], None] = env._compute_intermediate_values  # pyright: ignore[reportPrivateUsage]
        env._compute_intermediate_values = self._tapped  # pyright: ignore[reportPrivateUsage]

    def release(self) -> None:
        self.env._compute_intermediate_values = self._compute_intermediate_values  # pyright: ignore[reportPrivateUsage]

    def _tapped(self) -> None:
        self._compute_intermediate_values()
        obj = self.env.object
        assert obj is not None
        self.root_pos = obj.root_pos.detach().cpu().numpy().astype(np.float64)
        self.root_quat = obj.root_quat.detach().cpu().numpy().astype(np.float64)
        self.joint_pos = obj.dof_pos.detach().cpu().numpy().astype(np.float64)
        # The same test as BaseEnv._get_dones; under is_eval it is a record only, not a reset.
        self.dropped = (obj.root_pos[:, 2] < self.env.table_height).detach().cpu().numpy()
        self.episode_length = self.env.episode_length_buf.detach().cpu().numpy().astype(np.int64)
        self.nan_mask = self.env.nan_envs.detach().cpu().numpy().copy()


def rollout_episode(
    env: "BaseEnv",
    next_actions: "Callable[[torch.Tensor], torch.Tensor]",
    hand: str,
    clip: str,
    spec_hash: str,
    dexmachina_sha: str,
) -> Trajectory:
    """Runs one eval episode of an env from `make_eval_env` and records it.

    `next_actions` maps the observation before each step to that step's actions.
    """
    if env.object is None or env.object.demo_states is None:
        raise ValueError(
            f"the env for {hand}/{clip} has no object with a demo trajectory, so there is no ADD to record. "
            "Evaluate a spec with an object and a retargeted clip."
        )
    num_steps: int = env.max_episode_length
    num_demo_frames: int = env.object.demo_states.shape[0]
    recorder = EvalRecorder(num_steps, env.num_envs, num_demo_frames)
    tap = _PreResetTap(env)
    env_step = cast(EnvStep, env.step)
    observations, _ = cast(EnvReset, env.reset)()
    try:
        for _ in range(num_steps):
            observations, _, _, time_outs, extras = env_step(next_actions(observations["policy"]))
            add = np.stack(
                [extras["log"][ADD_LOG_KEY.format(part=part)].detach().cpu().numpy() for part in ADD_PARTS], axis=-1
            )
            recorder.record(
                StepRecord(
                    object_root_pos=tap.root_pos,
                    object_root_quat=tap.root_quat,
                    object_joint_pos=tap.joint_pos,
                    add=add.astype(np.float64),
                    nan_mask=tap.nan_mask,
                    reset_mask=time_outs.detach().cpu().numpy().copy(),
                    dropped=tap.dropped,
                    # The env's counter is not clamped to the last demo frame at the last step; ADDMetric clamps it.
                    demo_frame=np.minimum(tap.episode_length, num_demo_frames - 1),
                )
            )
    finally:
        tap.release()
    return recorder.trajectory(hand, clip, spec_hash, dexmachina_sha)


@dataclass(frozen=True)
class EvalScore:
    """Every eval metric of one trajectory. ADD in metres, the rest in [0, 1]; not normalised by the floor."""

    auc_add: float
    auc_add_of_mean: float
    auc_add_part: Mapping[str, float]
    mean_add: float
    success_2cm: float
    success_5cm: float
    nan_rate: float
    drop_rate: float
    # Each of the caller's `part_aggregations` applied to `auc_add_part`, under the caller's name for it.
    auc_add_aggregated: Mapping[str, float] = field(default_factory=dict[str, float])


def score(trajectory: Trajectory, part_aggregations: Mapping[str, PartAggregation] | None = None) -> EvalScore:
    """Score a trajectory over all (env, frame) values; a NaN frame is a miss at every threshold.

    `auc_add` is the paper's definition: AUC over `ADD_AUC_THRESHOLDS` (strict `<`) per part, then the mean over
    parts. `auc_add_of_mean` takes the AUC of the part-averaged ADD instead (REPPO's online definition).
    `mean_add` averages the frames before each env's first NaN. The state after the last step is scored against the
    last demo frame, which the env pairs twice; frame 0 is never scored.
    `part_aggregations` adds more summaries of the per-part AUCs: entries of `PART_AGGREGATIONS` or any function.
    """
    num_steps = trajectory.add.shape[0]
    scored = np.arange(num_steps)[:, None] < _nan_start(trajectory.nan_from, num_steps)[None, :]
    # inf is above every threshold, so an unscored frame counts as a miss in the AUC and the success rates.
    add = np.where(scored[..., None], trajectory.add, np.inf)
    per_part = {part: compute_auc(add[..., i], ADD_AUC_THRESHOLDS) for i, part in enumerate(ADD_PARTS)}
    return EvalScore(
        auc_add=float(np.mean(list(per_part.values()))),
        auc_add_of_mean=compute_auc(add.mean(axis=-1), ADD_AUC_THRESHOLDS),
        auc_add_part=per_part,
        mean_add=float(trajectory.add[scored].mean()) if scored.any() else float("nan"),
        success_2cm=_success_rate(add, SUCCESS_2CM_M),
        success_5cm=_success_rate(add, SUCCESS_5CM_M),
        nan_rate=float((trajectory.nan_from != NEVER_NAN).mean()),
        drop_rate=float((trajectory.dropped & scored).any(axis=0).mean()),
        auc_add_aggregated={name: aggregate(per_part) for name, aggregate in (part_aggregations or {}).items()},
    )


def zero_action_floor(env: "BaseEnv", hand: str, clip: str) -> EvalScore:
    """What the all-zero action scores on `env` (from `make_eval_env`): the level a policy must beat to have learned.

    Run it on the env of the evaluation, before the policy's episode: both episodes then share the env, so the floor
    cannot differ from the protocol the policy is scored on. `rollout_episode` resets the env first.
    """
    zero = torch.zeros((env.num_envs, env.action_dim), device=env.device)
    trajectory = rollout_episode(
        env, lambda _observation: zero, hand, clip, spec_hash=NOT_STORED, dexmachina_sha=NOT_STORED
    )
    return score(trajectory)


def _nan_start(nan_from: IntArray, num_steps: int) -> IntArray:
    return np.where(nan_from == NEVER_NAN, num_steps, nan_from)


def _success_rate(add: FloatArray, threshold: float) -> float:
    return float(np.mean([np.mean(add[..., i] < threshold) for i in range(len(ADD_PARTS))]))
