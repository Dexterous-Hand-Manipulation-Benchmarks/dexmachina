"""The benchmark task: one EnvSpec that every method reads, and the few settings a run may choose.

The task values are those the FastTD3, REPPO and dexhand-ppo configs used. Every `EnvSpec` field that is not in
`RUN_FIELDS` is a task field and cannot be changed per run.
"""

from dataclasses import dataclass, fields, replace

from dexmachina.envs.constructors import parse_clip_string
from dexmachina.envs.spec import EnvSpec

BENCHMARK_CLIPS = ("box-30-230-s01-01", "mixer-0-150-s01-01")

BENCHMARK_TASK = replace(
    EnvSpec(),
    action_mode="hybrid",
    hybrid_scales=(0.1, 1.0),
    group_collisions=True,
    observe_tip_dist=True,
    observe_contact_force=True,
    early_reset_threshold=0.6,
    actuate_object=True,
    retarget_name="para",
    # Every method's adapter forces it on: rl_games-style return (obs dict, separate time-out flag) and reward scale.
    use_rl_games=True,
    imi_rew_weight=0.3,
    contact_rew_weight=3.0,
    bc_rew_weight=0.3,
    action_penalty=0.01,
    use_retarget_contact=True,
    kp_init=80.0,
    kv_init=5.0,
    upper_ratios=(0.9, 0.9, 1.0),
    lower_ratios=(0.8, 0.8, 1.0),
    curr_schedule="fixed",
    budget_env_steps=50_000_000,
    fixed_start_fraction=0.01,
    fixed_zero_fraction=0.6,
    wait_epochs=100,
    skip_grad=True,
    dialback_ep_len=80,
    curr_rew_thres=(0.6, 0.01, 0.01, 0.01),
    max_epochs=10000,
    # A fraction of the resets start at a random demo frame instead of the first one.
    rand_init_ratio=0.02,
)


@dataclass(frozen=True)
class RunChoices:
    """What a run may choose; every other env setting is fixed by `BENCHMARK_TASK`."""

    hand: str
    clip: str
    seed: int
    num_envs: int
    vis: bool = EnvSpec.vis
    record_video: bool = EnvSpec.record_video
    record_interval: int = EnvSpec.record_interval


RUN_FIELDS = tuple(f.name for f in fields(RunChoices))


def benchmark_spec(choices: RunChoices) -> EnvSpec:
    wanted = parse_clip_string(choices.clip)
    matches = [clip for clip in BENCHMARK_CLIPS if parse_clip_string(clip) == wanted]
    if not matches:
        raise ValueError(
            f"clip {choices.clip!r} is not a benchmark clip. The benchmark compares methods only on "
            f"{', '.join(BENCHMARK_CLIPS)}; pick one of them. The short form object-start-end is accepted too."
        )
    run = replace(choices, clip=matches[0])
    return replace(BENCHMARK_TASK, **{name: getattr(run, name) for name in RUN_FIELDS})
