"""All DexMachina env settings with their defaults, in one frozen dataclass.

The command line parser (`get_common_argparser`) is generated from `EnvSpec`, so every setting, its flags and its
default are defined once, here.
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, cast, get_args, get_origin, get_type_hints

ARG_METADATA_KEY = "argparse"


@dataclass(frozen=True)
class ArgSpec:
    """What the command line needs besides the field's name, type and default."""

    flags: tuple[str, ...]
    help: str | None = None
    choices: tuple[str, ...] | None = None
    # `action_mode` has always been parsed without a type function; keep that exactly.
    untyped: bool = False


def option(
    default: Any,
    *flags: str,
    help: str | None = None,
    choices: tuple[str, ...] | None = None,
    untyped: bool = False,
) -> Any:
    return field(default=default, metadata={ARG_METADATA_KEY: ArgSpec(flags, help, choices, untyped)})


@dataclass(frozen=True)
class EnvSpec:
    num_envs: int = option(1, "-B", "--num_envs")
    vis: bool = option(False, "--vis", "-v")
    overlay: bool = option(False, "--overlay", "-ov")
    n_envs_per_row: int | None = option(None, "--n_envs_per_row", "-nrow")
    texture_object: bool = option(
        False,
        "--texture_object",
        "-to",
        help="Show texture for the object and hide the actual URDF",
    )
    arctic_object: str = option("box", "--arctic_object", "-ao")
    hand: str = option("inspire_hand", "--hand")
    frame_start: int = option(40, "--frame_start", "-fs")
    frame_end: int = option(200, "--frame_end", "-fe")
    clip: str = option("box-40-200-s01-u01", "--clip", "-cl")
    show_markers: bool = option(False, "--show_markers", help="Whether to show contact markers")
    interp: int = option(1, "--interp", help="Interpolation multiplier for the demo data")
    seed: int = option(42, "--seed")
    action_mode: str = option(
        "residual",
        "--action_mode",
        "-am",
        choices=("residual", "absolute", "relative", "hybrid", "kinematic"),
        untyped=True,
    )
    show_kpts: bool = option(False, "--show_kpts", help="Whether to show keypoints")
    res_cap: bool = option(False, "--res_cap")
    hybrid_scales: tuple[float, ...] = option((0.04, 0.5), "--hybrid_scales")
    hide_hand: bool = option(False, "--hide_hand")
    last_n_frame: int = option(-1, "--last_n_frame")
    early_reset_threshold: float = option(0.5, "--early_reset_threshold", "-ert")
    aux_reset_thres: tuple[float, ...] = option((0, 0, 0), "--aux_reset_thres")
    record_video: bool = option(False, "--record_video", "-rv")
    render_camera: str = option("front", "--render_camera", "-rcam")
    raytrace: bool = option(False, "--raytrace", "-ray")
    record_interval: int = option(-1, "--record_interval", "-ri")
    use_teleop: bool = option(False, "--use_teleop")
    retarget_name: str = option("genesis", "--retarget_name", "-rn")
    teleop_fname: str = option("data/scripted/tmp_box/tmp_box.npz", "--teleop_fname")
    observe_tip_dist: bool = option(False, "--observe_tip_dist", "-obt")
    observe_contact_force: bool = option(False, "--observe_contact_force", "-obf")
    task_rew_betas: tuple[float, ...] = option((10, 1, 5), "--task_rew_betas", "-trb")
    no_object: bool = option(False, "--no_object", "-no_obj")
    show_fps: bool = option(False, "--show_fps", "-fps")
    color_object: str | None = option(None, "--color_object", "-co", help="Color of the object to be rendered")
    action_penalty: float = option(0.0, "--action_penalty", "-ap")
    imi_rew_weight: float = option(0.0, "--imi_rew_weight", "-imi")
    imi_wrist_weight: float = option(0.5, "--imi_wrist_weight", "-imw")
    contact_rew_weight: float = option(0.0, "--contact_rew_weight", "-con")
    bc_rew_weight: float = option(0.0, "--bc_rew_weight", "-bc")
    bc_beta: float = option(500.0, "--bc_beta", "-beta")
    contact_beta: float = option(10.0, "--contact_beta", "-cbeta")
    skip_wrist_frame_contact: bool = option(False, "--skip_wrist_frame_contact", "-swfc")
    nomask_zero_contact: bool = option(False, "--nomask_zero_contact")
    contact_rew_function: str = option("exp", "--contact_rew_function", "-crf", choices=("exp", "sigmoid"))
    use_retarget_contact: bool = option(False, "--use_retarget_contact", "-urc")
    retarget_worldfr: bool = option(False, "--retarget_worldfr", "-rwf")
    kp: float = option(200.0, "--kp", "-kp")
    kv: float = option(20.0, "--kv", "-kv")
    force_range: float = option(100.0, "--force_range", "-fr")
    obj_stiffness: float = option(0.0, "--obj_stiffness", "-ostiff")
    obj_damping: float = option(0.0, "--obj_damping", "-odamp")
    vis_contact: bool = option(False, "--vis_contact", "-vc")
    use_rl_games: bool = option(False, "--use_rl_games", "-rlg")
    is_eval: bool = option(False, "--is_eval", "-eval")
    rand_init_ratio: float = option(0.0, "--rand_init_ratio", "-randr")
    chunk_ep_length: int = option(-1, "--chunk_ep_length", "-chunk")
    use_rand: bool = option(False, "--use_rand", "-rand")
    rand_friction: bool = option(False, "--rand_friction", "-rf")
    rand_com: bool = option(False, "--rand_com", "-rc")
    rand_mass: bool = option(False, "--rand_mass", "-rm")
    external_force: bool = option(False, "--external_force", "-ef")
    force_scale: float = option(5.0, "--force_scale", "-fscale")
    torque_scale: float = option(0.0, "--torque_scale", "-tscale")
    convexify_object: bool = option(False, "--convexify_object", "-cvf")
    actuate_object: bool = option(False, "--actuate_object", "-act")
    objdex_baseline: bool = option(False, "--objdex_baseline", "-objdex")
    multiply_all_rew: bool = option(False, "--multiply_all_rew", "-mar")
    group_collisions: bool = option(False, "--group_collisions", "-gcol")
    interval: int = option(500, "--interval")
    first_ratio: float = option(0.3, "--first_ratio")
    first_stop_iter: int = option(7000, "--first_stop_iter")
    second_stop_iter: int = option(16000, "--second_stop_iter")
    kp_init: float = option(100.0, "--kp_init")
    kv_init: float = option(10.0, "--kv_init")
    force_range_init: float = option(50.0, "--force_range_init")
    max_epochs: int = option(5000, "--max_epochs", "-me", help="Maximum number of epochs to train the agent.")
    num_zero_epoch: int = option(1000, "--num_zero_epoch", "-nze")
    curr_schedule: str = option("uniform", "--curr_schedule", choices=("fixed", "exp", "uniform"))
    fixed_mode: str = option("uniform", "--fixed_mode", choices=("exp", "lin", "uniform"))
    decay_rew: bool = option(False, "--decay_rew", "-drw")
    wait_epochs: int = option(500, "--wait_epochs")
    decrease_rew: bool = option(False, "--decrease_rew", "-dr")
    curr_rew_thres: tuple[float, ...] = option((0.5, 0, 0, 0), "--curr_rew_thres")
    deque_freq: int = option(1, "--deque_freq")
    grad_threshold: float = option(0.001, "--grad_threshold")
    gain_mode: str = option("all", "--gain_mode", "-gm", choices=("all", "kpkv", "fr"))
    uniform_mode: str = option("slow", "--uniform_mode", "-um", choices=("fast", "slow"))
    upper_ratios: tuple[float, ...] = option((0.8, 0.95, 0.95), "--upper_ratios")
    lower_ratios: tuple[float, ...] = option((0.7, 0.9, 0.9), "--lower_ratios")
    deque_len: int = option(30, "--deque_len")
    decay_solimp: bool = option(False, "--decay_solimp", "-ds")
    solip_multiplier: float = option(0.95, "--solip_multiplier", "-solip")
    resample_every_epoch: int = option(-1, "--resample_every_epoch", "-resample")
    skip_grad: bool = option(False, "--skip_grad")
    dialback_ep_len: int = option(30, "--dialback_ep_len")
    dialback_min_epochs: int = option(500, "--dialback_min_epochs")
    dialback_ratios: tuple[float, ...] = option((0.98, 1.0, 1.0), "--dialback_ratios")
    maniptrans: bool = option(False, "--maniptrans", "-mpt")
    friction_range: tuple[float, ...] = option((4, 1), "--friction_range")
    eps_finger_range: tuple[float, ...] = option((0.1, 0.05), "--eps_finger_range", "-efr")
    eps_object_Prange: tuple[float, ...] = option((0.1, 0.03), "--eps_object_Prange", "-eopr")
    eps_object_Rrange: tuple[float, ...] = option((1.57, 0.52), "--eps_object_Rrange", "-eorr")
    maniptrans_mode: str = option("fixed", "--maniptrans_mode", "-mpmode", choices=("fixed", "auto"))
    maniptrans_schedule: str = option("exp", "--maniptrans_schedule", "-mpsch", choices=("exp", "linear"))

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EnvSpec":
        """Read the spec from a parsed namespace; attributes that are not env settings are ignored."""
        values: dict[str, Any] = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {name: tuple(v) if isinstance(v, list) else v for name, v in values.items()}
        return cls(**kwargs)

    def to_args(self) -> argparse.Namespace:
        """The namespace the parser produces for these settings."""
        values = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return argparse.Namespace(**{name: list(v) if isinstance(v, tuple) else v for name, v in values.items()})

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EnvSpec":
        """The spec `to_json` wrote; every field must be present, so nothing silently takes today's default."""
        values: dict[str, Any] = json.loads(text)
        names = {f.name for f in dataclasses.fields(cls)}
        missing, unknown = sorted(names - values.keys()), sorted(values.keys() - names)
        if missing or unknown:
            raise ValueError(
                f"The saved EnvSpec does not match this dexmachina's EnvSpec: missing fields {missing}, unknown fields "
                f"{unknown}. EnvSpec changed since the spec was saved, so a default would stand in for the value the "
                "run used. Add each missing field with the value the run used (or remove the unknown ones) and load "
                "again."
            )
        kwargs: dict[str, Any] = {name: tuple(v) if isinstance(v, list) else v for name, v in values.items()}
        return cls(**kwargs)


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    """Add one command line argument per `EnvSpec` field, in field order."""
    hints = get_type_hints(EnvSpec)
    for f in dataclasses.fields(EnvSpec):
        arg: ArgSpec = f.metadata[ARG_METADATA_KEY]
        hint = hints[f.name]
        if hint is bool:
            parser.add_argument(*arg.flags, action="store_true", help=arg.help)
            continue
        kwargs: dict[str, Any] = {"default": f.default, "help": arg.help, "choices": arg.choices}
        if get_origin(hint) is tuple:
            kwargs["type"] = get_args(hint)[0]
            kwargs["nargs"] = "+"
            kwargs["default"] = list(cast(tuple[float, ...], f.default))
        elif not arg.untyped:
            kwargs["type"] = next(t for t in get_args(hint) or (hint,) if t is not type(None))
        parser.add_argument(*arg.flags, **kwargs)
