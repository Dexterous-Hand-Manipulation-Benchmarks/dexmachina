"""The env a run trained on, saved next to its checkpoints so that evaluation rebuilds the same task.

`env_spec.json` holds the `EnvSpec`, the reward overrides that have no `EnvSpec` field, and the dexmachina commit the
run was started from, so every method can save and load the same file.
"""

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from dexmachina.envs.spec import EnvSpec

RUN_SPEC_FILENAME = "env_spec.json"
SPEC_KEY = "spec"
REWARD_OVERRIDES_KEY = "reward_overrides"
SHA_KEY = "dexmachina_sha"
DIRTY_KEY = "dexmachina_dirty"
RUN_SPEC_KEYS = frozenset({SPEC_KEY, REWARD_OVERRIDES_KEY, SHA_KEY, DIRTY_KEY})
DEXMACHINA_ROOT = Path(__file__).resolve().parents[2]


class DexmachinaState(NamedTuple):
    """The dexmachina checkout's commit and whether it had uncommitted changes."""

    sha: str
    dirty: bool


@dataclass(frozen=True)
class RunSpec:
    spec: EnvSpec
    reward_overrides: Mapping[str, Any]
    dexmachina_sha: str
    dexmachina_dirty: bool


def _git(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(DEXMACHINA_ROOT), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not read the dexmachina git state with `git {' '.join(args)}` in {DEXMACHINA_ROOT}. A saved run "
            "spec records the dexmachina commit so that evaluation can tell when the code changed. Run from a git "
            "checkout of dexmachina (an editable install), with git installed."
        ) from error
    return completed.stdout.strip()


def current_dexmachina_state() -> DexmachinaState:
    return DexmachinaState(sha=_git("rev-parse", "HEAD"), dirty=bool(_git("status", "--porcelain")))


def save_run_spec(directory: Path, spec: EnvSpec, reward_overrides: Mapping[str, Any]) -> Path:
    """Writes `env_spec.json` into the directory and returns its path; an existing file is never replaced."""
    path = directory / RUN_SPEC_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{path} exists already. Replacing it would change the env that the checkpoints next to it are "
            "evaluated on. Save into a new directory, or delete the file if it is stale."
        )
    state = current_dexmachina_state()
    content = {
        SPEC_KEY: json.loads(spec.to_json()),
        REWARD_OVERRIDES_KEY: dict(reward_overrides),
        SHA_KEY: state.sha,
        DIRTY_KEY: state.dirty,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, sort_keys=True, indent=2) + "\n")
    return path


def load_run_spec(directory: Path) -> RunSpec:
    """Reads the `env_spec.json` that `save_run_spec` wrote into the directory."""
    path = directory / RUN_SPEC_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Runs write it into their checkpoint directory when they start, so this run "
            "predates it or the directory is wrong. Point at the run's checkpoint directory, or write the file for "
            "an older run from its logged settings."
        )
    content: dict[str, Any] = json.loads(path.read_text())
    missing, unknown = sorted(RUN_SPEC_KEYS - content.keys()), sorted(content.keys() - RUN_SPEC_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path} does not have the keys of a run spec: missing {missing}, unknown {unknown}. It was not written "
            f"by save_run_spec or by another version of it. Expected exactly {sorted(RUN_SPEC_KEYS)}; rewrite the "
            "file with these keys."
        )
    sha, dirty = content[SHA_KEY], content[DIRTY_KEY]
    if not isinstance(sha, str) or not isinstance(dirty, bool):
        raise ValueError(
            f"{path} has a {SHA_KEY} of type {type(sha).__name__} and a {DIRTY_KEY} of type {type(dirty).__name__}; "
            "they must be a string and a boolean. The file was edited or written by something else, so the dexmachina "
            f"commit it names cannot be trusted. Rewrite {SHA_KEY} as the commit hash string and {DIRTY_KEY} as "
            "true or false."
        )
    return RunSpec(
        spec=EnvSpec.from_json(json.dumps(content[SPEC_KEY])),
        reward_overrides=content[REWARD_OVERRIDES_KEY],
        dexmachina_sha=sha,
        dexmachina_dirty=dirty,
    )
