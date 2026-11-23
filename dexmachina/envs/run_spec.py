"""The env a run trained on, saved next to its checkpoints so that evaluation rebuilds the same task.

`env_spec.json` holds the `EnvSpec`, the reward overrides that have no `EnvSpec` field, and the dexmachina commit the
run was started from, so every method can save and load the same file.
"""

import hashlib
import json
import os
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
DIFF_KEY = "dexmachina_diff_sha256"
RUN_SPEC_KEYS = frozenset({SPEC_KEY, REWARD_OVERRIDES_KEY, SHA_KEY, DIRTY_KEY, DIFF_KEY})
DEXMACHINA_ROOT = Path(__file__).resolve().parents[2]
HASH_CHUNK_BYTES = 1 << 20


class DexmachinaState(NamedTuple):
    """The dexmachina checkout's commit, whether it had uncommitted changes, and a hash of them.

    `diff_sha256` is None for a clean checkout, and also for a dirty one whose changes are unknown (a run spec written
    after the fact). Equal hashes that are not None mean the same uncommitted changes on the same commit.
    """

    sha: str
    dirty: bool
    diff_sha256: str | None


@dataclass(frozen=True)
class RunSpec:
    spec: EnvSpec
    reward_overrides: Mapping[str, Any]
    dexmachina_sha: str
    dexmachina_dirty: bool
    dexmachina_diff_sha256: str | None

    @property
    def dexmachina_state(self) -> DexmachinaState:
        return DexmachinaState(self.dexmachina_sha, self.dexmachina_dirty, self.dexmachina_diff_sha256)


def _git_output(*args: str) -> bytes:
    """The raw stdout of a git command in the dexmachina checkout, unstripped so that hashes see every byte."""
    try:
        completed = subprocess.run(["git", "-C", str(DEXMACHINA_ROOT), *args], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not read the dexmachina git state with `git {' '.join(args)}` in {DEXMACHINA_ROOT}. A saved run "
            "spec records the dexmachina commit so that evaluation can tell when the code changed. Run from a git "
            "checkout of dexmachina (an editable install), with git installed."
        ) from error
    return completed.stdout


def _git(*args: str) -> str:
    return _git_output(*args).decode().strip()


def _untracked_entry_sha256(name: bytes) -> bytes:
    """A symlink is hashed by its target, a regular file by its bytes, read in chunks."""
    path = DEXMACHINA_ROOT / os.fsdecode(name)
    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(os.fsencode(os.readlink(path)))
    elif path.is_file():
        with path.open("rb") as file:
            while chunk := file.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    else:
        raise RuntimeError(
            f"{path} is untracked and neither a file nor a symlink (git lists a nested repository this way), so the "
            "hash of dexmachina's uncommitted changes cannot cover it. Remove it, commit it, or add it to "
            "dexmachina's .gitignore."
        )
    return digest.digest()


def _uncommitted_changes_sha256() -> str:
    """Hashes the diff of tracked files against HEAD and the path and content of every untracked, not ignored file.

    Each part enters the outer hash as its own sha256, so the boundaries between parts cannot shift. The diff options
    keep the user's git config (renames, external diff tools, text conversion) out of the hash.
    """
    diff = _git_output("diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv", "--no-renames")
    digest = hashlib.sha256(hashlib.sha256(diff).digest())
    untracked = _git_output("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    for name in sorted(name for name in untracked if name):
        digest.update(hashlib.sha256(name).digest() + _untracked_entry_sha256(name))
    return digest.hexdigest()


def current_dexmachina_state() -> DexmachinaState:
    sha, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain"))
    return DexmachinaState(sha, dirty, _uncommitted_changes_sha256() if dirty else None)


def save_run_spec(directory: Path, spec: EnvSpec, reward_overrides: Mapping[str, Any]) -> Path:
    """Writes `env_spec.json` into the directory and returns its path; an existing file is never replaced."""
    path = directory / RUN_SPEC_FILENAME
    if path.exists():
        raise FileExistsError(
            f"{path} exists already. Replacing it would change the env that the checkpoints next to it are "
            "evaluated on. Save into a new directory, or delete the file if it is stale."
        )
    state = current_dexmachina_state()
    if state.dirty:
        print(
            f"[WARNING] dexmachina at {DEXMACHINA_ROOT} has uncommitted changes (sha256 {state.diff_sha256}), so its "
            f"commit {state.sha[:12]} does not say which env code this run trains on, and its results cannot be "
            "reproduced from the commit. Commit or stash the changes in dexmachina before starting runs whose results "
            "count."
        )
    content = {
        SPEC_KEY: json.loads(spec.to_json()),
        REWARD_OVERRIDES_KEY: dict(reward_overrides),
        SHA_KEY: state.sha,
        DIRTY_KEY: state.dirty,
        DIFF_KEY: state.diff_sha256,
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
    sha, dirty, diff = content[SHA_KEY], content[DIRTY_KEY], content[DIFF_KEY]
    if not isinstance(sha, str) or not isinstance(dirty, bool) or not isinstance(diff, str | None):
        raise ValueError(
            f"{path} has a {SHA_KEY} of type {type(sha).__name__}, a {DIRTY_KEY} of type {type(dirty).__name__} and "
            f"a {DIFF_KEY} of type {type(diff).__name__}; they must be a string, a boolean and a string or null. The "
            "file was edited or written by something else, so the dexmachina commit it names cannot be trusted. "
            f"Rewrite {SHA_KEY} as the commit hash string, {DIRTY_KEY} as true or false and {DIFF_KEY} as the hash "
            "string or null."
        )
    if diff is not None and not dirty:
        raise ValueError(
            f"{path} has a {DIFF_KEY} but {DIRTY_KEY} false; only a checkout with uncommitted changes has a hash of "
            f"them. The file was edited by hand. Set {DIRTY_KEY} to true, or {DIFF_KEY} to null if the checkout was "
            "clean."
        )
    return RunSpec(
        spec=EnvSpec.from_json(json.dumps(content[SPEC_KEY])),
        reward_overrides=content[REWARD_OVERRIDES_KEY],
        dexmachina_sha=sha,
        dexmachina_dirty=dirty,
        dexmachina_diff_sha256=diff,
    )
