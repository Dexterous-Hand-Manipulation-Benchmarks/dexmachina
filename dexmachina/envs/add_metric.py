"""Online ADD (average distance) metric for articulated object tracking.

This mirrors the definition in ``dexmachina/eval/compute_add.py`` so that the
curves logged during training are directly comparable with the offline eval
numbers reported in the paper:

    for each object part (top / bottom), sample a fixed set of mesh vertices,
    transform them by the part's demo pose and by its actual pose, and take the
    mean per-vertex L2 distance.

The offline script gets the demo part poses by replaying the demo states into a
separate collision-free scene. Here they are precomputed once at startup, while
``ArticulatedObject.post_scene_build_setup`` is already driving the object
through every demo frame, and are then indexed per env by ``episode_length_buf``
-- which gives the same frame alignment as the offline replay.

IMPORTANT: ADD measures whatever physics the object is actually under. If the
assistive object controller (the curriculum's kp/kv/force_range) is non-zero,
part of the tracking is being done by that controller and not by the policy, so
the number is *not* comparable to an unassisted eval. Evaluation therefore runs on
a separate env whose gains are zeroed permanently (make_dexmachina_env
eval_mode=True).
"""

import numpy as np
import torch

from dexmachina.envs.math_utils import matrix_from_quat

# Same thresholds the offline script uses: np.arange(0.01, 0.1, 0.01)
ADD_AUC_THRESHOLDS = tuple(float(t) for t in np.arange(0.01, 0.1, 0.01))


def transform_vertices(verts, pose):
    """verts: (V, 3) in part-local frame, pose: (N, 7) as [pos, quat] -> (N, V, 3)."""
    matrices = matrix_from_quat(pose[:, 3:7])          # (N, 3, 3)
    offsets = pose[:, :3].unsqueeze(1)                 # (N, 1, 3)
    # expand instead of relying on einsum broadcasting, which torch does not do
    batched = verts.unsqueeze(0).expand(pose.shape[0], -1, -1)
    return torch.einsum("nij,nkj->nki", matrices, batched) + offsets


class ADDMetric:
    """Per-env ADD against the demo trajectory, computed from live part poses."""

    def __init__(self, obj, num_samples: int = 500, seed: int = 42, parts=("top", "bottom")):
        self.obj = obj
        self.parts = tuple(parts)
        self.num_samples = num_samples
        # deterministic vertex sampling -- must match compute_add.py (seed=42)
        self.verts = {
            part: obj.sample_mesh_vertices(num_samples, part=part, seed=seed).to(obj.device)
            for part in self.parts
        }
        self.demo_part_pose = {part: [] for part in self.parts}
        self.num_frames = 0
        self._finalized = False
        self._warned_bad_quat = False

    # -- startup precompute -------------------------------------------------
    def record_demo_frame(self):
        """Snapshot the current part poses as demo frame ``t``.

        Called once per demo frame during the startup sweep, after the object has
        been set to that frame's demo state. Env 0 is used because every env was
        set to the same demo state. Reads the links directly off the entity
        rather than going through ``update_value_buffers``, which also touches
        ``state_diff`` / ``episode_length_buf`` that are not valid yet at setup.
        """
        assert not self._finalized, "record_demo_frame after finalize()"
        link_pos = self.obj.entity.get_links_pos()    # (num_envs, n_links, 3)
        link_quat = self.obj.entity.get_links_quat()  # (num_envs, n_links, 4)
        for part in self.parts:
            idx = self.obj.link_names.index(part)
            pose = torch.cat([link_pos[0, idx], link_quat[0, idx]], dim=-1)
            self.demo_part_pose[part].append(pose.clone().to(self.obj.device))

    def finalize(self):
        for part in self.parts:
            self.demo_part_pose[part] = torch.stack(self.demo_part_pose[part], dim=0)
        self.num_frames = self.demo_part_pose[self.parts[0]].shape[0]
        self._finalized = True

    @property
    def ready(self) -> bool:
        return self._finalized and self.num_frames > 0

    # -- per-step -----------------------------------------------------------
    def _safe_pose(self, pose):
        """Guard against all-zero quaternions.

        BaseEnv.reset_idx -> ArticulatedObject.reset_idx zeroes part_quat, and
        matrix_from_quat divides by the quaternion's squared norm, so a zero quat
        yields inf -> NaN, which poisons the mean over every env. compute() is
        called before any reset so this should not fire; warn loudly if it does
        rather than silently logging a NaN curve.
        """
        norm_sq = (pose[:, 3:7] * pose[:, 3:7]).sum(-1)
        bad = norm_sq < 1e-8
        if bool(bad.any()):
            if not self._warned_bad_quat:
                self._warned_bad_quat = True
                print(
                    f"[ADD] WARNING: {int(bad.sum())} env(s) have a degenerate object "
                    "quaternion; substituting identity. ADD is being computed after a "
                    "reset zeroed part_quat -- check the call ordering in BaseEnv.step."
                )
            pose = pose.clone()
            pose[bad, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pose.device)
        return pose

    def compute(self, episode_length_buf) -> dict:
        """Return {'add_top', 'add_bottom', 'add'}, each shape (num_envs,) in metres."""
        idx = episode_length_buf.clamp(max=self.num_frames - 1).long()
        out, total = {}, None
        for part in self.parts:
            verts = self.verts[part]
            cur = transform_vertices(verts, self._safe_pose(self.obj.get_part_pose(part)))
            demo = transform_vertices(verts, self.demo_part_pose[part][idx])
            dist = torch.norm(cur - demo, dim=-1).mean(dim=-1)
            out[f"add_{part}"] = dist
            total = dist if total is None else total + dist
        out["add"] = total / len(self.parts)
        return out


def compute_auc(add_values, thresholds=ADD_AUC_THRESHOLDS) -> float:
    """AUC of the accuracy-vs-threshold curve, matching compute_add.py.

    ``add_values`` is a flat array of per-(env, frame) ADD values. The x axis is
    normalised to [0, 1] over the threshold list, so the result is in [0, 1].
    """
    add_values = np.asarray(add_values)
    accuracies = np.array([float(np.mean(add_values < t)) for t in thresholds])
    x_values = np.linspace(0, 1, len(thresholds))
    return float(np.trapz(accuracies, x=x_values))
