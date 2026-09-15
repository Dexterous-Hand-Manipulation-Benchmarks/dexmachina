import os  
import torch 
import numpy as np
from os.path import join
from dexmachina.envs.reward_utils import position_distance, rotation_distance, chamfer_distance, transform_contact


def get_reward_cfg(last_n_frame=-1):
    reward_cfg = {
        # Task Tracking Sharpness (from task_rew_betas=[10.0, 1.0, 5.0])
        "obj_pos_beta": 10.0,
        "obj_rot_beta": 1.0,
        "obj_arti_beta": 5.0,
        "obj_pos_weight": 2.0,
        "obj_rot_weight": 3.0,
        "obj_arti_weight": 5.0,

        "last_n_frame": last_n_frame,
        "multiply_task_rew": True,
        "multiply_all_rew": False,
        "task_rew_weight": 1.0,

        # Keypoint & Wrist Imitation
        "exp_kpt_first": True,
        "imi_rew_weight": 0.3,
        "imi_wrist_weight": 0.5,
        "imi_wrist_rot_beta": 3.0,
        "imi_wrist_pos_beta": 10.0,
        "imi_fingertip_beta": 20.0,

        # Behavioral Cloning Reward
        "bc_rew_weight": 0.3,
        "bc_beta": 500.0,

        # Contact Matching Reward
        "contact_rew_weight": 3.0,
        "contact_rew_function": "exp",
        "wrist_frame_contact": True,
        "contact_beta": 10.0,
        "contact_a": 100.0,
        "contact_b": 5.0,
        "multiply_frame_contact": True,
        "mask_zero_contact": True,
        "contact_phase_penalty": 0,

        "mask_well_track": False,
        "scale_well_track": 1.0,
        "force_penalty": 0.1,
        "action_penalty": 0.01,

        "objdex_baseline": False,
        "use_retarget_contact": True,
        "retarget_objframe": True,
    }
    return reward_cfg

class RewardModule:
    def __init__(self, reward_cfg, demo_data, retarget_data, device):
        print("\n" + "=" * 50)
        print("[DEBUG] ACTIVE REWARD CONFIGURATION:")
        print(f"  -> Force Penalty:   {reward_cfg.get('force_penalty')}")
        print(f"  -> Action Penalty:  {reward_cfg.get('action_penalty')}")
        print(f"  -> Imi Rew Weight:  {reward_cfg.get('imi_rew_weight')}")
        print(f"  -> Con Rew Weight:  {reward_cfg.get('contact_rew_weight')}")
        print("=" * 50 + "\n")
        self.cfg = reward_cfg
        self.last_n_frame = reward_cfg.get("last_n_frame", -1)
        self.exp_kpt_first = reward_cfg.get("exp_kpt_first", False)
        self.imi_rew_weight = reward_cfg["imi_rew_weight"]        
        self.imi_wrist_weight = reward_cfg.get("imi_wrist_weight", 0.0)
        self.task_rew_weight = reward_cfg["task_rew_weight"] 
        self.bc_rew_weight = reward_cfg.get("bc_rew_weight", 0.0)
        self.bc_wrist_weight = reward_cfg.get("bc_wrist_weight", 0.0)  
        self.demo_data = demo_data  
        self.multiply_task_rew = reward_cfg["multiply_task_rew"]
        self.multiply_all_rew = reward_cfg.get("multiply_all_rew", False)
        self.mask_well_track = reward_cfg.get("mask_well_track", False)
        self.scale_well_track = reward_cfg.get("scale_well_track", 1.0)
        self.obj_pos_beta = reward_cfg["obj_pos_beta"]
        self.obj_rot_beta = reward_cfg["obj_rot_beta"]
        self.obj_arti_beta = reward_cfg["obj_arti_beta"]
        if reward_cfg.get("objdex_baseline", False):
            print("Using the lambdas from ObjDex baseline")
            self.obj_pos_beta = 1.0
            self.obj_rot_beta = 20.0
            self.obj_arti_beta = 5.0
            self.imi_rew_weight = 0.0
            self.bc_rew_weight = 0.0
            self.task_rew_weight = 1.0
            self.contact_rew_weight = 0.0
        
        self.use_imi_rew = self.imi_rew_weight > 0.0
        self.use_bc_rew = self.bc_rew_weight > 0.0
        self.obj_pos_weight = reward_cfg["obj_pos_weight"] # not use because we multiply the rewards
        self.obj_rot_weight = reward_cfg["obj_rot_weight"]
        self.obj_arti_weight = reward_cfg["obj_arti_weight"]  

        self.wrist_frame_contact = reward_cfg.get("wrist_frame_contact", False)
        self.use_retarget_contact = reward_cfg.get("use_retarget_contact", False) # if True, the loaded contact is from retargeted hands
        self.retarget_objframe = reward_cfg.get("retarget_objframe", True) # if True, the contact is in the object frame, otherwise in the wrist frame
        self.contact_rew_weight = reward_cfg.get("contact_rew_weight", 0.0)
        self.contact_beta = reward_cfg.get("contact_beta", 10.0)
        self.contact_a = reward_cfg.get("contact_a", 100.0)
        self.contact_b = reward_cfg.get("contact_b", 5.0)
        self.sigmoid_offset = torch.tensor(-1 / (1.0 + np.exp(-self.contact_b)), device=device) + 1.0
        self.contact_phase_penalty = reward_cfg.get("contact_phase_penalty", 0.0)
        self.multiply_frame_contact = reward_cfg.get("multiply_frame_contact", True) 
        self.mask_zero_contact = reward_cfg.get("mask_zero_contact", True)
        self.contact_rew_function = reward_cfg.get("contact_rew_function", "exp")
        self.load_demo(demo_data, retarget_data, device)

    def load_demo(self, demo_data, retarget_data, device):
        self.demo_tensors = dict()
        demo_keys = ["obj_pos", "obj_quat", "obj_arti"]
        if self.contact_rew_weight > 0.0 and not self.use_retarget_contact:
            demo_keys += ["contact_links_left", "contact_links_right"]

        for key in demo_keys:
            assert key in demo_data, f"{key} not in demo_data"
            self.demo_tensors[key] = torch.tensor(
                demo_data[key], dtype=torch.float32, device=device
            )

        # Load retargeted contacts for robot hand matching
        if self.contact_rew_weight > 0.0 and self.use_retarget_contact:
            for side in ['left', 'right']:
                key = f"contact_links_{side}"
                if retarget_data and side in retarget_data and 'contact_links' in retarget_data[side]:
                    self.demo_tensors[key] = torch.tensor(
                        retarget_data[side]['contact_links'], dtype=torch.float32, device=device
                    )
                elif key in demo_data:
                    self.demo_tensors[key] = torch.tensor(
                        demo_data[key], dtype=torch.float32, device=device
                    )

        if self.use_imi_rew:
            for side in ['left', 'right']:
                key = f"kpts_{side}"
                if retarget_data and side in retarget_data and 'kpts_data' in retarget_data[side]:
                    self.demo_tensors[key] = torch.tensor(
                        retarget_data[side]['kpts_data']['kpt_pos'],
                        dtype=torch.float32, device=device
                    )
        if self.contact_rew_weight > 0.0 or (self.use_imi_rew and self.imi_wrist_weight > 0.0):
            for side in ['left', 'right']:
                key = f"wrist_pose_{side}"
                if retarget_data and side in retarget_data and 'wrist_pose' in retarget_data[side]:
                    if isinstance(retarget_data[side]['wrist_pose'], torch.Tensor):
                        self.demo_tensors[key] = retarget_data[side]['wrist_pose'].clone().to(device)
                    else:
                        self.demo_tensors[key] = torch.tensor(
                            retarget_data[side]['wrist_pose'], dtype=torch.float32, device=device
                        )

        assert all(
            [self.demo_tensors[key].shape[0] == self.demo_tensors["obj_pos"].shape[0] for key in
             self.demo_tensors.keys()]
        ), f"First dim size mismatch: {[self.demo_tensors[key].shape[0] for key in self.demo_tensors.keys()]}"
        self.demo_length = self.demo_tensors["obj_pos"].shape[0]
        print(f"Loaded demo data with length {self.demo_length}")
    
    def get_demo_length(self):  
        return self.demo_length
    
    def match_demo_state(self, demo_key, episode_length_buf):
        """ returns shape (num_envs, num_features) """
        assert demo_key in self.demo_tensors, f"Key {demo_key} not found in demo tensors"
        demo_t = torch.where(episode_length_buf >= self.demo_length, self.demo_length-1, episode_length_buf)
        # print(demo_t, episode_length_buf)
        return self.demo_tensors[demo_key][demo_t]
    
    def compute_task_reward(self, obj_pos, obj_quat, obj_arti, demo_pos, demo_quat, episode_length_buf):
        if obj_pos is None or obj_quat is None or obj_arti is None or self.task_rew_weight == 0.0:
            # dummy reward
            task_rew = torch.zeros(episode_length_buf.shape, device=episode_length_buf.device)
            return task_rew, dict(task_rew=task_rew)
        
        demo_arti = self.match_demo_state("obj_arti", episode_length_buf)
        pos_dist = position_distance(obj_pos, demo_pos)
        rot_dist = rotation_distance(obj_quat, demo_quat)
        # arti_dist = torch.mean((obj_arti - demo_arti)**2, dim=-1)

        if len(obj_arti.shape) > 1:
            obj_arti = obj_arti.flatten(start_dim=0)
            
        arti_dist = (obj_arti - demo_arti)**2 / 2.0
        
        # these should all be shape (num_envs, )!!
        obj_pos_rew = torch.exp(-self.obj_pos_beta * pos_dist)  
        obj_rot_rew = torch.exp(-self.obj_rot_beta * rot_dist)
        obj_arti_rew = torch.exp(-self.obj_arti_beta * arti_dist)
        assert obj_pos_rew.shape == obj_rot_rew.shape == obj_arti_rew.shape, "Shape mismatch"
        # if joint is closed, don't compute arti reward 
        if self.multiply_task_rew:
            task_rew = self.task_rew_weight * obj_pos_rew * obj_rot_rew * obj_arti_rew # scale is stil [0,1]
            # print(f"========= weighted task rew: {task_rew} | rew weight: {self.task_rew_weight * 10 } obj pos rew: {obj_pos_rew} | obj rot rew: {obj_rot_rew} | obj arti rew: {obj_arti_rew} =========")
        else:
            # obj_arti_rew = torch.where(demo_arti <= 0.0001, torch.zeros_like(obj_arti_rew), obj_arti_rew)
            task_rew = self.task_rew_weight * (
                self.obj_pos_weight * obj_pos_rew + 
                self.obj_rot_weight * obj_rot_rew + 
                self.obj_arti_weight * obj_arti_rew
                )
            
        # give a bonus if obj state is well tracked and articulation joint is open
        well_track = (pos_dist < 0.005) & (rot_dist < 0.1) & (arti_dist < 0.1)  & (demo_arti > 0.1)
        if self.scale_well_track > 1.0:
            task_rew = torch.where(well_track, task_rew * self.scale_well_track, task_rew)

        # Cross-implementation success metric, matching FastTD3's
        # envs/FastTD3/dexmachina_wrapper.py exactly so eval/success_* compares
        # like for like. NOTE: this is NOT the paper's ADD-based success (see
        # add_success_* / auc_add), and the "2cm" label refers only to the
        # position term -- rot/arti carry their own, looser thresholds.
        success_2cm = (pos_dist < 0.02) & (rot_dist < 0.20) & (arti_dist < 0.10)
        success_5cm = (pos_dist < 0.05) & (rot_dist < 0.35) & (arti_dist < 0.15)

        rew_dict = dict(
            pos_dist=pos_dist,
            rot_dist=rot_dist,
            arti_dist=arti_dist,
            success_2cm=success_2cm,
            success_5cm=success_5cm,
            obj_pos_rew=obj_pos_rew,
            obj_rot_rew=obj_rot_rew,
            obj_arti_rew=obj_arti_rew,
            task_rew=task_rew.clone(), # NOTE: otherwise it's the same tensor as the total reward
            well_track=well_track,
        )

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame, 
                torch.zeros(task_rew.shape, device=task_rew.device, dtype=torch.bool),
                torch.ones(task_rew.shape, device=task_rew.device, dtype=torch.bool)
                )
            task_rew[tomask] = 0.0
        return task_rew, rew_dict

    def compute_keypoint_dist(self, keypoint_pos, episode_length_buf, left_hand=True):
        demo_key = "kpts_left" if left_hand else "kpts_right"
        demo_kpts = self.match_demo_state(demo_key, episode_length_buf) 
        return position_distance(keypoint_pos, demo_kpts)
    
    def compute_wrist_reward(self, wrist_pose, episode_length_buf, side='left'):
        demo_wrist = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        wrist_rot_dist = rotation_distance(wrist_pose[:, 3:], demo_wrist[:, 3:])
        wrist_pos_dist = position_distance(wrist_pose[:, :3], demo_wrist[:, :3])
        rot_beta = self.cfg["imi_wrist_rot_beta"]
        pos_beta = self.cfg["imi_wrist_pos_beta"]
        wrist_rew = (torch.exp(-rot_beta * wrist_rot_dist) + torch.exp(-pos_beta * wrist_pos_dist)) / 2.0 
        return wrist_rew, wrist_pos_dist, wrist_rot_dist

    def compute_imitation_reward(
        self,
        wrist_pose_left,
        wrist_pose_right,
        kpts_left: torch.Tensor,
        kpts_right: torch.Tensor,
        episode_length_buf: torch.Tensor,
    ): 
        fingertip_dist_left = self.compute_keypoint_dist(kpts_left, episode_length_buf, left_hand=True)
        fingertip_dist_right = self.compute_keypoint_dist(kpts_right, episode_length_buf, left_hand=False)
        fingertip_dist = torch.mean( (fingertip_dist_left + fingertip_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
        beta = self.cfg["imi_fingertip_beta"]
        if self.exp_kpt_first:
            fingertip_rew_left = torch.exp(- beta * fingertip_dist_left)
            fingertip_rew_right = torch.exp(- beta * fingertip_dist_right)
            fingertip_rew = torch.mean( (fingertip_rew_left + fingertip_rew_right) / 2.0 , dim=-1) # (B, num_links) -> (B,) 
        else:  
            fingertip_rew = torch.exp(-self.cfg["imi_fingertip_beta"] * fingertip_dist) 
        # 
        if self.imi_wrist_weight > 0.0:
            # do a rotation + position distance for wrist pose
            wrist_rew_left, pos_dist_left, rot_dist_left = self.compute_wrist_reward(wrist_pose_left, episode_length_buf, side='left')
            wrist_rew_right, pos_dist_right, rot_dist_right = self.compute_wrist_reward(wrist_pose_right, episode_length_buf, side='right')
            wrist_rew = (wrist_rew_left + wrist_rew_right) / 2.0
            imi_rew = self.imi_wrist_weight * wrist_rew + (1.0 - self.imi_wrist_weight) * fingertip_rew

            wrist_dist = torch.mean( (pos_dist_left + pos_dist_right) / 2.0 , dim=-1) # (B, num_links) -> (B,)
            keypoint_dist = self.imi_wrist_weight * wrist_dist + (1.0 - self.imi_wrist_weight) * fingertip_dist
        else:
            imi_rew = fingertip_rew
            keypoint_dist = fingertip_dist
            
        imi_rew *= self.imi_rew_weight

        rew_dict = dict(
            kpts_dist_left=fingertip_dist_left,
            kpts_dist_right=fingertip_dist_right,  
            imi_rew=imi_rew, 
            keypoint_dist=keypoint_dist,
        )
        if self.imi_wrist_weight > 0.0:
            rew_dict["wrist_pdist_left"] = pos_dist_left
            rew_dict["wrist_pdist_right"] = pos_dist_right
            rew_dict["wrist_rdist_left"] = rot_dist_left
            rew_dict["wrist_rdist_right"] = rot_dist_right
            rew_dict["wrist_rew_left"] = wrist_rew_left  
            rew_dict["wrist_rew_right"] = wrist_rew_right 
            rew_dict["fingertip_rew"] = fingertip_rew
            if not self.exp_kpt_first:
                rew_dict["fingertip_dist"] = fingertip_dist
                

        if self.last_n_frame > 0:
            # mask out rewards that are not from the last n frames
            tomask = torch.where(
                episode_length_buf < self.demo_length - self.last_n_frame,
                torch.zeros(imi_rew.shape, device=imi_rew.device, dtype=torch.bool),
                torch.ones(imi_rew.shape, device=imi_rew.device, dtype=torch.bool)
                )
            imi_rew[tomask] = 0.0
        return imi_rew, rew_dict 

    def contact_dist_to_rew(self, dist, function='exp'):
        if function == 'exp':
            return torch.exp(-self.contact_beta * dist)
        elif function == 'sigmoid':
            axb = self.contact_a * dist + self.contact_b
            rew = 1.0 / (1.0 + torch.exp(-axb)) + self.sigmoid_offset.to(dist.device) 
            return rew
        else:
            raise ValueError(f"Unknown contact reward function: {function}")

    def compute_hand_contact_reward(
        self,
        contact_link_pos, # shape (N, num_obj_links * num_hand_links, 4)
        contact_link_valid, # shape (N, num_obj_links * num_hand_links, 1)
        wrist_pose, # shape (N, 7)
        obj_pose, # shape (N, 7)
        episode_length_buf,
        demo_obj_pose, # shape (N, 7)
        side='left',
    ):
        demo_wrist_pose = self.match_demo_state(f"wrist_pose_{side}", episode_length_buf)
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf) 
        # N, num_links * 2, 4 (last dim is contact pair ID) -> NOTE in ARCTIC, part_id=2 is 'bottom' link, part_id=1 is 'top' 
        demo_positions = demo_contacts[:, :, :3]
        demo_valid_contact = demo_contacts[:, :, -1] > 0.0 # (part id is <= 0 if no contact)
        chamfer_dists = dict()
        contact_rewards = dict()

        positions = contact_link_pos[:, :, :3]
        valid_mask = contact_link_pos[:, :, -1] > 0.0 
        for part_id in [1, 2]:
            # consider contact invalid if part_id is different
            demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
            part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                 
            for frame in ['obj', 'wrist']:
                if not self.wrist_frame_contact and frame == 'wrist':
                    continue
                demo_pose = demo_obj_pose if frame == 'obj' else demo_wrist_pose
                demo_in_frame = transform_contact(
                    demo_positions, demo_pose
                )
                pose = obj_pose if frame == 'obj' else wrist_pose
                in_frame = transform_contact(
                    positions, pose
                )
                dist = chamfer_distance(
                    in_frame, demo_in_frame, part_valid, demo_part_valid
                )
                chamfer_dists[f"CD_{side}_p{part_id}_frame_{frame}"] = dist
                rew = self.contact_dist_to_rew(dist, self.contact_rew_function)
                both_zero = (part_valid.sum(dim=-1) == 0) & (demo_part_valid.sum(dim=-1) == 0)
                if self.mask_zero_contact:
                    rew = torch.where(both_zero, torch.zeros_like(rew), rew)
                else:
                    rew = torch.where(both_zero, torch.ones_like(rew), rew)
                contact_rewards[f"conrew_{side}_p{part_id}_frame_{frame}"] = rew
            
        # per-part contact should be added, because sometimes full contact coverage is not feasible 
        if self.multiply_frame_contact:
            # multiply the rew in obj & wrist frame such that one does not dominate the other
            contact_rews = []
            for part_id in [1, 2]:
                obj_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_obj"]
                if not self.wrist_frame_contact:
                    wrist_rew = 1.0
                else:
                    wrist_rew = contact_rewards[f"conrew_{side}_p{part_id}_frame_wrist"] 
                multi_rew = obj_rew * wrist_rew
                contact_rewards[f"mul_conrew_{side}_p{part_id}"] = multi_rew
                contact_rews.append(multi_rew)
            contact_rew = torch.stack(contact_rews, dim=-1).mean(dim=-1)

        else: # sum & average all contact rewards
            contact_rew = sum(contact_rewards.values()) / len(contact_rewards)

        contact_dict = {**chamfer_dists, **contact_rewards, f"contact_rew_{side}": contact_rew}
        # add penalty for scenarios where demo has 0 contact but env has contact
        if self.contact_phase_penalty != 0.0: 
            mismatch = torch.zeros_like(contact_rew).bool()
            num_mismatch = torch.zeros_like(contact_rew)
            for part_id in [1, 2]:
                demo_part_valid = (demo_contacts[:, :, -1] == part_id) & demo_valid_contact
                part_valid = (contact_link_pos[:, :, -1] == part_id ) & valid_mask
                part_mismatch = (demo_part_valid.sum(dim=-1) == 0) & (part_valid.sum(dim=-1) > 0)
                mismatch = mismatch | part_mismatch
                num_mismatch += part_mismatch.float()
            contact_rew[mismatch] += self.contact_phase_penalty
            contact_dict['num_mismatch'] = num_mismatch

        return contact_rew, contact_dict

    def compute_matched_contact_per_hand(
            self,
            contact_link_pos,  # shape (N, num_obj_links, num_hand_links, 4)
            contact_link_valid,  # shape (N, num_obj_links, num_hand_links, 1)
            episode_length_buf,
            obj_pose,  # shape (N, 7)
            demo_obj_pose,
            side='left',
            max_distance=1.0,
            return_valid=False,
    ):
        demo_contacts = self.match_demo_state(f"contact_links_{side}", episode_length_buf)

        # --- VIEW FIX: Restore 4D representation (N, 2, num_links, 4) ---
        if demo_contacts.dim() == 3:
            num_obj_links = contact_link_pos.shape[1]
            num_hand_links = contact_link_pos.shape[2]
            demo_contacts = demo_contacts.view(-1, num_obj_links, num_hand_links, 4)

        demo_contacts = demo_contacts.clone()[:, [1, 0], :, :]
        assert demo_contacts.shape[1] == contact_link_pos.shape[
            1], f"Shape mismatch: {demo_contacts.shape} vs {contact_link_pos.shape}"
        assert demo_contacts.shape[2] == contact_link_pos.shape[
            2], f"Shape mismatch: {contact_link_pos.shape} vs {demo_contacts.shape}"
        demo_valids = demo_contacts[:, :, :, -1] > 0.0

        if self.retarget_objframe:
            bsize, nparts, nlinks = demo_contacts.shape[:3]
            demo_pos_global = demo_contacts[:, :, :, :3].reshape(
                bsize, nparts * nlinks, 3
            )
            demo_pos = transform_contact(demo_pos_global, demo_obj_pose)
            demo_pos = demo_pos.reshape(bsize, nparts, nlinks, 3)

            policy_pos_global = contact_link_pos[:, :, :, :3].reshape(
                bsize, nparts * nlinks, 3
            )
            policy_pos = transform_contact(policy_pos_global, obj_pose)
            policy_pos = policy_pos.reshape(bsize, nparts, nlinks, 3)
        else:
            demo_pos = demo_contacts[:, :, :, :3]
            policy_pos = contact_link_pos[:, :, :, :3]

        both_invalid_mask = torch.logical_not(demo_valids) & torch.logical_not(contact_link_valid)
        one_valid_mask = torch.logical_xor(demo_valids, contact_link_valid)
        dist = position_distance(demo_pos, policy_pos)

        if self.mask_zero_contact:
            dist = torch.where(both_invalid_mask, torch.ones_like(dist) * max_distance, dist)
        else:
            dist = torch.where(both_invalid_mask, torch.zeros_like(dist), dist)

        dist = torch.where(one_valid_mask, torch.ones_like(dist) * max_distance, dist)
        if return_valid:
            return dist, demo_valids
        return dist

    def compute_matched_contact_reward(
            self,
            contacts_link_left,
            contacts_link_valid_left,
            contacts_link_right,
            contacts_link_valid_right,
            obj_pose,
            demo_obj_pose,
            episode_length_buf,
    ):
        rews = dict()
        # Initialize as a CUDA tensor of shape (num_envs,) instead of a Python float
        contact_rew = torch.zeros(
            episode_length_buf.shape[0],
            device=episode_length_buf.device,
            dtype=torch.float32
        )

        num_terms = 0
        # --- Diagnostics for why con_rew is small. compute_matched_contact_per_hand
        # forces dist to max_distance (1.0) wherever NEITHER the demo nor the policy
        # is touching, so a clip whose demo has contact on only a few (frame, link)
        # pairs caps con_rew far below contact_rew_weight no matter how good the
        # policy is. These three per-env scalars separate the two causes:
        #   con_active_frac  fraction of (part, link) pairs the demo wants in contact
        #   con_dist_active  mean matched distance over just those pairs (metres)
        #   con_rew_ceiling  con_rew a policy matching every demo contact exactly
        #                    would score -- the real target for the curriculum gate
        n_envs = episode_length_buf.shape[0]
        dev = episode_length_buf.device
        valid_sum = torch.zeros(n_envs, device=dev)
        valid_cnt = torch.zeros(n_envs, device=dev)
        dist_sum = torch.zeros(n_envs, device=dev)
        ceiling = torch.zeros(n_envs, device=dev)

        for side, contacts, valids in zip(
                ['left', 'right'],
                [contacts_link_left, contacts_link_right],
                [contacts_link_valid_left, contacts_link_valid_right]
        ):
            if contacts is None or valids is None:
                continue

            part_dist, demo_valid = self.compute_matched_contact_per_hand(
                contacts, valids, episode_length_buf,
                obj_pose, demo_obj_pose, side=side, return_valid=True
            )
            for i, part in enumerate(['bottom', 'top']):
                con_dist = part_dist[:, i]
                if self.exp_kpt_first:
                    con_rew = self.contact_dist_to_rew(con_dist, self.contact_rew_function).mean(dim=-1)
                else:
                    con_rew = self.contact_dist_to_rew(con_dist.mean(dim=-1), self.contact_rew_function)
                rews[f"conrew_{side}_{part}"] = con_rew
                rews[f"matched_condist_{side}_{part}"] = con_dist
                contact_rew = contact_rew + con_rew
                num_terms += 1

                v = demo_valid[:, i].to(con_dist.dtype)
                valid_sum = valid_sum + v.sum(dim=-1)
                valid_cnt = valid_cnt + v.shape[-1]
                dist_sum = dist_sum + (con_dist * v).sum(dim=-1)
                # perfect play: distance 0 where the demo wants contact, and the
                # same max_distance elsewhere. Mirrors the exp_kpt_first branch.
                best_dist = torch.where(
                    v > 0, torch.zeros_like(con_dist), torch.ones_like(con_dist)
                )
                ceiling = ceiling + self.contact_dist_to_rew(
                    best_dist, self.contact_rew_function
                ).mean(dim=-1)

        # Preserves the exact 4.0 scaling factor from the Dexmachina codebase
        contact_rew = (contact_rew / 4.0) * self.contact_rew_weight
        rews['con_rew'] = contact_rew
        if num_terms > 0:
            rews['con_active_frac'] = valid_sum / valid_cnt.clamp(min=1.0)
            rews['con_dist_active'] = dist_sum / valid_sum.clamp(min=1.0)
            rews['con_rew_ceiling'] = (ceiling / num_terms) * self.contact_rew_weight
        return contact_rew, rews

    def compute_contact_reward(
        self,
        obj_pose,
        wrist_pose_left,
        wrist_pose_right,
        contacts_link_left,
        contacts_link_valid_left,  # shape (N, num_obj_links, num_hand_links)
        contacts_link_right,
        contacts_link_valid_right,
        episode_length_buf,
    ):
        demo_obj_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_obj_quat = self.match_demo_state("obj_quat", episode_length_buf)
        demo_obj_pose = torch.cat([demo_obj_pos, demo_obj_quat], dim=1)

        contact_rew = 0.0
        contact_dict = {}
        hand_count = 0

        if contacts_link_left is not None:
            contact_rew_left, contact_dict_left = self.compute_hand_contact_reward(
                contacts_link_left, contacts_link_valid_left,
                wrist_pose_left, obj_pose, episode_length_buf, demo_obj_pose, side='left'
            )
            contact_rew += contact_rew_left
            contact_dict.update(contact_dict_left)
            hand_count += 1

        if contacts_link_right is not None:
            contact_rew_right, contact_dict_right = self.compute_hand_contact_reward(
                contacts_link_right, contacts_link_valid_right,
                wrist_pose_right, obj_pose, episode_length_buf, demo_obj_pose, side='right'
            )
            contact_rew += contact_rew_right
            contact_dict.update(contact_dict_right)
            hand_count += 1

        if hand_count > 0:
            contact_rew = contact_rew / hand_count
        else:
            # Fallback zero tensor if no hands are found
            contact_rew = torch.zeros_like(demo_obj_pos[:, 0])

        contact_rew *= self.contact_rew_weight
        contact_dict['con_rew'] = contact_rew
        return contact_rew, contact_dict

    def reshape_contact_with_label(self, contact_link_pos, contact_link_valid):
        """ 
        Reshape the environment contact to better match with demos:
        In: 
            contact_link_pos_left: (N, 2, num_hand_links, 3), 2 is each object part
            contact_link_valid_left: (N, 2, num_hand_links)
        Out: 
            reshaped_contact_link_pos_left: (N, 2*num_hand_links, 4): 4 is (x, y, z, part_id), part_id is 1 or 2 or 0 if no valid
            reshaped contact_link_valid: (N, 2, num_hand_links) -> (N, 2*num_hand_links)
        """
        # first, add a part_id dim so (N, 2, num_links, 4) and last dim is part_id label
        labeled = torch.cat(
            [contact_link_pos, torch.zeros_like(contact_link_pos)[:, :, :, 0:1]], dim=-1
        ) # (N, 2, num_links, 4) 
        # fill in the labels 1 or 2 if valid contact
        valid_reshaped = contact_link_valid[..., None] # (N, 2, num_links, 1)
        valid = torch.cat(
            [valid_reshaped, torch.zeros_like(valid_reshaped)], dim=-1
        ) # -> (N, 2, num_links, 2)
        for part_idx in [0, 1]: # bottom, top (sice obj urdf is flipped for parsing error)
            label_id = part_idx + 1 
            # Need flipping, because in demos, id=2 is bottom
            label_id = 1 if label_id == 2 else 2 
            labeled[:, part_idx, :, -1] = torch.where(
                contact_link_valid[:, part_idx, :],  label_id, 0
            ) 
            valid[:, part_idx, :, 1] = torch.where(
                contact_link_valid[:, part_idx, :], label_id, 0
            )

        # reshape to (N, num_links*2, 4)
        contact_link_reshaped = labeled.view(
            labeled.shape[0], -1, labeled.shape[-1]
        )
        contact_valid_reshaped = valid.view(
            valid.shape[0], -1, valid.shape[-1]
        )
        
        return contact_link_reshaped, contact_valid_reshaped

    def compute_reward(
        self, 
        actions,
        bc_dist, # (num_envs, left+right hand num of joints) 
        obj_pos,
        obj_quat,
        obj_arti,
        kpts_left,
        kpts_right,
        contact_link_pos_left, 
        contact_link_valid_left,  # shape (N, num_obj_links, num_hand_links)
        contact_link_pos_right,
        contact_link_valid_right,
        wrist_pose_left,
        wrist_pose_right,
        contact_forces, # shape (N, num_obj_links, num_hand_links, 3)
        episode_length_buf,
    ):  
        demo_pos = self.match_demo_state("obj_pos", episode_length_buf)
        demo_quat = self.match_demo_state("obj_quat", episode_length_buf) 
        rew, rew_dict = self.compute_task_reward(
            obj_pos, obj_quat, obj_arti, 
            demo_pos, demo_quat,
            episode_length_buf
        )
        if self.bc_rew_weight > 0.0:
            bc_rew = self.bc_rew_weight * torch.exp(-self.cfg["bc_beta"] * bc_dist)
            bc_rew = torch.mean(bc_rew, dim=-1)
            rew_dict["bc_dist"] = bc_dist.mean(dim=-1)
            rew_dict["bc_rew"] = bc_rew
            # rew += bc_rew

        if self.use_imi_rew: 
            imi_rew, imi_rew_dict = self.compute_imitation_reward(
                wrist_pose_left, wrist_pose_right,
                kpts_left, kpts_right, episode_length_buf
            )
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the imitation reward
                imi_rew = torch.where(rew_dict["well_track"], torch.zeros_like(imi_rew), imi_rew) 
            # rew += imi_rew
            rew_dict.update(imi_rew_dict)

        if self.contact_rew_weight > 0.0:
            obj_pose = torch.cat([obj_pos, obj_quat], dim=1)
            demo_pose = torch.cat([demo_pos, demo_quat], dim=1)
            if self.use_retarget_contact:
                contact_rew, contact_dict = self.compute_matched_contact_reward(
                    contact_link_pos_left, contact_link_valid_left,
                    contact_link_pos_right, contact_link_valid_right,
                    obj_pose, demo_pose, episode_length_buf
                )
            else:
                # --- NONE-SAFE RESHAPING ---
                if contact_link_pos_left is not None:
                    left_reshaped, left_valid = self.reshape_contact_with_label(
                        contact_link_pos_left, contact_link_valid_left
                    )
                else:
                    left_reshaped, left_valid = None, None

                if contact_link_pos_right is not None:
                    right_reshaped, right_valid = self.reshape_contact_with_label(
                        contact_link_pos_right, contact_link_valid_right
                    )
                else:
                    right_reshaped, right_valid = None, None
                # ---------------------------

                contact_rew, contact_dict = self.compute_contact_reward(
                    obj_pose,
                    wrist_pose_left,
                    wrist_pose_right,
                    left_reshaped, left_valid,
                    right_reshaped, right_valid,
                    episode_length_buf
                )
            if "well_track" in rew_dict and self.mask_well_track:
                # use well_track to mask out the contact reward
                contact_rew = torch.where(rew_dict["well_track"], torch.zeros_like(contact_rew), contact_rew)

                # rew += contact_rew
            rew_dict.update(contact_dict)
        
        if self.multiply_all_rew:
            aux_rew = 1.0
            if self.use_imi_rew:
                aux_rew *= imi_rew
            if self.contact_rew_weight > 0.0:
                aux_rew *= contact_rew
            if self.bc_rew_weight > 0.0:
                aux_rew *= bc_rew
            rew += aux_rew * 0.5
        else:
            if self.use_imi_rew:
                rew += imi_rew
            if self.contact_rew_weight > 0.0:
                rew += contact_rew
            if self.bc_rew_weight > 0.0:
                rew += bc_rew
        
        
        if self.cfg["force_penalty"] > 0.0 and contact_forces is not None: # shape (B, 2, num_links*2, 3)
            # Correct for NaNs.
            contact_forces = torch.nan_to_num(contact_forces, nan=0.0, posinf=500.0, neginf=-500.0)

            force_norm = torch.norm(contact_forces, dim=-1).flatten(start_dim=1) # shape (B, 2*num_links*2) -> this goes to up to 5k
            high_force = torch.where(force_norm > 500.0, force_norm - 500.0, torch.zeros_like(force_norm))
            high_force = torch.mean(high_force, dim=-1) # shape (B, )
            force_penalty = self.cfg["force_penalty"] * high_force 
            rew -= force_penalty
            rew_dict["force_penalty"] = force_penalty
        
        # rew *= 0.02 
        # penalize action to stay close to 0 
        if self.cfg["action_penalty"] > 0.0:
            action_penalty = torch.mean(actions**2, dim=-1) * self.cfg["action_penalty"]
            rew -= action_penalty
            rew_dict["action_penalty"] = action_penalty  
        return rew, rew_dict
 
    def get_reward_keys(self):
        keys = []
        if self.task_rew_weight > 0:
            keys.append('task')
        if self.imi_rew_weight > 0:
            keys.append('imi')
        if self.contact_rew_weight > 0:
            keys.append('con')
        if self.bc_rew_weight > 0:
            keys.append('bc')
        return keys 