import os  
import warnings
from bisect import bisect_right
import torch 
import numpy as np
import genesis as gs
from collections import deque

def get_curriculum_cfg(kwargs=dict()):
    curr_cfg = {
        "kp_init": 1000.0,
        "kv_init": 10.0,
        "force_range_init": 50.0, 
        "wait_epochs": 2000,
        "decay_rew": False,
        "schedule": "exp", # or exp or uniform 
        "deque_len": 100, # length of reward deques 
        # kept for compatibility with dexmachina's argparser; no effect since the epoch-based fixed mode was removed
        "interval": 500,
        "first_ratio": 0.3,
        "first_stop_iter": 7000,
        "second_stop_iter": 16000,
        "fixed_mode": "exp", # or exp 
        # schedule=fixed runs on the env step clock: decays at fractions of the budget, zero from fixed_zero_fraction on
        "budget_env_steps": None,
        "fixed_start_fraction": 0.01,
        "fixed_zero_fraction": 0.6,

        # if schedule=exp: auto linear 
        "deque_freq": 1, # epoch count
        "grad_threshold": 0.0001,
        "gain_mode": "all", # which terms to reduce gain

        # if schedule=uniform: auto uniform distribution of gains
        "uniform_mode": "fast", # or slow
        # "upper_ratio": 0.9, # upper = curr_upper * (upper_ratio)
        # "lower_ratio": 0.8, # if fast: lower=curr_lower * lower_ratio, if slow: lower=curr_upper * lower_ratio
        "upper_ratios": dict(kp=0.5, kv=0.9, fr=0.95),
        "lower_ratios": dict(kp=0.5, kv=0.8, fr=0.9),
        "seed": 42,
        "decay_solimp": False,
        "solip_multiplier": 0.98,
        "d0_lower": 0.9,
        "dmid_lower": 0.95,
        "tconst_lower": 0.1,
        "tconst_upper": 0.15, # 0.2 is too soft
        "rew_thresholds": dict(task=0.5, con=0.05, imi=0.05, bc=0.05),
        "resample_every_epoch": -1, # if true and using uniform, re-sample gains every epoch
        "skip_grad": False, # if true, skip gradient check
        "zero_epoch": 30000, # set all zeros once beyond this hardstop
        "dialback_ep_len": 50,
        "dialback_min_epochs": 500,
        "dialback_ratios": dict(kp=0.98, kv=0.98, fr=0.98), # don't fully reset to prev gains, instead decay by this ratio
    }
    # update with kwargs
    curr_cfg.update(kwargs)
    return curr_cfg

SCHEDULE_OPTIONS=["fixed", "exp", "uniform"] 

INERT_OPTIONS = ("interval", "first_ratio", "first_stop_iter", "second_stop_iter")

# The gated schedules snap the gains to zero once kp (or fr) falls below these thresholds. The fixed schedule uses the
# kp one to count its decays: with the benchmark's kp_init=80 and upper ratio 0.9, 80 * 0.9**70 = 0.0515 and
# 80 * 0.9**71 = 0.0464 < 0.05, so the 71st ratio decay is the one that reaches zero.
KP_ZERO_THRESHOLD = 0.05
FR_ZERO_THRESHOLD = 0.01

class Curriculum:
    def __init__(self, curr_cfg, task_object, reward_keys, num_envs, achieved_length, max_episode_length):
        self.curr_cfg = curr_cfg
        defaults = get_curriculum_cfg()
        changed = {name: curr_cfg[name] for name in INERT_OPTIONS if curr_cfg[name] != defaults[name]}
        if changed:
            warnings.warn(
                f"Curriculum options {changed} have no effect since the epoch-based fixed mode was removed; "
                "remove them from the config.",
                UserWarning,
                stacklevel=2,
            )
        self.num_envs = num_envs
        self.object = task_object
        self.init_gains = {
            "kp": curr_cfg['kp_init'],
            "kv": curr_cfg['kv_init'],
            "fr": curr_cfg['force_range_init'],
        }
        self.curr_gains = self.init_gains.copy()
        self.curr_gains_lower = self.init_gains.copy() # use for uniform mode
        self.gain_history = [] # keep a list of gains for prev epochs 
        self.gain_mode = curr_cfg['gain_mode']
        self.decay_rew = curr_cfg['decay_rew']
        # first figure out which terms to decay:
        decay_terms = []
        if self.gain_mode == "all":
            decay_terms = ["kp", "kv", "fr"]
        elif "kp" in self.gain_mode:
            decay_terms.append("kp")
        elif "kv" in self.gain_mode:
            decay_terms.append("kv")
        elif "fr" in self.gain_mode:
            decay_terms.append("fr")
        else:
            raise ValueError("Invalid gain mode")
        self.decay_terms = decay_terms
        self.wait_epochs = curr_cfg['wait_epochs']

        # next get the params for different schedules
        self.schedule = curr_cfg['schedule']
        assert self.schedule in SCHEDULE_OPTIONS, "Invalid schedule"

        self.fixed_mode = curr_cfg['fixed_mode']
        assert self.fixed_mode in ['lin', 'exp', 'uniform'], "Invalid fixed mode"
        
        self.uniform_mode = curr_cfg['uniform_mode']
        self.obj_ndof = 7 # base + articulation
        assert self.uniform_mode in ['fast', 'slow'], "Invalid uniform mode"
        self.deque_appends = 0
        self.deque_append_freq = curr_cfg['deque_freq'] 

        self.decay_solimp = curr_cfg['decay_solimp']
        self.solip_multiplier = curr_cfg['solip_multiplier']
        self.d0_lower = curr_cfg['d0_lower']
        self.dmid_lower = curr_cfg['dmid_lower']
        self.tconst_lower = curr_cfg['tconst_lower']
        self.tconst_upper = curr_cfg['tconst_upper']

        self.upper_ratios = curr_cfg.get('upper_ratios', dict(kp=0.9, kv=0.9, fr=0.9))
        self.lower_ratios = curr_cfg.get('lower_ratios', dict(kp=0.8, kv=0.8, fr=0.8))

        self.env_steps = 0 # training env steps, see advance_env_steps
        self.fixed_decays_done = 0
        self.fixed_decay_steps = []
        if self.schedule == "fixed":
            self.fixed_decay_steps = self.get_fixed_decay_steps(curr_cfg)

        self.rew_deques = dict()
        self.rew_grads = dict() 
        self.deque_len = curr_cfg['deque_len'] 
        for key in reward_keys:
            assert key in ["task", "imi", "bc", "con"], f"Invalid reward key: {key}"
            self.rew_deques[key] = deque(maxlen=self.deque_len)
            self.rew_grads[key] = 1.0 
        self.ep_lens = deque(maxlen=self.deque_len) 

        self.reward_keys = reward_keys
        self.grad_threshold = curr_cfg['grad_threshold']  
        self.rew_thresholds = curr_cfg['rew_thresholds']
        self.max_episode_length = max_episode_length
        # self.achieved_length = achieved_length # need to update this from env!
        self.seed = curr_cfg['seed']
        # Epochs (not set_curriculum calls) since the last gain change. Wrappers that map training
        # progress onto epochs may skip epochs or repeat them, so counting calls would make the
        # 40-epoch spacing and dialback_min_epochs mean something different for every algorithm.
        self.last_decay_epoch = 0
        self.num_epoch_since_last_decay = 0
        self.num_epoch_since_zero = 0
        self.resample_every_epoch = curr_cfg['resample_every_epoch'] 
        # Own generators: seeding the global ones here would overwrite the run's seed at every env build.
        self.np_rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device=self.object.device).manual_seed(self.seed)
        self.skip_grad = curr_cfg['skip_grad']
        self.zero_epoch = curr_cfg.get('zero_epoch', 50000)

    def post_scene_build_setup(self):
        if self.decay_solimp:
            new_params = torch.tensor([self.tconst_upper, 1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0]) 
            self.object.entity._solver.set_geom_sol_params(new_params)
        return 
        
    def get_fixed_decay_steps(self, curr_cfg):
        """Env step of each fixed decay: equally spaced from the start fraction to the zero fraction of the budget."""
        budget = curr_cfg['budget_env_steps']
        if budget is None or budget <= 0:
            raise ValueError(
                f"schedule 'fixed' needs budget_env_steps > 0, got {budget}. The fixed schedule is a fraction of the "
                "run's training env steps, so it cannot be placed without them. Set budget_env_steps (--budget_env_steps)."
            )
        if self.fixed_mode == "lin":
            raise ValueError(
                "schedule 'fixed' does not support fixed_mode 'lin': it is defined on epochs, not on the env step "
                "clock. Use fixed_mode 'exp' or 'uniform'."
            )
        start, zero = curr_cfg['fixed_start_fraction'], curr_cfg['fixed_zero_fraction']
        if not 0 <= start < zero <= 1:
            raise ValueError(
                f"fixed schedule needs 0 <= fixed_start_fraction < fixed_zero_fraction <= 1, got {start}, {zero}; "
                "the decays are spaced evenly between the two fractions, so set fixed_start_fraction < fixed_zero_fraction."
            )
        count = self.get_fixed_decay_count()
        if count < 2:
            raise ValueError(
                f"schedule 'fixed' needs at least 2 decays, got {count}: kp starts at {self.init_gains['kp']} and the "
                f"upper ratio {self.upper_ratios['kp']} brings it below {KP_ZERO_THRESHOLD} in {count} step(s), so the "
                "decays cannot be spaced between the start and zero fractions. Raise kp_init or use a ratio closer to 1."
            )
        return [round(budget * (start + (zero - start) * i / (count - 1))) for i in range(count)]

    def get_fixed_decay_count(self):
        """Ratio decays until kp is below the zero threshold; the last of them snaps the gains to zero."""
        ratio = self.upper_ratios['kp']
        if 'kp' not in self.decay_terms or not 0 < ratio < 1:
            raise ValueError(
                f"schedule 'fixed' counts its decays by kp falling below {KP_ZERO_THRESHOLD}, so kp must be decayed "
                f"with an upper ratio in (0, 1); got gain_mode {self.gain_mode!r}, kp upper ratio {ratio}."
            )
        count, kp = 0, self.init_gains['kp']
        while kp >= KP_ZERO_THRESHOLD:
            kp *= ratio
            count += 1
        return count

    def advance_env_steps(self, num_envs):
        """Count training env steps and apply the fixed decays that are due; returns whether the gains changed.

        Gains depend only on the total count, not on how often or in what batches it is advanced.
        """
        self.env_steps += num_envs
        due = bisect_right(self.fixed_decay_steps, self.env_steps)
        changed = due > self.fixed_decays_done
        while self.fixed_decays_done < due:
            self.fixed_decays_done += 1
            self.set_fixed_decay(zero=self.fixed_decays_done == len(self.fixed_decay_steps))
        return changed

    def set_fixed_decay(self, zero):
        """One decay of the fixed schedule: the upper (and, if uniform, lower) ratios, or all zero for the last."""
        if zero:
            for k in self.decay_terms:
                self.curr_gains[k] = 0.0
                self.curr_gains_lower[k] = 0.0
        elif self.fixed_mode == "exp" or self.fixed_mode == "uniform":
            for k in self.decay_terms:
                fixed_ratio = self.upper_ratios.get(k, 0.9)
                # if self.curr_gains[k] < 1 and k == 'kp':
                #     fixed_ratio = 0.96
                self.curr_gains[k] *= fixed_ratio
            # if uniform, decay lower bound as well
            if self.fixed_mode == "uniform":
                for k in self.decay_terms:
                    low_ratio = self.lower_ratios[k]
                    if self.uniform_mode == "fast":
                        self.curr_gains_lower[k] *= low_ratio
                    elif self.uniform_mode == "slow":
                        self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio

    def determine_decay(self, epoch_num):
        reduce_gains = True 
        reason = f"Epoch {epoch_num}: "
        if epoch_num < self.wait_epochs:
            reason += f"waiting for {self.wait_epochs} epochs"
            reduce_gains = False
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                reduce_gains = False 
                reason += f"{key} deque too short: {len(self.rew_deques[key]) } "
                break 
            grad = self.rew_grads[key] 
            rew_mean = np.mean(self.rew_deques[key])
            rew_thres = self.rew_thresholds.get(key, 0.01)
            if rew_mean < rew_thres:
                reason += f"{key} reward too low: {rew_mean} "
                reduce_gains = False
            elif not self.skip_grad and abs(grad) > self.grad_threshold:
                reason += f"{key} grad too low/high: {grad} "
                reduce_gains = False 
            if not reduce_gains:
                break # no need to check other terms 
         
        if self.num_epoch_since_last_decay < 40:
            reason += f"too soon until last decay: {self.num_epoch_since_last_decay} epochs "
            reduce_gains = False
        
        if len(self.ep_lens) < self.deque_len:
            reason += f"ep lens deque too short: {len(self.ep_lens)} "
            reduce_gains = False
        else:
            achieved_len = np.mean(self.ep_lens)
            min_required_len = self.curr_cfg.get("min_achieved_len", self.max_episode_length * 0.85)
            if achieved_len < min_required_len:
                reason += f"max achieved length: {achieved_len:.1f} < {min_required_len:.1f} too low"
                reduce_gains = False
        return reduce_gains, reason 

    def determine_dialback(self, epoch_num):
        """ if the acheived ep length has been too low for too long, dial back the gains """
        max_past = self.curr_cfg["dialback_min_epochs"]
        min_ep_len = self.curr_cfg["dialback_ep_len"]
        ratios = self.curr_cfg["dialback_ratios"]
        dialed = False
        # print(f"Epoch {epoch_num}: Dialback check: {len(self.ep_lens)} past ep lens")
        if len(self.ep_lens) < self.deque_len:
            return dialed
        achieved_len = np.mean(self.ep_lens) 
        # print(f"Epoch {epoch_num}: Achieved length: {achieved_len} {self.num_epoch_since_last_decay}")
        if achieved_len < min_ep_len:
            if self.num_epoch_since_last_decay > max_past:
                # gain_history is only appended when a decay actually happens, so
                # before the first decay there is no previous gain level to return
                # to -- the gains are still at their init values and dialback is a
                # no-op. Previously this raised IndexError. It became reachable once
                # curriculum epochs became a normalised progress counter: with a
                # 10000-epoch budget, num_epoch_since_last_decay can exceed
                # dialback_min_epochs (500) while the reward gates are still
                # blocking the very first decay.
                if not self.gain_history:
                    return dialed
                # copy: these dicts live in gain_history, and mutating them in
                # place would corrupt the recorded history and alias curr_gains
                # to it, compounding across repeated dialbacks.
                upper_gains = dict(self.gain_history[-1]['gains'])
                lower_gains = dict(self.gain_history[-1]['lower'])
                for k in self.decay_terms:
                    if k in upper_gains:
                        ratio = ratios.get(k, 1.0)
                        upper_gains[k] *= ratio
                        lower_gains[k] *= ratio
                self.curr_gains = upper_gains
                self.curr_gains_lower = lower_gains
                dialed = True
                # TODO: also decay the solimp params
        return dialed

    def set_auto_exp_decay(self, epoch_num): 
        new_gains = dict()
        for k, v in self.curr_gains.items():
            ratio = self.upper_ratios[k]
            new_gains[k] = v * ratio
        if "kp" in new_gains and new_gains['kp'] < KP_ZERO_THRESHOLD or "fr" in new_gains and new_gains['fr'] < FR_ZERO_THRESHOLD:
            new_gains = {k: 0.0 for k in new_gains}
        for k in self.decay_terms:
            self.curr_gains[k] = new_gains[k]
        return True, ""

    def set_auto_uniform_decay(self, epoch_num):
        """ update an upper and lower bound for gains """  
        for k in self.decay_terms: 
            upp_ratio = self.upper_ratios[k]
            self.curr_gains[k] = self.curr_gains[k] * upp_ratio
        if self.uniform_mode == "fast":
            # use self.curr_gains as upper bound 
            for k in self.decay_terms: 
                low_ratio = self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains_lower[k] * low_ratio
        elif self.uniform_mode == "slow":
            # use self.curr_gains as upper bound 
            for k in self.decay_terms:
                low_ratio = self.lower_ratios[k]
                self.curr_gains_lower[k] = self.curr_gains[k] * low_ratio
        else:
            raise ValueError("Invalid uniform mode") 
            
        if ('kp' in self.decay_terms and self.curr_gains['kp'] < KP_ZERO_THRESHOLD) or ('fr' in self.decay_terms and self.curr_gains['fr'] < FR_ZERO_THRESHOLD):
            for k in self.decay_terms:
                self.curr_gains[k] = 0.0
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0 
        # if the lower bound kp is within 0.1 thres, reduce it to 0 
        if ('kp' in self.decay_terms and self.curr_gains_lower.get('kp', 1) < 0.1):
            for k in self.decay_terms:
                self.curr_gains_lower[k] = 0.0
            self.low_ratio = 0.0
        return True, ""
    
    def update_progress(self, rewards, achieved_length):
        self.deque_appends += 1
        if self.deque_appends % self.deque_append_freq == 0:
            for k, val in rewards.items():
                assert k in self.rew_deques, f"Invalid key: {k}"
                self.rew_deques[k].append(val)
            self.deque_appends = 0 
        # self.achieved_length = achieved_length
        self.ep_lens.append(achieved_length)

    def get_current_gains(self):
        ret_gains = self.curr_gains.copy()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            ret_gains.update(
                {f"{k}_lower": v for k, v in self.curr_gains_lower.items()}
            )
        if self.decay_solimp:
            ret_gains.update(
                # dict(d0_lower=self.d0_lower, dmid_lower=self.dmid_lower)
                dict(tconst_lower=self.tconst_lower, tconst_upper=self.tconst_upper)
            )
        return ret_gains

    def reset_object_gains(self):
        rand_gains = dict()
        if self.schedule == "uniform" or self.fixed_mode == "uniform":
            if any([self.curr_gains[key] == 0 for key in self.decay_terms]):
                rand_gains = {k: 0.0 for k in self.decay_terms}
            else:
                # sample for every joint dim and env dim! shape should be (num_envs, num_joints)
                for key in self.decay_terms:
                    lower = self.curr_gains_lower[key]
                    upper = self.curr_gains[key] 
                    # uniform from lower to upper, sample shape (num_envs, num_joints)
                    rand_gains[key] = torch.rand((self.num_envs, self.obj_ndof), dtype=torch.float32, device=self.object.device, generator=self.torch_rng) * (upper - lower) + lower
        else:
            # same params for all 
            for key in self.decay_terms:
                rand_gains[key] = self.curr_gains[key] 
        # this should handle missing shapes:
        self.object.set_joint_gains(
            kp=rand_gains.get("kp", None),
            kv=rand_gains.get("kv", None),
            force_range=rand_gains.get("fr", None),
            env_idxs=None,
        )
    
    def reset_solimp(self): 
        self.tconst_upper = max(self.tconst_upper * self.solip_multiplier, 0.02)
        self.tconst_lower = max(self.tconst_lower * self.solip_multiplier, 0.02)
        if self.tconst_upper == 0.02:
            tconst = 0.02
        else:
            tconst = self.np_rng.uniform(self.tconst_lower, self.tconst_upper)
        new_params = torch.tensor([tconst,1.0, self.d0_lower, self.dmid_lower, 0.001, 0.5, 2.0]) 
        self.object.entity._solver.set_geom_sol_params(new_params)
        return 

    def get_reward_grads(self):
        return self.rew_grads

    def update_reward_grads(self):
        for key in self.rew_deques.keys():
            if len(self.rew_deques[key]) < self.deque_len:
                continue 
            grad = np.gradient(self.rew_deques[key]).mean()
            self.rew_grads[key] = grad

    def decay_reward_weights(self, reward_module):
        decayed = False
        if not self.decay_rew:
            return decayed
        info = ""
        for attr in ['imi_rew_weight', 'bc_rew_weight', 'contact_rew_weight']:
            if hasattr(reward_module, attr):
                val = getattr(reward_module, attr)
                if val > 0.01:
                    setattr(reward_module, attr, val * 0.95)
                    info += f"{attr}: to {val * 0.95} "
                    decayed = True
        # also decay the rew thresholds
        for key in self.rew_thresholds.keys():
            self.rew_thresholds[key] = self.rew_thresholds[key] * 0.95
        if len(info) > 0:
            print(info)
        return decayed

    def set_curriculum(self, epoch_num):
        if self.schedule == "fixed":
            zero_gains = all(self.curr_gains[key] == 0.0 for key in self.decay_terms)
            return zero_gains, False, "fixed schedule: the gains follow the env step clock, epochs are ignored"
        # if gains are already zero, no need to reset
        self.update_reward_grads()
        # Identical to counting calls when called once per epoch (rl_games), correct otherwise.
        self.num_epoch_since_last_decay = epoch_num - self.last_decay_epoch
        learning_stablized, reason = self.determine_decay(epoch_num) 
        zero_gains = True 
        for key in self.decay_terms:
            if self.curr_gains[key] > 0.0:
                zero_gains = False
                break 
        # add to the count of epochs since last zero gains
        if zero_gains:
            self.num_epoch_since_zero += 1 
        gains_decayed = False 
        if epoch_num > self.zero_epoch:
            if not zero_gains:
                # set all gains to zero
                for key in self.decay_terms:
                    self.curr_gains[key] = 0.0
                    self.curr_gains_lower[key] = 0.0
                reason += f"Epoch {epoch_num}: Gains set to zero"  
                gains_decayed = True 
                self.reset_object_gains()
            return zero_gains, gains_decayed, reason 
        
        if zero_gains:
            return zero_gains, gains_decayed, reason

        # decide if need to dial back the gains - this must be checked independent
        # of `learning_stablized`, since that already requires achieved_len to
        # clear the (higher) ep_len_gate; gating dialback behind it made its own
        # (lower) dialback_ep_len check impossible to ever be true.
        dialed_back = self.determine_dialback(epoch_num)
        if not dialed_back and not learning_stablized:
            return zero_gains, gains_decayed, reason

        if dialed_back:
            print(f"Epoch {epoch_num}: Dialed back gains")
            gains_decayed = True # if the gains are dialed back, don't append to the history
        else:
            self.gain_history.append(
                dict(gains=self.curr_gains.copy(), lower=self.curr_gains_lower.copy())
            )

            if self.schedule == "exp":
                gains_decayed, reason = self.set_auto_exp_decay(epoch_num)
            elif self.schedule == "uniform":
                gains_decayed, reason = self.set_auto_uniform_decay(epoch_num)
            else:
                raise ValueError("Invalid schedule")

        if gains_decayed:
            new_gains = self.get_current_gains()
            print(f"Epoch {epoch_num}: New Gains: {new_gains}")
            self.reset_object_gains() 
            # also clear the deques!!
            for key in self.rew_deques.keys():
                self.rew_deques[key].clear()
                self.rew_grads[key] = 1.0
            self.ep_lens.clear()
            if self.decay_solimp:
                self.reset_solimp()
            self.deque_appends = 0
            self.last_decay_epoch = epoch_num
            self.num_epoch_since_last_decay = 0
        elif self.resample_every_epoch > 0 and epoch_num % self.resample_every_epoch == 0:
            self.reset_object_gains()
        return zero_gains, gains_decayed, reason
    
