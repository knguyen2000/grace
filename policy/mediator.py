# policy/mediator.py
import json
import numpy as np
from ast import literal_eval
from config import DEFAULT_POLICY, CE_POLICY_FILE, SIG_THRESHOLDS
from utils.policy_utils import get_signal_bin as compute_signal_bin

class CeMediator:
    def __init__(self, policy_file_path: str = CE_POLICY_FILE):
        print(f"Loading CE policy from {policy_file_path}...")
        with open(policy_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.policy = data.get('policy', {})
        # Use thresholds learned in Phase 1; if missing, default to config
        self.thresholds = data.get('signal_thresholds', SIG_THRESHOLDS)
        print("Policy bins:", len(self.policy))
        print("Thresholds:", self.thresholds)

    def get_signal_bin(self, G, start_node_id, path_metrics):
        return compute_signal_bin(G, start_node_id, self.thresholds, path_metrics)

    def get_joint_action(self, G, start_node_id, path_metrics, deterministic: bool = False):
        s = self.get_signal_bin(G, start_node_id, path_metrics)
        dist = self.policy.get(s)
        
        if not dist:
            return tuple(DEFAULT_POLICY)
        
        actions = list(dist.keys())
        probs = np.array([float(dist[a]) for a in actions], dtype=float)
        
        if probs.sum() == 0:
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = probs / probs.sum()

        chosen = actions[np.argmax(probs)] if deterministic else np.random.choice(actions, p=probs)
        
        try:
            return tuple(literal_eval(chosen))
        except Exception:
            if isinstance(chosen, (list, tuple)) and len(chosen) == 3:
                return tuple(chosen)
            return tuple(DEFAULT_POLICY)