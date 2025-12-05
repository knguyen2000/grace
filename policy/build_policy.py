import json
import collections
import numpy as np
from typing import Dict, List, Any
from config import CE_POLICY_FILE, CE_TEMPERATURE, CE_MIN_PROB_FLOOR, SIG_THRESHOLDS

def _softmax_with_floor(x: np.ndarray, temperature: float = CE_TEMPERATURE, eps: float = 0.0) -> np.ndarray:
    z = (x - x.max()) / max(1e-8, temperature)
    p = np.exp(z); p = p / p.sum()
    if eps > 0:
        p = (1.0 - eps * len(x)) * p + eps
        p = np.maximum(p, 0.0)
        p = p / p.sum()
    return p

def build_ce_policy(input_simulation_file: str,
                    output_policy_file: str = CE_POLICY_FILE,
                    temperature: float = CE_TEMPERATURE,
                    min_prob_floor: float = CE_MIN_PROB_FLOOR) -> None:
    print(f"Building CE policy from {input_simulation_file}...")
    with open(input_simulation_file, 'r', encoding='utf-8') as f:
        simulation_data: List[Dict[str, Any]] = json.load(f)

    payoffs = collections.defaultdict(lambda: collections.defaultdict(list))
    thresholds_seen = None

    for row in simulation_data:
        s = row.get('signal_bin')
        if s is None: continue
        thresholds_seen = thresholds_seen or row.get('signal_thresholds')
        for a_str, net_u in row.get('action_utilities', {}).items():
            if net_u is not None:
                payoffs[s][a_str].append(float(net_u))

    ce_policy = {}
    print("\n--- CE (mixed) Policy ---")
    print(f"{'Signal Bin':<24} | {'Action':<55} | {'Prob':<7} | {'AvgU':<8}")
    print("-" * 105)

    for s, actions in payoffs.items():
        action_list = list(actions.keys())
        if not action_list:
            continue
        utilities = np.array([np.mean(actions[a]) for a in action_list], dtype=float)
        probs = _softmax_with_floor(utilities, temperature=temperature, eps=min_prob_floor)
        ce_policy[s] = {action_list[i]: float(probs[i]) for i in range(len(action_list))}
        for a, p, u in zip(action_list, probs, utilities):
            if p > 1e-6:
                print(f"{s:<24} | {a:<55} | {p:<7.3f} | {u:<8.2f}")

    if thresholds_seen is None:
        thresholds_seen = SIG_THRESHOLDS
        print("Warning: No thresholds bundled in simulation data; using config SIG_THRESHOLDS.")

    payload = {
        'policy': ce_policy,
        'signal_thresholds': thresholds_seen,
        'temperature': temperature,
        'min_prob_floor': min_prob_floor
    }
    with open(output_policy_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n CE policy saved to {output_policy_file}")
