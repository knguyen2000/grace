import torch

# --- Model & Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation ---
BERTSCORE_THRESHOLD = 0.90 # Central place for this threshold

# --- Retrieval ---
DEFAULT_HOPS = 2 # This will be superseded by CE policy, but good as a fallback
DEFAULT_WIDTH = 15

# --- File Paths ---
CE_POLICY_FILE = "policy_ce.json"
SIMULATION_DATA_FILE = "outputs/ce_simulation_data.json" # Pass 1 output
FINAL_RESULTS_FILE = "outputs/ce_final_results.json" # Pass 2 output

# =========================================================================
# --- Correlated Equilibrium (CE) Game Settings ---
# =========================================================================

CE_TEMPERATURE = 0.7         # was previously hardcoded in build_policy.py
CE_MIN_PROB_FLOOR = 0.02     # exploration floor

SIG_THRESHOLDS = {
    'reliability': [0.7, 0.85],
    'degree': [5, 20],
    'len': [1.5, 3.0],         # path length buckets
    'coherence': [0.5, 0.8],   # evidence coherence score
    'diversity': [1.5, 3.0],   # number of unique entities
}

# 1. PLAYER ACTION COSTS (how "expensive" is the action)
# These are unit-less, relative costs.
import torch

# --- Model & Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Evaluation ---
BERTSCORE_THRESHOLD = 0.90 # Central place for this threshold

# --- Retrieval ---
DEFAULT_HOPS = 2 # This will be superseded by CE policy, but good as a fallback
DEFAULT_WIDTH = 15

# --- File Paths ---
CE_POLICY_FILE = "policy_ce.json"
SIMULATION_DATA_FILE = "outputs/ce_simulation_data.json" # Pass 1 output
FINAL_RESULTS_FILE = "outputs/ce_final_results.json" # Pass 2 output

# =========================================================================
# --- Correlated Equilibrium (CE) Game Settings ---
# =========================================================================

CE_TEMPERATURE = 0.7         # was previously hardcoded in build_policy.py
CE_MIN_PROB_FLOOR = 0.02     # exploration floor

SIG_THRESHOLDS = {
    'reliability': [0.7, 0.85],
    'degree': [5, 20],
    'len': [1.5, 3.0],         # path length buckets
    'coherence': [0.5, 0.8],   # evidence coherence score
    'diversity': [1.5, 3.0],   # number of unique entities
}

# 1. PLAYER ACTION COSTS (how "expensive" is the action)
# These are unit-less, relative costs.
COSTS = {
    # Retriever
    'retrieve_shallow': 1.0,  # 1 hop
    'retrieve_deep': 3.0,     # 2 hops (more expensive)
    
    # Generator
    'generate': 2.0,          # Cost of running the T5 model
    'generate_consistency': 5.0, # Cost of running T5 5 times (discounted from 10.0)
    'refuse_to_generate': 0.0, # No cost
    
    # Verifier
    'run_nli_check': 4.0,     # NLI models are very expensive
    'skip_check': 0.0
}

# 2. OUTCOME UTILITIES (how "good" is the final result)
UTILITIES = {
    'correct_answer': 100.0,
    'incorrect_answer': -40.0,
    'correct_abstain': 10.0,
    'wrong_abstain': -10.0
}

# 3. ACTIONS (all possible joint actions)
# (R_action, G_action, V_action)
JOINT_ACTIONS = [
    # (Retriever, Generator, Verifier)
    ('retrieve_shallow', 'generate', 'run_nli_check'),
    ('retrieve_shallow', 'generate', 'skip_check'),
    ('retrieve_shallow', 'refuse_to_generate', 'skip_check'),
    
    ('retrieve_deep', 'generate', 'run_nli_check'),
    ('retrieve_deep', 'generate', 'skip_check'),
    ('retrieve_deep', 'refuse_to_generate', 'skip_check'),
    
    # New Self-Consistency Actions
    ('retrieve_shallow', 'generate_consistency', 'skip_check'),
    ('retrieve_deep', 'generate_consistency', 'skip_check'),
]

# 4. DEFAULT ACTION (if signal is unknown)
# This is our fallback for unseen signals: cheapest, safest action
DEFAULT_POLICY = ('retrieve_shallow', 'refuse_to_generate', 'skip_check')