import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation
BERTSCORE_THRESHOLD = 0.90

# Retrieval
DEFAULT_HOPS = 2
DEFAULT_WIDTH = 15

# Adaptive Retrieval Settings
MAX_HOPS_LIMIT = 10         # Safety limit to prevent combinatorial explosion
STOP_THRESHOLD_SHALLOW = 0.75 # Good enough score to stop early for shallow retrieval
STOP_THRESHOLD_DEEP = 0.90    # High confidence score required to stop for deep retrieval

# File Paths
CE_POLICY_FILE = "policy_ce.json"
SIMULATION_DATA_FILE = "outputs/ce_simulation_data.json" # Pass 1 output
FINAL_RESULTS_FILE = "outputs/ce_final_results.json" # Pass 2 output

# Correlated Equilibrium (CE) Game Settings

CE_TEMPERATURE = 0.7
CE_MIN_PROB_FLOOR = 0.02

SIG_THRESHOLDS = {
    'reliability': [0.7, 0.85],
    'degree': [5, 20],
    'len': [1.5, 3.0],         # path length buckets
    'coherence': [0.5, 0.8],   # evidence coherence score
    'diversity': [1.5, 3.0],   # number of unique entities
}

# 1. PLAYER ACTION COSTS (how expensive is the action)
# unit-less, relative costs

COSTS = {
    # Retriever
    'retrieve_shallow': 1.0,
    'retrieve_deep': 3.0,
    
    # Generator
    'generate': 2.0,
    'generate_consistency': 5.0,
    'refuse_to_generate': 0.0,
    
    # Verifier
    'run_nli_check': 4.0,
    'skip_check': 0.0,

    # Parametric Actions
    'skip_retrieval': 0.0,
    'generate_parametric': 2.0
}

# 2. OUTCOME UTILITIES
UTILITIES = {
    'correct_answer': 120.0,
    'incorrect_answer': -40.0,
    'correct_abstain': 10.0,
    'wrong_abstain': -20.0
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
    
    ('retrieve_shallow', 'generate_consistency', 'skip_check'),
    ('retrieve_deep', 'generate_consistency', 'skip_check'),

    # Parametric (Fallback) Action
    ('skip_retrieval', 'generate_parametric', 'skip_check')
]

# 4. DEFAULT ACTION (if signal is unknown)
# fallback for unseen signals: cheapest, safest action
DEFAULT_POLICY = ('retrieve_shallow', 'refuse_to_generate', 'skip_check')