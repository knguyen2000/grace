# eval/run_evaluation.py
import json
import sys
import io
import os
import random
import time
import argparse
from ast import literal_eval
from pathlib import Path

# --- Path Setup ---
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Third Party Imports ---
import numpy as np
import torch
import nltk
from bert_score import score as bert_score
from datasets import load_dataset

# Fix Windows/encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Mock tqdm if not present
def tqdm(iterable, **kwargs):
    return iterable

# --- NLTK Setup ---
try:
    print("Checking NLTK resources...")
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("NLTK resources found.")
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK resources downloaded.")

# --- Local Imports ---
from config import (
    DEVICE,
    BERTSCORE_THRESHOLD,
    CE_POLICY_FILE,
    SIMULATION_DATA_FILE,
    FINAL_RESULTS_FILE,
    COSTS,
    UTILITIES,
    JOINT_ACTIONS,
    DEFAULT_WIDTH,
    DEFAULT_POLICY
)

# Project Modules
from graph.build_networkx import build_graph
from retrieval.retriever import retrieve_paths, select_best_path
from retrieval.entity_linking import pick_start_node_from_question
from models.generator import Generator
from models.verifier import NliVerifier
from policy.mediator import CeMediator
from policy.build_policy import build_ce_policy
from eval.evaluation import evaluate_batch
from utils.policy_utils import get_signal_bin
from utils.text_normalization import TextNormalizer
from utils.answer_validation import AnswerValidator, validate_answer_batch

BASELINE_CONF_THRESHOLD = 0.7
GRAPHRAG_POLICY = ('retrieve_shallow', 'generate', 'skip_check')

# Helper functions
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _separate_thresholds(th):
    a, b = float(th[0]), float(th[1])
    if np.isclose(a, b, rtol=0, atol=1e-3):
        return [a - 0.05, b + 0.05]
    return [a, b]

def load_hotpot_subset(limit=200, start_index=0):
    ds = load_dataset("hotpot_qa", "distractor")["validation"]
    end_index = start_index + limit
    if end_index > len(ds):
        end_index = len(ds)
    subset = []
    for i in range(start_index, end_index):
        ex = ds[i]
        subset.append({
            "id": str(i),
            "question": ex["question"].strip(),
            "gold_answer": ex["answer"].strip()
        })
    return subset

def is_correct_bertscore(preds, golds, exact_match_threshold_len=5):

    if not preds or not golds or len(preds) != len(golds):
        print(f"Warning: Mismatched lengths or empty lists in is_correct_bertscore. Preds: {len(preds)}, Golds: {len(golds)}")
        return [False] * len(preds)

    valid_preds = [p if p is not None else "" for p in preds]
    valid_golds = [g if g is not None else "" for g in golds]
    
    # Helper function for BERTScore computation (single pair)
    def compute_bertscore_single(pred, gold):
        try:
            P, R, F1 = bert_score([pred], [gold], lang="en", device=DEVICE, verbose=False)
            return float(F1[0].item())
        except Exception:
            return 0.0
    
    # Helper function for BERTScore batch computation
    def compute_bertscore_batch(preds_batch, golds_batch):
        try:
            P, R, F1 = bert_score(preds_batch, golds_batch, lang="en", device=DEVICE, verbose=False)
            return [float(f.item()) for f in F1]
        except Exception as e:
            print(f"Error during BERTScore batch calculation: {e}")
            return [0.0] * len(preds_batch)
    
    # Use the improved validation system
    validation_results = validate_answer_batch(
        preds=valid_preds,
        golds=valid_golds,
        bertscore_threshold=float(BERTSCORE_THRESHOLD),
        use_bertscore=True,
        bertscore_batch_func=compute_bertscore_batch
    )
    
    # Extract boolean results
    is_correct_list = [bool(result[0]) for result in validation_results]
    
    return is_correct_list

def build_evidence_from_path(G, path):
    if not path:
        return ""
    all_nodes = list(dict.fromkeys(path))
    parts = []
    if len(path) > 1:
        try:
            for u, v in zip(path[:-1], path[1:]):
                if u not in G.nodes or v not in G.nodes:
                    continue
                ulabel = G.nodes[u].get("label", u)
                vlabel = G.nodes[v].get("label", v)
                rel = G.edges[u, v].get("relation", "related_to") if G.has_edge(u, v) else "related_to (edge missing?)"
                parts.append(f"{ulabel} --[{rel}]--> {vlabel}")
        except Exception as e:
            print(f"Warning: Unexpected error during evidence path string creation: {e}")
    node_list = " ; ".join(G.nodes[n].get("label", n) for n in all_nodes if n in G.nodes)
    if not parts:
        return f"NODES: {node_list}" if node_list else ""
    return f"{' | '.join(parts)} NODES: {node_list}"

def compute_evidence_metrics(G, path):

    if not path:
        return {
            "path_len": 0,        # edges in path
            "diversity": 0,       # unique entities count
            "coherence": 0.0      # 0..1 heuristic
        }
    # path length in edges
    path_len = max(0, len(path) - 1)
    # diversity = unique node labels along the path
    labels = []
    for n in path:
        if n in G.nodes:
            labels.append(G.nodes[n].get("label", str(n)))
    diversity = len(set(labels))

    # coherence: relation-consistency heuristic in [0,1]
    # count how often the relation label remains the same across consecutive edges
    rels = []
    for u, v in zip(path[:-1], path[1:]):
        rels.append(G.edges[u, v].get("relation", "related_to") if G.has_edge(u, v) else "MISSING")
    if len(rels) <= 1:
        coherence = 1.0 if len(rels) == 1 else 0.0
    else:
        same = sum(1 for a, b in zip(rels[:-1], rels[1:]) if a == b)
        coherence = float(same) / float(len(rels) - 1)

    return {
        "path_len": int(path_len),
        "diversity": int(diversity),
        "coherence": float(coherence)
    }

def _fmt(x, nd=2):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)

def execute_joint_action(G, gen, verifier, question, start_node_id, joint_action, width):

    if not isinstance(joint_action, tuple) or len(joint_action) != 3:
        joint_action = DEFAULT_POLICY

    action_R, action_G, action_V = joint_action
    total_cost = 0.0
    best_path = None
    evidence = ""
    metrics = {"path_len": 0, "diversity": 0, "coherence": 0.0}

    # Retrieval
    if action_R == 'skip_retrieval':
         # Parametric mode: intentionally skip retrieval
        best_path = []
        evidence = ""
        total_cost += float(COSTS.get(action_R, 0.0))
    elif start_node_id and start_node_id in G.nodes:
        try:
            paths = retrieve_paths(G, start_node_id, action=action_R, max_width=width)
            if paths:
                best_path = select_best_path(G, paths)
                evidence = build_evidence_from_path(G, best_path)
                metrics = compute_evidence_metrics(G, best_path)
        except Exception as e:
            print(f"\n Error during retrieval/path selection for node {start_node_id}: {e}")
            evidence = ""
            best_path = None
            metrics = {"path_len": 0, "diversity": 0, "coherence": 0.0}
        total_cost += float(COSTS.get(action_R, 1.0))
    else:
        total_cost += float(COSTS.get(action_R, 1.0))

    # Handle Retrieval Failure
    if not evidence and action_R != 'skip_retrieval':
        # If we tried to retrieve but got nothing (dead end or disconnected node)
        # It is irrational to pay for 'generate' or 'run_nli_check' => fallback.
        
        if action_G == 'generate':
            action_G = 'generate_parametric'
            
        if action_V == 'run_nli_check':
            action_V = 'skip_check'

    # Generation
    answer = None
    if action_G == 'generate':
        total_cost += float(COSTS.get(action_G, 2.0))
        if evidence:
            try:
                # Pass best_path to the generator ---
                answer = gen.generate_from_evidence(question, evidence, path=best_path)
                # Standardize "I don't know" as abstention
                if answer and answer.lower().strip().strip('"').strip("'") == "i don't know":
                    answer = None
            except Exception as e:
                print(f"\n Error during generation for question {question}: {e}")
                answer = None
    elif action_G == 'generate_parametric':
         # Parametric generation: use baseline generator (no evidence)
         total_cost += float(COSTS.get(action_G, 2.0))
         try:
             ans, _ = gen.generate_baseline(question, return_confidence=True)
             answer = ans
         except Exception as e:
             print(f"\nError during parametric generation for {question}: {e}")
             answer = None
    elif action_G == 'generate_consistency':
        total_cost += float(COSTS.get(action_G, 5.0))
        try:
            answer = gen.generate_with_consistency(question, k=5, threshold=0.8)
            # Standardize "I don't know" as abstention
            if answer and answer.lower().strip().strip('"').strip("'") == "i don't know":
                answer = None 
        except Exception as e:
            print(f"\n Error during consistency generation for {question}: {e}")
            answer = None
    else:
        total_cost += float(COSTS.get(action_G, 0.0))
        answer = None

    # Verification
    v_label = "skipped"
    if action_V == 'run_nli_check':
        total_cost += float(COSTS.get(action_V, 4.0))
        if evidence and answer:
            try:
                v_label, _ = verifier.verify(evidence, answer)
                if v_label == 'contradicts':
                    # force abstain when contradiction detected
                    answer = None
            except Exception as e:
                print(f"\nError during verification for {question}: {e}")
                v_label = "error"
                answer = None
        else:
            v_label = "no_answer_to_verify" if not answer else "no_evidence_to_verify"
    else:
        total_cost += float(COSTS.get(action_V, 0.0))

    return answer, evidence, v_label, float(total_cost), best_path, metrics

def get_ce_justification(joint_action, outcome, signal_bin=None, metrics=None, reliability=None, degree=None, evidence=None):
    
    if not isinstance(joint_action, tuple) or len(joint_action) != 3:
        joint_action = DEFAULT_POLICY
    R_act, G_act, V_act = joint_action

    parts = [f"Policy: [R:{R_act}, G:{G_act}, V:{V_act}]"]
    if signal_bin:
        parts.append(f"Signal bin: {signal_bin}")

    # Evidence metrics line (no '?')
    m = metrics or {}
    parts.append(
        "Evidence metrics: "
        f"path_len={_fmt(m.get('path_len'))}, "
        f"coherence={_fmt(m.get('coherence'))}, "
        f"diversity={_fmt(m.get('diversity'))}, "
        f"reliability={_fmt(reliability)}, degree={_fmt(degree, 0)}"
    )

    if outcome == "abstain":
        if G_act == 'refuse_to_generate':
            parts.append("Decision: Abstained due to weak/insufficient signal (policy refused to generate).")
        else:
            parts.append("Decision: Abstained after pipeline produced no safe answer.")
    else:
        parts.append("Decision: Answered with sufficient structural confidence.")

    if evidence:
        ev = evidence.replace("\n", " ")
        parts.append(f"EVIDENCE_USED: {ev[:200]}{'...' if len(ev) > 200 else ''}")

    return parts

def get_abstain_description(joint_action, v_label, evidence):

    # Case: Explicit refusal to generate
    if isinstance(joint_action, (list, tuple)) and len(joint_action) == 3:
        if joint_action[1] == 'refuse_to_generate':
            return (
                "I’m not able to give an answer because the signals I detected "
                "suggest that responding could lead to an unreliable or unsafe answer."
            )

    # Other abstention reasons
    if v_label == 'contradicts':
        return (
            "I chose not to answer because the information I found disagrees with "
            "the possible answer. Since the evidence does not support a clear conclusion, "
            "I prefer not to guess."
        )

    elif not evidence:
        return (
            "I’m not able to answer because I could not find any information related "
            "to the question. Without supporting evidence, I prefer not to guess."
        )

    elif v_label == 'error':
        return (
            "I could not answer due to an unexpected issue while checking the information. "
            "Because I couldn’t confirm the reliability of the evidence, I preferred to abstain."
        )

    elif v_label == 'no_answer_to_verify':
        return (
            "I’m not able to answer because I couldn't form a clear response from the information available. "
            "Since there was no meaningful answer to evaluate, I chose to abstain."
        )

    elif v_label == 'no_evidence_to_verify':
        return (
            "I found a possible answer, but I could not locate any information to support it. "
            "Without evidence to verify the answer, I preferred to abstain."
        )

    elif v_label == 'skipped':
        return (
            "I’m not certain enough to answer the question. The information I found did not "
            "give me enough confidence, so I chose not to guess."
        )

    # Default fallback
    return (
        "I chose not to answer because I couldn’t ensure the information was reliable enough. "
        "To avoid giving a potentially incorrect answer, I preferred to abstain."
    )

# Main evaluation
def main(args):
    seed_everything()
    Path("outputs").mkdir(exist_ok=True)

    print("--- Initializing Models ---")
    G = build_graph()
    print(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    gen = Generator()
    verifier = NliVerifier()

    calibration_questions = load_hotpot_subset(limit=args.calib_size, start_index=0)
    evaluation_questions = load_hotpot_subset(limit=args.eval_size, start_index=args.calib_size)
    if not calibration_questions:
        print("Error: No calibration questions.")
        sys.exit(1)

    all_reliabilities, all_degrees = [], []
    all_path_lens, all_coherences, all_diversities = [], [], []

    for ex in tqdm(calibration_questions, desc="Gathering stats for thresholds"):
        start_node_id = pick_start_node_from_question(G, ex["question"])
        if start_node_id and start_node_id in G.nodes:
            node = G.nodes[start_node_id]
            all_reliabilities.append(node.get('reliability', 0.5))
            all_degrees.append(G.degree(start_node_id))
            # probe a shallow retrieval to estimate typical path metrics
            try:
                paths = retrieve_paths(G, start_node_id, action='retrieve_shallow', max_width=DEFAULT_WIDTH)
                best_path = select_best_path(G, paths) if paths else None
                m = compute_evidence_metrics(G, best_path or [])
            except Exception:
                m = {"path_len": 0, "diversity": 0, "coherence": 0.0}
            all_path_lens.append(m["path_len"])
            all_coherences.append(m["coherence"])
            all_diversities.append(m["diversity"])
        else:
            all_reliabilities.append(0.5)
            all_degrees.append(0)
            all_path_lens.append(0)
            all_coherences.append(0.0)
            all_diversities.append(0)

    rel_thresh = _separate_thresholds(np.percentile(all_reliabilities, [33.3, 66.6]).tolist())
    deg_thresh = _separate_thresholds(np.percentile(all_degrees, [33.3, 66.6]).tolist())
    len_thresh = _separate_thresholds(np.percentile(all_path_lens, [33.3, 66.6]).tolist())
    coh_thresh = _separate_thresholds(np.percentile(all_coherences, [33.3, 66.6]).tolist())
    div_thresh = _separate_thresholds(np.percentile(all_diversities, [33.3, 66.6]).tolist())

    adaptive_thresholds = {
        'reliability': rel_thresh,
        'degree': deg_thresh,
        'len': len_thresh,
        'coherence': coh_thresh,
        'diversity': div_thresh
    }
    print("Adaptive thresholds:")
    print(f"  reliability: {rel_thresh}")
    print(f"  degree     : {deg_thresh}")
    print(f"  len        : {len_thresh}")
    print(f"  coherence  : {coh_thresh}")
    print(f"  diversity  : {div_thresh}")

    # Phase 1 Simulation
    if os.path.exists(SIMULATION_DATA_FILE) and not args.force_calib:
        print(f"Skipping simulation: Found existing data at {SIMULATION_DATA_FILE}.")
    else:
        simulation_data = []
        all_sims_to_score = []
        all_sim_preds = []
        
        all_baseline_preds_for_calib = []
        print("Running baseline model for calibration set...")
        for ex in tqdm(calibration_questions, desc="Running baseline"):
            all_baseline_preds_for_calib.append(gen.generate_baseline(ex["question"]))
        
        baseline_correct_list_for_calib = is_correct_bertscore(
            all_baseline_preds_for_calib,
            [ex["gold_answer"] for ex in calibration_questions]
        )

        for i, ex in enumerate(tqdm(calibration_questions, desc="Simulating Actions")):
            q, gold = ex["question"], ex["gold_answer"]
            start_node_id = pick_start_node_from_question(G, q)
            
            path_metrics = {"path_len": 0, "diversity": 0, "coherence": 0.0}
            if start_node_id and start_node_id in G.nodes:
                try:
                    # Use a 'default' retrieval action for probing
                    paths = retrieve_paths(G, start_node_id, action='retrieve_shallow', max_width=args.width)
                    best_path = select_best_path(G, paths) if paths else None
                    path_metrics = compute_evidence_metrics(G, best_path or [])
                except Exception:
                    pass # Keep default metrics

            signal_bin = get_signal_bin(G, start_node_id, adaptive_thresholds, path_metrics)
            
            sim_entry = {
                "id": ex["id"],
                "question": q,
                "gold": gold,
                "signal_bin": signal_bin,
                "signal_thresholds": adaptive_thresholds,
                "action_utilities": {}
            }
            simulation_data.append(sim_entry)
            
            for joint_action in JOINT_ACTIONS:
                answer, _, v_label, cost, _, _ = execute_joint_action(
                    G, gen, verifier, q, start_node_id, joint_action, args.width
                )
                all_sims_to_score.append((i, str(joint_action), gold, answer is None))
                all_sim_preds.append(answer if answer else "")

        correct_list = is_correct_bertscore(all_sim_preds, [s[2] for s in all_sims_to_score])
        
        for i, is_correct in enumerate(correct_list):
            data_index, action_str, gold, was_abstain = all_sims_to_score[i]
            baseline_was_correct = baseline_correct_list_for_calib[data_index]

            joint_action = literal_eval(action_str)
            cost = sum(float(COSTS[a]) for a in joint_action)
            
            if was_abstain:
                if not baseline_was_correct:
                    outcome_utility = float(UTILITIES['correct_abstain'])
                else:
                    outcome_utility = float(UTILITIES['wrong_abstain'])
            elif is_correct:
                outcome_utility = float(UTILITIES['correct_answer'])
            else:
                outcome_utility = float(UTILITIES['incorrect_answer'])
                
            net_utility = outcome_utility - cost
            simulation_data[data_index]['action_utilities'][action_str] = float(net_utility)

        with open(SIMULATION_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(simulation_data, f, indent=2, ensure_ascii=False)
        print(f"Simulation data saved to {SIMULATION_DATA_FILE}")

    # Build CE Policy
    build_ce_policy(SIMULATION_DATA_FILE, CE_POLICY_FILE)

    # Phase 2 Evaluation
    if args.skip_eval or not evaluation_questions:
        print("Skipping Phase 2 evaluation.")
        return

    mediator = CeMediator(CE_POLICY_FILE)

    final_results, all_golds = [], []
    # Model Lists
    preds_baseline = []
    preds_graphrag = []
    preds_baseline_ce = []
    preds_grace = []

    # Overhead Measurement Accumulators
    total_signal_time = 0.0
    total_policy_time = 0.0
    total_execution_time = 0.0
    total_samples_measured = 0
    for ex in tqdm(evaluation_questions, desc="Evaluating 4 Models"):
        q, gold = ex["question"], ex["gold_answer"]
        
        # Model 1: Baseline & Model 3: Baseline + CE
        baseline_text, baseline_conf = gen.generate_baseline(q, return_confidence=True)
        
        # Model 3 Logic
        if baseline_conf < BASELINE_CONF_THRESHOLD:
            b_ce_text = "" # Abstain
        else:
            b_ce_text = baseline_text

        # Model 2: GraphRAG
        start_node_id = pick_start_node_from_question(G, q)
        gr_answer, _, _, gr_cost, _, _ = execute_joint_action(
            G, gen, verifier, q, start_node_id, GRAPHRAG_POLICY, args.width
        )
        gr_text = gr_answer if gr_answer else ""

        # Model 4: GRACE (Existing Logic)
        path_metrics = {"path_len": 0, "diversity": 0, "coherence": 0.0}
        best_path_probe = None # Store the probed path for metrics
        if start_node_id and start_node_id in G.nodes:
            try:
                # Use retrieve_shallow for probing consistently
                paths_probe = retrieve_paths(G, start_node_id, action='retrieve_shallow', max_width=args.width)
                best_path_probe = select_best_path(G, paths_probe) if paths_probe else None
                path_metrics = compute_evidence_metrics(G, best_path_probe or [])
            except Exception as e:
                print(f"Warning: Error during metric probing for {ex['id']}: {e}")
                pass # Keep default metrics

        # Measure Overhead
        t0 = time.perf_counter()
        signal_bin_determined = mediator.get_signal_bin(G, start_node_id, path_metrics)
        t1 = time.perf_counter()
        joint_action_taken = mediator.get_joint_action(G, start_node_id, path_metrics)
        t2 = time.perf_counter()
        
        signal_time = t1 - t0
        policy_time = t2 - t1
        
        total_signal_time += signal_time
        total_policy_time += policy_time

        # Execute the chosen action
        t3 = time.perf_counter()
        ce_answer_text, evidence_used, v_label, cost_incurred, actual_path, actual_metrics = execute_joint_action(
            G, gen, verifier, q, start_node_id, joint_action_taken, args.width
        )
        t4 = time.perf_counter()
        
        exec_time = t4 - t3
        total_execution_time += exec_time
        total_samples_measured += 1

        # Node reliability/degree for logging (using start_node_id)
        if start_node_id and start_node_id in G.nodes:
            reliability = float(G.nodes[start_node_id].get('reliability', 0.5))
            degree = int(G.degree(start_node_id))
        else:
            reliability, degree = 0.5, 0

        # Use the metrics from the actual executed path for logging
        final_metrics = actual_metrics

        system_abstained = (ce_answer_text is None)
        if system_abstained:
            final_answer_output = get_abstain_description(joint_action_taken, v_label, evidence_used)
            preds_grace.append("") 
        else:
            final_answer_output = ce_answer_text
            preds_grace.append(ce_answer_text)

        all_golds.append(gold)
        preds_baseline.append(baseline_text)
        preds_graphrag.append(gr_text)
        preds_baseline_ce.append(b_ce_text)

        # Store boolean for robust checking later
        final_results.append({
            "id": ex["id"],
            "question": q,
            "gold": gold,
            "baseline_pred": baseline_text,
            "signal_bin": signal_bin_determined,
            "joint_action": joint_action_taken,
            "final_answer": final_answer_output, # Contains either answer or descriptive abstention
            "final_cost": float(cost_incurred),
            "evidence": evidence_used,
            "system_abstained": system_abstained,
            "evidence_metrics": {
                "path_len": int(final_metrics.get("path_len", 0)),
                "coherence": float(final_metrics.get("coherence", 0.0)),
                "diversity": int(final_metrics.get("diversity", 0)),
                "reliability": float(reliability), # Start node reliability
                "degree": int(degree) # Start node degree
            }
        })

    # --- Compute Metrics for All 4 Models ---
    def get_metrics(preds, golds, model_name):
        correct_list = is_correct_bertscore(preds, golds)
        # Accuracy (overall)
        acc = sum(correct_list) / len(correct_list)
        
        # Abstention Rate
        num_abstain = sum(1 for p in preds if not p)
        abstain_rate = num_abstain / len(preds)
        
        # Accuracy on Answered
        num_answered = len(preds) - num_abstain
        acc_answered = sum(correct_list) / max(1, num_answered)
        
        return acc, abstain_rate, acc_answered

    acc_base, abs_base, acc_ans_base = get_metrics(preds_baseline, all_golds, "Baseline")
    acc_gr, abs_gr, acc_ans_gr = get_metrics(preds_graphrag, all_golds, "GraphRAG")
    acc_bce, abs_bce, acc_ans_bce = get_metrics(preds_baseline_ce, all_golds, "Baseline+CE")
    acc_grace, abs_grace, acc_ans_grace = get_metrics(preds_grace, all_golds, "GRACE")

    ce_correct_list = is_correct_bertscore(preds_grace, all_golds)
    baseline_correct_list = is_correct_bertscore(preds_baseline, all_golds)

    all_net_utilities = []
    processed_results = []

    # --- Process results for evaluation metrics ---
    for i, r in enumerate(final_results):
        is_ce_correct = bool(ce_correct_list[i])
        baseline_correct = bool(baseline_correct_list[i])
        answer_text = r['final_answer']
        cost = float(r.get('final_cost', 0.0))
        r['baseline_correct'] = bool(baseline_correct)

        # Determine if the system intended to abstain based on the action taken
        system_chose_to_abstain = r.get('system_abstained', False)
        
        # Also catch older cases if 'system_abstained' key wasn't present (backward compat)
        if not system_chose_to_abstain:
             if isinstance(r.get('joint_action'), (list, tuple)) and len(r['joint_action']) == 3:
                  # Policy refused, or verifier contradicted (forcing abstention in execute_joint_action)
                  if r['joint_action'][1] == 'refuse_to_generate' or (str(answer_text).startswith("Abstained: Verifier found contradiction")):
                       system_chose_to_abstain = True
             # Fallback check on the string content if needed
             elif isinstance(answer_text, str) and answer_text.startswith("Abstained:"):
                  system_chose_to_abstain = True


        if system_chose_to_abstain:
            outcome = "abstain"
            
            # --- Human-in-the-Loop Clarification ---
            # Instead of a generic message, generate a specific request
            try:
                 clarification = gen.generate_clarification(r['question'], r.get('evidence', ''))
                 r['final_answer'] = f"Abstained: {clarification}" 
            except Exception as e:
                 r['final_answer'] = "Abstained: I need more information to answer this question safe ly."
            
            # Reward correct abstain (baseline also failed), penalize wrong abstain (baseline succeeded)
            if not baseline_correct:
                outcome_utility = float(UTILITIES['correct_abstain'])
                r['abstain_correct'] = True
            else:
                outcome_utility = float(UTILITIES['wrong_abstain'])
                r['abstain_correct'] = False
            # These flags are used by evaluate_batch
            r['graph_action'] = "abstain"
            r['graph_correct'] = False # Abstention doesn't count as a correct *answer* for accuracy metrics
        
        # --- Case: System attempted to answer ---
        elif is_ce_correct: # Check BERTScore only if system didn't abstain
            outcome = "correct_answer"
            outcome_utility = float(UTILITIES['correct_answer'])
            r['graph_action'] = "answer"
            r['graph_correct'] = True
            r['abstain_correct'] = None # Not applicable
        else: # System answered, but BERTScore was low
             # This includes cases where generator said "I don't know"
            outcome = "incorrect_answer"
            outcome_utility = float(UTILITIES['incorrect_answer'])
            r['graph_action'] = "answer"
            r['graph_correct'] = False
            r['abstain_correct'] = None # Not applicable

        # Prep justification (remains the same, uses 'outcome')
        m = r.get("evidence_metrics", {})
        r['graph_justification'] = get_ce_justification(
            tuple(r.get('joint_action', DEFAULT_POLICY)),
            outcome, # Pass the determined outcome (abstain, correct_answer, incorrect_answer)
            signal_bin=r.get('signal_bin'),
            metrics=m,
            reliability=m.get('reliability'),
            degree=m.get('degree'),
            evidence=r.get('evidence')
        )

        net_utility = float(outcome_utility - cost)
        r['net_utility'] = net_utility
        all_net_utilities.append(net_utility)
        processed_results.append(r) # Append the fully processed result

    # --- Summary Metrics ---
    # `evaluate_batch` uses 'graph_action' and 'graph_correct' which are set correctly above
    baseline_acc = sum(r.get('baseline_correct', False) for r in processed_results) / max(1, len(processed_results))
    graph_eval = evaluate_batch(processed_results, mode="graph")
    avg_net_utility = float(np.mean(all_net_utilities)) if all_net_utilities else 0.0

    # Abstain diagnostics
    num_correct_abstain = sum(1 for r in processed_results if r.get('graph_action') == 'abstain' and r.get('abstain_correct') is True)
    num_wrong_abstain = sum(1 for r in processed_results if r.get('graph_action') == 'abstain' and r.get('abstain_correct') is False)
    num_total_abstain = num_correct_abstain + num_wrong_abstain

    print("\n=== 4-Model Comparison Results ===")
    print(f"{'Model':<20} | {'Acc (Overall)':<15} | {'Abstain Rate':<15} | {'Acc (Answered)':<15}")
    print("-" * 75)
    print(f"{'1. Baseline':<20} | {acc_base:<15.3%} | {abs_base:<15.2%} | {acc_ans_base:<15.3%}")
    print(f"{'2. GraphRAG':<20} | {acc_gr:<15.3%} | {abs_gr:<15.2%} | {acc_ans_gr:<15.3%}")
    print(f"{'3. Baseline + CE':<20} | {acc_bce:<15.3%} | {abs_bce:<15.2%} | {acc_ans_bce:<15.3%}")
    print(f"{'4. GRACE':<20} | {acc_grace:<15.3%} | {abs_grace:<15.2%} | {acc_ans_grace:<15.3%}")
    
    print("\n=== GRACE Detailed Analysis ===")
    print(f"Average Net Utility (CE): {avg_net_utility:.3f}")
    print(f"Abstains: {num_total_abstain}  |  Correct: {num_correct_abstain}  |  Wrong: {num_wrong_abstain}")
    if num_total_abstain > 0:
        print(f"Correct Abstain Rate: {num_correct_abstain / num_total_abstain:.2%}")

    # --- Overhead Summary ---
    if total_samples_measured > 0:
        avg_signal = (total_signal_time / total_samples_measured) * 1000 # ms
        avg_policy = (total_policy_time / total_samples_measured) * 1000 # ms
        avg_exec = (total_execution_time / total_samples_measured) * 1000 # ms
        avg_overhead = avg_signal + avg_policy
        
        print("\n=== Efficiency / Overhead Analysis ===")
        print(f"Avg Signal Extraction Time: {avg_signal:.2f} ms")
        print(f"Avg Policy Lookup Time    : {avg_policy:.2f} ms")
        print(f"Avg Total Overhead        : {avg_overhead:.2f} ms")
        print(f"Avg Execution Time        : {avg_exec:.2f} ms")
        if avg_exec > 0:
            print(f"Overhead Ratio            : {(avg_overhead / avg_exec):.2%}")

    def _py(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        return obj

    serializable_results = json.loads(json.dumps(processed_results, default=_py, ensure_ascii=False))
    with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"Saved final CE results to {FINAL_RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphTrustQA CE Evaluation")
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help=f'Beam width for graph retrieval (default: {DEFAULT_WIDTH})')
    parser.add_argument('--calib_size', type=int, default=100,
                        help='Number of samples for calibration/learning')
    parser.add_argument('--eval_size', type=int, default=100,
                        help='Number of samples for final evaluation')
    parser.add_argument('--force-calib', action='store_true',
                        help='Force re-running the simulation (Phase 1) even if data exists')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip Phase 2 (final evaluation)')

    args = parser.parse_args()
    print(f"Running CE evaluation with calib_size={args.calib_size}, eval_size={args.eval_size}, width={args.width}")
    if args.force_calib: print("Forcing Phase 1 re-run.")
    if args.skip_eval: print("Skipping Phase 2 evaluation.")
    main(args)