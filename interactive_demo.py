import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph.build_networkx import build_graph
from models.generator import Generator
from models.verifier import NliVerifier
from policy.mediator import CeMediator
from retrieval.entity_linking import pick_start_node_from_question
from retrieval.retriever import retrieve_paths, select_best_path
from eval.run_evaluation import execute_joint_action, compute_evidence_metrics
from config import CE_POLICY_FILE, DEFAULT_WIDTH

def main():
    print("--- GRACE Interactive Demo ---")
    print("Loading Graph, Models, and Policy... (this may take a minute)")
    
    G = build_graph()
    print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges.")
    
    gen = Generator()
    verifier = NliVerifier()
    
    if not os.path.exists(CE_POLICY_FILE):
        print(f"Error: Policy file {CE_POLICY_FILE} not found. Run evaluation first to generate it.")
        return

    mediator = CeMediator(CE_POLICY_FILE)
    print("\nSystem Ready! Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nUser Question: ").strip()
        if not question:
            continue
        if question.lower() in ['exit', 'quit']:
            break

        print("-" * 50)
        print(f"Processing: {question}")
        
        # 1. Entity Linking
        start_node_id = pick_start_node_from_question(G, question)
        print(f"Entity Linking: Found start node '{start_node_id}'")

        # 2. Signal Extraction (Probing)
        path_metrics = {"path_len": 0, "diversity": 0, "coherence": 0.0}
        if start_node_id and start_node_id in G.nodes:
            try:
                paths = retrieve_paths(G, start_node_id, action='retrieve_shallow', max_width=DEFAULT_WIDTH)
                best_path = select_best_path(G, paths) if paths else None
                path_metrics = compute_evidence_metrics(G, best_path or [])
            except Exception as e:
                print(f"Warning: Probing failed: {e}")

        print(f"Graph Signals: {path_metrics}")

        # 3. Mediator Decision
        signal_bin = mediator.get_signal_bin(G, start_node_id, path_metrics)
        joint_action = mediator.get_joint_action(G, start_node_id, path_metrics)
        print(f"Mediator State: {signal_bin}")
        print(f"Mediator Action: {joint_action}")

        # 4. Execution
        print("Executing Action...")
        t0 = time.time()
        answer, evidence, v_label, cost, final_path, final_metrics = execute_joint_action(
            G, gen, verifier, question, start_node_id, joint_action, width=DEFAULT_WIDTH
        )
        duration = time.time() - t0

        # 5. Result
        print("-" * 50)
        if answer:
            print(f"FINAL ANSWER: {answer}")
            print(f"Verification: {v_label}")
        else:
            print("FINAL ANSWER: [Abstained]")
            if joint_action[1] == 'refuse_to_generate':
                print("Reason: Mediator refused (Safety).")
            elif v_label == 'contradicts':
                print("Reason: Verifier detected contradiction.")
            else:
                print("Reason: No evidence or generation failure.")
        
        print(f"Cost: {cost} | Time: {duration:.2f}s")
        if evidence:
            print(f"Evidence Used: {evidence[:100]}...")

if __name__ == "__main__":
    main()
