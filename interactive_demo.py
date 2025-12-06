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
            if final_path:
                print("Visualizing evidence path...")
                visualize_path(G, final_path)

def visualize_path(G, path):
    """
    Visualizes the evidence path using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import textwrap
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return

    if not path or len(path) < 2:
        return

    # Create subgraph
    subgraph = G.subgraph(path).copy()
    
    # Create layout
    pos = {}
    for i, node in enumerate(path):
        pos[node] = (i, 0) # Linear layout left-to-right

    plt.figure(figsize=(16, 6)) # Wider and taller
    
    # Draw nodes
    nx.draw_networkx_nodes(subgraph, pos, node_color='lightblue', node_size=5000, alpha=0.9) # Huge nodes
    
    # Draw edges
    nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True, width=2, arrowsize=30)
    
    # Labels with very tight wrapping
    labels = {}
    for n in subgraph.nodes():
        raw_label = G.nodes[n].get("label", str(n))
        # Wrap very tight (10 chars)
        labels[n] = "\n".join(textwrap.wrap(raw_label, width=10))
        
    nx.draw_networkx_labels(subgraph, pos, labels=labels, font_size=7, font_weight='bold')
    
    # Edge Labels
    edge_labels = {}
    for u, v in zip(path[:-1], path[1:]):
        if G.has_edge(u, v):
            rel = G.edges[u, v].get("relation", "rel")
            edge_labels[(u, v)] = rel
            
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=7, label_pos=0.5, font_color='red', rotate=False)
    
    plt.title("Evidence Path", fontsize=16)
    plt.axis('off')
    
    # Large margins to handle overflow
    plt.margins(x=0.3, y=0.3)
    plt.tight_layout()
    
    # Save to file
    output_file = os.path.abspath("evidence_viz_latest.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Visualization saved to: {output_file}")
    plt.close() # Close figure to free memory

if __name__ == "__main__":
    import networkx as nx # Import here for the visualizer
    main()
