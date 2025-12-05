import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from graph.build_networkx import build_graph
from retrieval.entity_linking import pick_start_node_from_question

def debug_system():
    print("--- Debugging System ---")
    
    # 1. Check Graph Loading
    print("Loading Graph...")
    try:
        G = build_graph()
        print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        if len(G.nodes) == 0:
            print("CRITICAL ERROR: Graph is empty!")
            return
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load graph: {e}")
        return

    # 2. Check Entity Linking
    test_questions = [
        "Which of these talented men was incarcerated?",
        "Where did the descendants of the group of black Indians settle?",
        "Based on a True Story is an album by which country music star?"
    ]
    
    print("\nTesting Entity Linking...")
    for q in test_questions:
        try:
            start_node = pick_start_node_from_question(G, q)
            print(f"Q: '{q}' -> Start Node: {start_node}")
            if start_node:
                print(f"   Node Data: {G.nodes[start_node]}")
                succ = list(G.successors(start_node))
                pred = list(G.predecessors(start_node))
                print(f"   Successors: {len(succ)} {succ[:5]}")
                print(f"   Predecessors: {len(pred)} {pred[:5]}")
            else:
                print("   FAILED to find start node.")
        except Exception as e:
            print(f"   ERROR during linking: {e}")

if __name__ == "__main__":
    debug_system()
