import sys
sys.path.insert(0, '.')

from graph.build_networkx import build_graph
from retrieval.entity_linking import pick_start_node_from_question
from datasets import load_dataset

print("Loading graph...")
G = build_graph()

print("\nLoading HotpotQA validation dataset...")
ds = load_dataset("hotpot_qa", "distractor")["validation"]

print("\nFinding questions with entities in the graph...\n")

found = 0
for i, ex in enumerate(ds):
    if found >= 5:
        break
    
    question = ex["question"]
    start_node = pick_start_node_from_question(G, question)
    
    if start_node:
        print(f" Question: {question}")
        print(f"  Entity: {G.nodes[start_node]['label']}")
        print(f"  Gold Answer: {ex['answer']}")
        print()
        found += 1

if found == 0:
    print("No questions found with entities in the graph.")