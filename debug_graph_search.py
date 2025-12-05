import sys
sys.path.insert(0, '.')

from graph.build_networkx import build_graph

print("Loading graph...")
G = build_graph()
print(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Search for "Doctor Strange"
search_terms = ["doctor strange", "doctor", "strange"]

for term in search_terms:
    matches = [n for n in G.nodes() if term in G.nodes[n].get('label', '').lower()]
    print(f"\nSearching for '{term}': {len(matches)} matches")
    for n in matches[:5]:
        print(f"  - {G.nodes[n].get('label')} (ID: {n})")
