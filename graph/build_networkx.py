# graph/build_networkx.py
import networkx as nx
import csv

def build_graph(nodes_file='data/nodes.csv', edges_file='data/edges.csv'):
    G = nx.DiGraph()

    with open(nodes_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['id'] 
            G.add_node(
                node_id,
                label=row['label'],
                type=row['type'],
                reliability=float(row['reliability'])
            )

    # Load edges
    with open(edges_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['src']
            dst = row['dst']
            
            # Add edge only if both nodes exist (handles potential orphans)
            if src in G and dst in G:
                G.add_edge(src, dst, relation=row['relation'], weight=float(row['weight']))
            else:
                # This warning can be noisy but is good for debugging
                # print(f"Warning: Skipping edge ({src}, {dst}) - one or both nodes not in graph.")
                pass

    # Compute PageRank for Reliability
    try:
        # Use a personalized PageRank or standard? Standard is fine for general node importance.
        # We can use the 'weight' attribute for weighted PageRank.
        pr = nx.pagerank(G, weight='weight')
        
        # Update node reliability with PageRank scores
        # Normalize or scale if needed, but raw probabilities are fine for relative ranking.
        # To make them more "human readable" as reliability, maybe scale them?
        # Or just use them as is. The quantile binning handles the scale.
        for node_id, score in pr.items():
            G.nodes[node_id]['reliability'] = score
            
        print("PageRank computed and assigned to 'reliability' attribute.")
    except Exception as e:
        print(f"Warning: PageRank computation failed: {e}")

    return G

if __name__ == "__main__":
    G = build_graph()
    print(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")