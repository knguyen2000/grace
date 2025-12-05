import networkx as nx
import csv

def build_graph(nodes_file='data/nodes.csv', edges_file='data/edges.csv'):
    G = nx.DiGraph()

    with open(nodes_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row['id']
            # Load all attributes from the row
            attributes = {k: v for k, v in row.items() if k != 'id'}
            
            # Ensure critical types are correct
            if 'reliability' in attributes:
                attributes['reliability'] = float(attributes['reliability'])
            
            G.add_node(node_id, **attributes)

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
        # Use 'weight' attribute for weighted PageRank.
        pr = nx.pagerank(G, weight='weight')
        
        # Update node reliability with PageRank scores
        # We need to normalize these scores to [0, 1] so they match the thresholds in config.py (0.7, 0.85)
        # Otherwise raw PageRank (which sums to 1.0) will be way too small.
        
        pr_values = list(pr.values())
        if pr_values:
            min_pr = min(pr_values)
            max_pr = max(pr_values)
            range_pr = max_pr - min_pr if max_pr > min_pr else 1.0
            
            for node_id, score in pr.items():
                # Protect Question nodes: they should always be trusted (reliability=1.0)
                if G.nodes[node_id].get('type') == 'Question':
                    G.nodes[node_id]['reliability'] = 1.0
                else:
                    # Min-Max Normalization
                    normalized_score = (score - min_pr) / range_pr
                    G.nodes[node_id]['reliability'] = normalized_score
        else:
             print("Warning: PageRank returned empty dict.")
            
        print("PageRank computed and assigned to 'reliability' attribute (Questions fixed at 1.0).")
    except Exception as e:
        print(f"Warning: PageRank computation failed: {e}")

    return G

if __name__ == "__main__":
    G = build_graph()
    print(f"Graph loaded with {len(G.nodes())} nodes and {len(G.edges())} edges.")