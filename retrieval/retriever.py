# retrieval/retriever.py
import networkx as nx

def retrieve_paths(G, start_entity, action, max_width=15):
    """
    Retrieves paths using BFS based on a mediator-recommended action.
    
    action: 'retrieve_shallow' (hops=1) or 'retrieve_deep' (hops=2)
    """
    
    # 1. Determine parameters from the action
    if action == 'retrieve_shallow':
        hops = 1
    elif action == 'retrieve_deep':
        hops = 2
    else:
        # Fallback for unknown action
        hops = 1

    paths = []
    # Frontier stores tuples: (current_node, current_path_list)
    frontier = [(start_entity, [start_entity])]

    for current_hop in range(hops):
        next_frontier = []
        processed_in_level = set() # Avoid cycles within the same hop level expansion

        for node, path in frontier:
            
            # Safely get neighbors using bidirectional traversal
            # We combine successors and predecessors because the graph is sparse
            neighbors = []
            if node in G:
                try:
                    succ = list(G.successors(node))
                    pred = list(G.predecessors(node))
                    # Filter out nodes already in the current path to avoid cycles
                    neighbors = [n for n in (succ + pred) if n not in path]
                except Exception:
                    neighbors = []
            
            # --- Adaptive Width Logic ---
            node_degree = len(neighbors)
            current_width = min(max_width, node_degree)
            # --------------------------

            # Sort neighbors for determinism (heuristic: edge weight if available)
            try:
                sorted_neighbors = sorted(
                    neighbors, 
                    key=lambda n: G.get_edge_data(node, n, default={}).get('weight', 0.0) if G.has_edge(node, n) else G.get_edge_data(n, node, default={}).get('weight', 0.0), 
                    reverse=True
                )
            except Exception:
                sorted_neighbors = sorted(neighbors)

            for i, neighbor in enumerate(sorted_neighbors):
                if i >= current_width: # Apply the calculated width limit
                    break

                # Avoid re-expanding the same node multiple times within this level
                if (current_hop, neighbor) in processed_in_level:
                        continue

                new_path = path + [neighbor]
                paths.append(new_path)
                next_frontier.append((neighbor, new_path))
                processed_in_level.add((current_hop, neighbor))

        if not next_frontier: # Stop if we can't expand further
            break
        frontier = next_frontier

    return paths

def select_best_path(G, paths):
    """
    Selects the best path from a list of paths.
    Heuristic: Prioritize shortest path with highest avg reliability.
    """
    if not paths:
        return None

    best_path = None
    best_score = -1.0

    for path in paths:
        if not path: continue
        try:
            # Calculate average reliability of nodes in the path
            avg_reliability = sum(G.nodes[n]['reliability'] for n in path) / len(path)
            # Simple score: reliability minus a penalty for length
            # This prefers high-reliability, short paths
            score = avg_reliability - (len(path) * 0.05) 
            
            if score > best_score:
                best_score = score
                best_path = path
        except KeyError:
            # Path contains a node not in G, which shouldn't happen but good to guard
            continue 
            
    return best_path