import networkx as nx
from config import MAX_HOPS_LIMIT, STOP_THRESHOLD_SHALLOW, STOP_THRESHOLD_DEEP

def retrieve_paths(G, start_entity, action, max_width=15):

    # Determine parameters from the action
    if action == 'retrieve_shallow':
        stop_threshold = STOP_THRESHOLD_SHALLOW
    elif action == 'retrieve_deep':
        stop_threshold = STOP_THRESHOLD_DEEP
    else:
        # Fallback for unknown action
        stop_threshold = 0.5

    paths = []
    # Frontier stores tuples: (current_node, current_path_list)
    frontier = [(start_entity, [start_entity])]

    # Run up to MAX_HOPS_LIMIT, but check for early stopping
    for current_hop in range(MAX_HOPS_LIMIT):
        next_frontier = []
        processed_in_level = set() # Avoid cycles within the same hop level expansion

        for node, path in frontier:
            
            # Safely get neighbors using bidirectional traversal
            # Combine successors and predecessors because the graph is sparse
            neighbors = []
            if node in G:
                try:
                    succ = list(G.successors(node))
                    pred = list(G.predecessors(node))
                    # Filter out nodes already in the current path to avoid cycles
                    neighbors = [n for n in (succ + pred) if n not in path]
                except Exception:
                    neighbors = []
            
            node_degree = len(neighbors)
            current_width = min(max_width, node_degree)

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

        if not next_frontier:
            break
        
        # EARLY STOPPING CHECK
        # Calculate scores for all current paths
        # If we find a good enough path, we stop to save time
        best_path_in_batch = select_best_path(G, paths)
        if best_path_in_batch:
            # Re-calculate score to check against threshold (select_best_path returns the path, not score)
            try:
                avg_rel = sum(G.nodes[n]['reliability'] for n in best_path_in_batch) / len(best_path_in_batch)
                path_score = avg_rel + (len(best_path_in_batch) * 0.1)
                
                if path_score >= stop_threshold:
                    # Found a good path! Stop expansion.
                    break
            except KeyError:
                pass
        
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
            # Simple score: reliability PLUS a bonus for length to encourage multi-hop
            # This prefers high-reliability, DEEPER paths
            score = avg_reliability + (len(path) * 0.1) 
            
            if score > best_score:
                best_score = score
                best_path = path
        except KeyError:
            # Path contains a node not in G, which shouldn't happen but good to guard
            continue 
            
    return best_path