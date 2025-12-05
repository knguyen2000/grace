from config import SIG_THRESHOLDS

def get_signal_bin(G, start_node_id, thresholds, path_metrics):
    """
    thresholds: dict with keys 'reliability','degree','len','coherence','diversity'
    If any key is missing, we fall back to SIG_THRESHOLDS from config.
    
    path_metrics: dict with keys 'path_len', 'coherence', 'diversity'
    """
    if not start_node_id or start_node_id not in G.nodes:
        # Get reliability/degree from node
        reliability = 0.5
        degree      = 0
    else:
        node = G.nodes[start_node_id]
        reliability = node.get('reliability', 0.5)
        degree      = G.degree(start_node_id)

    # Get path metrics from the provided dict, not from node attributes
    path_len   = path_metrics.get('path_len', 0)
    coherence  = path_metrics.get('coherence', 0.0)
    diversity  = path_metrics.get('diversity', 0)

    def t(key):  # thresholds accessor with config fallback
        return thresholds.get(key, SIG_THRESHOLDS[key])

    def bin_val(val, th):
        return "low" if val < th[0] else ("mid" if val < th[1] else "high")

    rel_bin = 0 if reliability < t('reliability')[0] else (1 if reliability < t('reliability')[1] else 2)
    deg_bin = 0 if degree < t('degree')[0] else (1 if degree < t('degree')[1] else 2)
    
    # Quality metrics
    len_bin = 0 if path_len < t('len')[0] else (1 if path_len < t('len')[1] else 2)
    coh_bin = 0 if coherence < t('coherence')[0] else (1 if coherence < t('coherence')[1] else 2)
    div_bin = 0 if diversity < t('diversity')[0] else (1 if diversity < t('diversity')[1] else 2)

    # Dimensionality Reduction: Aggregate into Meta-Signals
    # Trust Score (0-4): Sum of Reliability and Degree bins
    trust_score = rel_bin + deg_bin
    
    # Quality Score (0-6): Sum of Len, Coh, and Div bins
    quality_score = len_bin + coh_bin + div_bin

    # Bin the Meta-Signals into Low/Mid/High
    # Trust: 0-1 -> Low(0), 2 -> Mid(1), 3-4 -> High(2)
    if trust_score <= 1: t_bin = "low"
    elif trust_score <= 2: t_bin = "mid"
    else: t_bin = "high"

    # Quality: 0-2 -> Low(0), 3-4 -> Mid(1), 5-6 -> High(2)
    if quality_score <= 2: q_bin = "low"
    elif quality_score <= 4: q_bin = "mid"
    else: q_bin = "high"

    return f"T:{t_bin}_Q:{q_bin}"