# utils/policy_utils.py
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

    rel_bin = bin_val(reliability, t('reliability'))
    deg_bin = bin_val(degree,      t('degree'))
    len_bin = bin_val(path_len,    t('len'))
    coh_bin = bin_val(coherence,   t('coherence'))
    div_bin = bin_val(diversity,   t('diversity'))

    return f"rel:{rel_bin}_deg:{deg_bin}_len:{len_bin}_coh:{coh_bin}_div:{div_bin}"