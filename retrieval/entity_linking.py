# retrieval/entity_linking.py
from rapidfuzz import process
from utils.text_normalization import TextNormalizer
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def _important_tokens(text_norm):
    """Extract important tokens (nouns/proper nouns)"""
    try:
        tokens = nltk.word_tokenize(text_norm)
        tagged_tokens = nltk.pos_tag(tokens)
        # Keep Nouns (NN, NNS), Proper Nouns (NNP, NNPS)
        # Exclude stopwords and short tokens
        important = [
            word for word, tag in tagged_tokens
            if tag in ['NN', 'NNS', 'NNP', 'NNPS']
            and word not in STOPWORDS
            and len(word) > 2 # Keep slightly shorter nouns now
        ]
        return important
    except LookupError:
        print("NLTK tokenizers/taggers not found. Downloading...")
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        # Fallback to old method on first run after download
        return [t for t in text_norm.split() if len(t) >= 4 and t not in STOPWORDS]

def _build_label_index(G):
    # Map normalized page labels -> node id
    label_to_id_raw = {d.get("label", nid): nid for nid, d in G.nodes(data=True) if d.get("type") == "Document"}
    return TextNormalizer.normalize_dict_keys(label_to_id_raw)

def _best_from_candidates(query_norm, candidates_norm):
    # Use a reasonably high cutoff to avoid bad matches
    res = process.extractOne(query_norm, candidates_norm, score_cutoff=78)
    if res:
        cand_norm, score, _ = res
        return cand_norm, score
    return None, None

def pick_start_node_from_question(G, question, k=5, debug=False):
    """
    Robust entity linking:
    1) If the question EXACTLY matches a question-node (normalized equality), use it.
    2) Otherwise, match against PAGE TITLES only.
       - Filter candidates by token overlap first (important tokens).
       - Then apply fuzzy match with a higher cutoff.
    """
    q_norm = TextNormalizer.normalize_text(question)

    # === Pass 0: exact match to a question-node (only if identical after normalization)
    q_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "Question"]
    q_nodes_norm = set(TextNormalizer.normalize_list(q_nodes))
    if q_norm in q_nodes_norm:
        if debug: print(f"[EntityLink] Exact question-node match: {q_norm}")
        # Return the original ID (already normalized in graph)
        return q_norm

    # === Build page-label index
    label_to_id = _build_label_index(G)
    labels_norm = list(label_to_id.keys())

    # === Token-overlap filter (shrink search space)
    tokens = set(_important_tokens(q_norm))
    if tokens:
        scored = []
        for lab in labels_norm:
            lab_tokens = set(lab.split())
            overlap = len(tokens & lab_tokens)
            if overlap > 0:
                scored.append((overlap, lab))
        # take top-K by overlap (wider if few candidates)
        scored.sort(reverse=True, key=lambda x: x[0])
        cand_pool = [lab for _, lab in scored[:max(50, 5*len(tokens))]] or labels_norm
    else:
        cand_pool = labels_norm

    cand_norm, score = _best_from_candidates(q_norm, cand_pool)
    if cand_norm:
        if debug: print(f"[EntityLink] Matched page label: {cand_norm} (score={score})")
        return label_to_id[cand_norm]

    if debug: print(f"[EntityLink] No match for: {question}")
    return None
