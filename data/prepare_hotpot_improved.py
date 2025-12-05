import csv
import random
import sys
from pathlib import Path
from datasets import load_dataset
from collections import defaultdict
import re
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.text_normalization import TextNormalizer

# Try to import spaCy for better NLP
try:
    import spacy
    HAS_SPACY = True
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Loaded spaCy model: en_core_web_sm")
    except OSError:
        print("Warning: spaCy model 'en_core_web_sm' not found.")
        print("Install with: python -m spacy download en_core_web_sm")
        HAS_SPACY = False
        nlp = None
except ImportError:
    print("Warning: spaCy not installed. Install with: pip install spacy")
    HAS_SPACY = False
    nlp = None

# Try to import sentence transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
    try:
        st_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Loaded sentence transformer: all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Warning: Could not load sentence transformer: {e}")
        HAS_SENTENCE_TRANSFORMERS = False
        st_model = None
except ImportError:
    print("Warning: sentence-transformers not installed.")
    HAS_SENTENCE_TRANSFORMERS = False
    st_model = None

OUT_NODES = "data/nodes.csv"
OUT_EDGES = "data/edges.csv"
SAMPLE_SIZE = 5000

_normalize_cache = {}
_text_normalizer = TextNormalizer()

def cached_normalize_text(text):
    """Cached version of TextNormalizer.normalize_text"""
    if text not in _normalize_cache:
        _normalize_cache[text] = _text_normalizer.normalize_text(text)
    return _normalize_cache[text]

def extract_svo(doc):
    """
    Extract Subject-Verb-Object triples from a spaCy Doc
    Returns list of (subject, verb, object) tuples
    """
    triples = []
    for token in doc:
        # Look for main verbs
        if token.pos_ == "VERB":
            subj = None
            obj = None
            
            # Find subject
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                    subj = child
                    break
            
            # Find object
            for child in token.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child
                    break
            
            if subj and obj:
                # Expand to full phrases (simple version)
                subj_text = " ".join([t.text for t in subj.subtree]).strip()
                obj_text = " ".join([t.text for t in obj.subtree]).strip()
                verb_text = token.lemma_
                triples.append((subj_text, verb_text, obj_text))
    
    return triples

def extract_entities_and_concepts(doc):
    """
    Extract Named Entities and important Noun Chunks (Concepts)
    Returns list of (text, type, label) tuples
    """
    items = []
    
    # 1. Named Entities
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'DATE', 'MONEY']:
            items.append((ent.text, 'Entity', ent.label_))
            
    # 2. Noun Chunks (Concepts) - only if not overlapping with entities
    # This is a simplification; in production, check overlap indices
    ent_texts = {i[0] for i in items}
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        # Filter out pronouns and stop words
        if len(text.split()) < 4 and text not in ent_texts and not chunk.root.is_stop:
             items.append((text, 'Concept', 'NOUN_PHRASE'))
             
    return items

def compute_similarity(text1, text2):
    """Compute cosine similarity between two texts"""
    if not (HAS_SENTENCE_TRANSFORMERS and st_model):
        return 0.0
    
    embeddings = st_model.encode([text1, text2])
    from numpy import dot
    from numpy.linalg import norm
    return dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))

def main():
    ds = load_dataset("hotpot_qa", "distractor")["validation"]
    idxs = list(range(min(len(ds), SAMPLE_SIZE)))

    nodes = {}
    edges = defaultdict(lambda: {"relation": "supports", "weight": 0.9, "count": 0})
    
    def add_node(label, node_type, reliability=0.8, metadata=None):
        # Create a unique ID based on type and label to avoid collisions between same-named entities and docs
        # For sentences, we'll need a hash or unique ID passed in, but for now let's use label if unique enough
        # Actually, for Sentences, the label is the text, which might be long. Let's hash it or use a counter?
        # For this implementation, we'll use normalized label as ID for Entities/Docs, but handle Sentences differently.
        
        if node_type == "Sentence":
            # Use a hash of the text for ID
            import hashlib
            node_id = f"sent_{hashlib.md5(label.encode()).hexdigest()[:12]}"
        else:
            node_id = cached_normalize_text(label)
        
        if not node_id:
            return None
        
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "reliability": reliability
            }
            if metadata:
                nodes[node_id].update(metadata)
        return node_id

    def add_edge(src, dst, relation, weight=1.0):
        if not src or not dst or src == dst:
            return
        
        key = (src, dst, relation)
        
        edges[key].update({
            "src": src,
            "dst": dst,
            "relation": relation,
            "weight": weight,
            "count": edges[key].get("count", 0) + 1
        })

    print(f"Processing {len(idxs)} samples with KG²RAG strategy...")

    for idx_count, i in enumerate(idxs):
        ex = ds[i]
        q = ex["question"].strip()
        ctx = ex["context"] # List of [title, sentences]
        
        if idx_count < 3:
            print(f"\n=== SAMPLE {i} ===")
            print(f"Question: {q}")

        # 1. Question Node
        q_node_id = add_node(q, "Question", reliability=1.0)
        
        # 2. Process Context (Documents -> Sentences)
        # We need to find relevant sentences first to prune the graph
        
        all_sentences = [] # (title, sent_index, text)
        
        # Handle different context formats
        context_items = []
        if isinstance(ctx, dict):
            titles = ctx.get("title", [])
            sentences_list = ctx.get("sentences", [])
            for t, s in zip(titles, sentences_list):
                context_items.append((t, s))
        elif isinstance(ctx, list):
             for item in ctx:
                 if len(item) >= 2:
                     context_items.append((item[0], item[1]))

        for title, sentences in context_items:
            for s_idx, sent in enumerate(sentences):
                all_sentences.append((title, s_idx, sent))
        
        # Relevance Pruning: Encode Question and all Sentences
        if HAS_SENTENCE_TRANSFORMERS and st_model:
            sent_texts = [s[2] for s in all_sentences]
            if not sent_texts: continue
            
            q_emb = st_model.encode(q)
            s_embs = st_model.encode(sent_texts)
            
            # Calculate similarities
            sims = np.dot(s_embs, q_emb) / (np.linalg.norm(s_embs, axis=1) * np.linalg.norm(q_emb))
            
            # Keep top-k or threshold
            # For KG^2RAG, we want a good seed set but also context. 
            # Let's keep top 20 sentences + any sentences from "Supporting Facts" (Gold)
            top_k_indices = np.argsort(sims)[-20:]
            relevant_indices = set(top_k_indices)
        else:
            # Fallback: keep all (or simple word overlap)
            relevant_indices = set(range(len(all_sentences)))

        # Always include gold supporting facts
        sp_facts = set()
        sp = ex["supporting_facts"]
        for title, sent_idx in zip(sp["title"], sp["sent_id"]):
            sp_facts.add((title, sent_idx))
            
        # Build the Graph
        prev_sent_node_id = None
        current_doc_node_id = None
        
        # Group by document to preserve order
        doc_sentences = defaultdict(list)
        for idx, (title, s_idx, sent) in enumerate(all_sentences):
            doc_sentences[title].append((idx, s_idx, sent))
            
        for title, sents in doc_sentences.items():
            # Add Document Node
            doc_node_id = add_node(title, "Document")
            prev_sent_id = None
            
            for global_idx, s_idx, sent_text in sents:
                # Check relevance
                is_relevant = global_idx in relevant_indices
                is_gold = (title, s_idx) in sp_facts
                
                if not (is_relevant or is_gold):
                    continue
                
                # Add Sentence Node (Chunk)
                sent_node_id = add_node(sent_text, "Sentence", 
                                      metadata={"is_gold": is_gold, "doc_title": title})
                
                # Edge: Document -> Sentence
                add_edge(doc_node_id, sent_node_id, "CONTAINS")
                
                # Edge: Sentence -> Next Sentence (Flow)
                if prev_sent_id:
                    add_edge(prev_sent_id, sent_node_id, "NEXT")
                prev_sent_id = sent_node_id
                
                # Edge: Question -> Sentence (if highly relevant)
                if is_relevant and HAS_SENTENCE_TRANSFORMERS:
                    # Add edge if similarity is high enough
                    # We already filtered by top-k, so these are "candidate" chunks
                    add_edge(q_node_id, sent_node_id, "CANDIDATE_CHUNK", weight=0.8)

                # NLP Processing for Sentence
                if HAS_SPACY and nlp:
                    doc = nlp(sent_text)
                    
                    # Extract Entities & Concepts
                    items = extract_entities_and_concepts(doc)
                    for text, type_, label in items:
                        ent_node_id = add_node(text, type_, metadata={"entity_type": label})
                        # Edge: Sentence -> Entity
                        add_edge(sent_node_id, ent_node_id, "MENTIONS")
                    
                    # Extract SVO and link Entities
                    triples = extract_svo(doc)
                    for subj, verb, obj in triples:
                        # Try to map subj/obj to existing nodes
                        subj_id = cached_normalize_text(subj)
                        obj_id = cached_normalize_text(obj)
                        
                        # Only add edge if both nodes exist (were extracted as entities/concepts)
                        if subj_id in nodes and obj_id in nodes:
                            add_edge(subj_id, obj_id, verb.upper()) # Use verb as relation type

        if (idx_count + 1) % 100 == 0:
            print(f"Processed {idx_count + 1}/{len(idxs)} samples...")

    # Write output
    print("\nWriting nodes...")
    node_fields = ["id", "label", "type", "reliability", "doc_title", "is_gold", "label"] # Union of all keys
    # Get all unique keys
    all_keys = set()
    for n in nodes.values():
        all_keys.update(n.keys())
    
    with open(OUT_NODES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(list(all_keys)))
        w.writeheader()
        for node_id in sorted(nodes.keys()):
            w.writerow(nodes[node_id])

    print("Writing edges...")
    with open(OUT_EDGES, "w", newline="", encoding="utf-8") as f:
        # Get all unique keys for edges
        all_edge_keys = set()
        for e in edges.values():
            all_edge_keys.update(e.keys())
            
        w = csv.DictWriter(f, fieldnames=sorted(list(all_edge_keys)))
        w.writeheader()
        for key in sorted(edges.keys()):
            w.writerow(edges[key])

    print(f"\nWrote {len(nodes)} nodes -> {OUT_NODES}")
    print(f"Wrote {len(edges)} edges -> {OUT_EDGES}")

if __name__ == "__main__":
    main()

