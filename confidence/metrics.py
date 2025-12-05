# # confidence/metrics.py
# import numpy as np
# import joblib
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV  # Added for better probability calibration

# # --- Constants for saving the model ---
# MODEL_PATH = "confidence_model.joblib"
# SCALER_PATH = "confidence_scaler.joblib"
# VEC_PATH = "confidence_vectorizer.joblib"

# class ConfidenceModel:
#     def __init__(self, st_model_name="all-MiniLM-L6-v2"):
#         print(f"Loading SentenceTransformer model {st_model_name}...")
#         self.st_model = SentenceTransformer(st_model_name)
#         print("SentenceTransformer model loaded.")
        
#         # Load the trained models
#         self.model = None
#         self.scaler = None
#         self.vectorizer = None
#         self.load_trained_model()
    
#     def load_trained_model(self):
#         """Loads the trained LR model, scaler, and vectorizer."""
#         try:
#             self.model = joblib.load(MODEL_PATH)
#             self.scaler = joblib.load(SCALER_PATH)
#             self.vectorizer = joblib.load(VEC_PATH)
#             print("Trained confidence model (LR, Scaler, Vec) loaded successfully.")
#             return True
#         except FileNotFoundError:
#             print("Warning: No pre-trained confidence model found.")
#             print("-> Path ranking will use reliability heuristic until model is trained.")
#             self.model = None
#             self.scaler = None
#             self.vectorizer = None
#             return False

#     def _compute_semantic_coherence(self, G, path):
#         """
#         Computes semantic coherence using sentence embeddings.
#         Falls back to lexical overlap if embedding fails.
#         """
#         if len(path) < 2:
#             return 1.0
#         labels = [G.nodes[n].get('label', '') for n in path]
#         if not any(labels):
#             return 0.5

#         try:
#             embs = self.st_model.encode(labels, convert_to_tensor=False)
#             sim_matrix = cosine_similarity(embs)
#             indices = np.triu_indices(len(embs), k=1)
#             if len(indices[0]) == 0:
#                 return 1.0
#             mean_sim = np.mean(sim_matrix[indices])
#             return float(mean_sim)
#         except Exception as e:
#             print(f"Warning: Embedding failed in coherence: {e}. Using lexical fallback.")
#             # Fallback: lexical overlap (Jaccard-like)
#             try:
#                 pairs = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))]
#                 overlaps = []
#                 for a, b in pairs:
#                     set_a = set(a.lower().split())
#                     set_b = set(b.lower().split())
#                     if set_a and set_b:
#                         overlaps.append(len(set_a & set_b) / len(set_a | set_b))
#                 return np.mean(overlaps) if overlaps else 0.5
#             except:
#                 return 0.5

#     def get_raw_features(self, G, path):
#         """
#         Computes the four raw features for a given path.
#         Returns None if path is invalid.
#         """
#         if not path or len(path) == 0:
#             return None

#         # 1. Length Score: penalize long paths
#         length_score = np.exp(-0.5 * (len(path) - 1))
        
#         # 2. Diversity Score: variety of edge relations
#         if len(path) < 2:
#             diversity_score = 1.0
#         else:
#             relations = set()
#             for u, v in zip(path[:-1], path[1:]):
#                 edge_data = G.get_edge_data(u, v)
#                 if edge_data:
#                     relations.add(edge_data.get('relation', 'unknown'))
#             diversity_score = len(relations) / (len(path) - 1) if relations else 0.0
            
#         # 3. Reliability Score: average node reliability
#         reliabilities = []
#         for n in path:
#             rel = G.nodes[n].get('reliability')
#             if rel is not None:
#                 reliabilities.append(rel)
#         reliability_score = np.mean(reliabilities) if reliabilities else 0.5
        
#         # 4. Semantic Coherence Score
#         coherence_score = self._compute_semantic_coherence(G, path)

#         return {
#             "length_score": float(length_score),
#             "diversity_score": float(diversity_score),
#             "reliability_score": float(reliability_score),
#             "coherence_score": float(coherence_score)
#         }

#     def train(self, results_data):
#         """
#         Trains the logistic regression model on the results data.
#         Uses calibrated classifier for better probability estimates.
#         """
#         print("Training confidence model...")
        
#         # 1. Prepare training data (X) and labels (y)
#         X_dicts = []
#         y_labels = []
#         for r in results_data:
#             if r.get('raw_features') is not None and r.get('is_correct') is not None:
#                 X_dicts.append(r['raw_features'])
#                 y_labels.append(int(r['is_correct']))  # Ensure int

#         if not X_dicts:
#             print("Error: No valid data to train on (all features were None).")
#             return

#         # 2. Fit vectorizer and scaler
#         self.vectorizer = DictVectorizer(sparse=False)
#         X = self.vectorizer.fit_transform(X_dicts)
#         joblib.dump(self.vectorizer, VEC_PATH)
        
#         self.scaler = StandardScaler()
#         X_scaled = self.scaler.fit_transform(X)
#         joblib.dump(self.scaler, SCALER_PATH)

#         # 3. Train calibrated logistic regression
#         base_model = LogisticRegression(class_weight='balanced', max_iter=1000)
#         self.model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
#         self.model.fit(X_scaled, y_labels)
#         joblib.dump(self.model, MODEL_PATH)

#         print("Confidence model trained and saved with calibration.")
        
#         try:
#             # Use base estimator's coefficients
#             importance = self.model.calibrated_classifiers_[0].base_estimator.coef_[0]
#             features = self.vectorizer.get_feature_names_out()
#             print("--- Feature Importance ---")
#             for feat, imp in sorted(zip(features, importance), key=lambda x: -abs(x[1])):
#                 print(f"{feat:<20}: {imp:.4f}")
#             print("--------------------------")
#         except Exception as e:
#             print(f"Could not show feature importance: {e}")
        
#         print("Reloading trained model components after training...")
#         self.load_trained_model()

#     def predict_confidence(self, raw_features_dict):
#         """
#         Predicts the confidence (probability of being correct)
#         for a single set of features.
#         """
#         if raw_features_dict is None:
#             return 0.0
             
#         if self.model is None or self.scaler is None or self.vectorizer is None:
#             print("Warning: Confidence model not loaded/trained. Using feature average as fallback.")
#             # Fallback: average of all available scores
#             valid_scores = [v for v in raw_features_dict.values() if isinstance(v, (int, float))]
#             return np.mean(valid_scores) if valid_scores else 0.5
            
#         try:
#             X_dict = [raw_features_dict]    
#             X = self.vectorizer.transform(X_dict)
#             X_scaled = self.scaler.transform(X)
            
#             # Predict calibrated probability
#             prob = self.model.predict_proba(X_scaled)[0]
#             return float(prob[1])  # Probability of True (correct)
#         except Exception as e:
#             print(f"Error during confidence prediction: {e}")
#             # Final fallback
#             return np.mean(list(raw_features_dict.values()))