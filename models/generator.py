from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import numpy as np

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        print(f"Loading generator: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("Generator loaded.")

    def generate_from_evidence(self, question, evidence, path=None, confidence=None):

        conf_hint = ""
        if confidence is not None:
            if confidence > 0.8:
                conf_hint = "I am highly confident in this answer.\n"
            elif confidence > 0.5:
                conf_hint = "I am moderately confident.\n"
            else:
                conf_hint = "The evidence is limited, so take this with caution.\n"

        prompt = f"""Answer the question using ONLY the evidence below. 
            If the evidence is insufficient or contradictory, say "I don't know".

            Evidence:
            {evidence}

            Confidence: {conf_hint}
            Question: {question}

            Answer with justification:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=0.0
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

    def generate_with_consistency(self, question, k=5, threshold=0.8):
        """
        Self-Consistency Protocol:
        1. Sample k answers with high temperature.
        2. Cluster answers (exact match or simple normalization).
        3. If largest cluster > threshold * k, return that answer.
        4. Else, return "I don't know".
        """
        prompt = f"Question: {question} Answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
        
        # Sample k times
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=k
        )
        
        answers = [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in outputs]
        
        # Simple clustering (exact match after lowercasing)
        from collections import Counter
        counts = Counter([a.lower() for a in answers])
        if not counts:
             return "I don't know"
             
        most_common, count = counts.most_common(1)[0]
        
        consistency_score = count / k
        
        if consistency_score >= threshold:
            # Return the original case of the most common answer (find first match)
            for a in answers:
                if a.lower() == most_common:
                    return a
        
        return "I don't know"

    def generate_baseline(self, question, return_confidence=False):
        """Fallback: no evidence"""
        prompt = f"Question: {question} Answer:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        if return_confidence:
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Manual confidence calculation
            sequences = outputs.sequences
            scores = outputs.scores
            
            probs = []
            for i, step_scores in enumerate(scores):
                log_probs = torch.nn.functional.log_softmax(step_scores, dim=-1)
                # Last len(scores) tokens in sequences correspond to the scores
                token_index = -len(scores) + i
                token_id = sequences[0, token_index].item()
                
                token_log_prob = log_probs[0, token_id].item()
                probs.append(token_log_prob)

            if probs:
                avg_log_prob = np.mean(probs)
                confidence = float(np.exp(avg_log_prob))
            else:
                confidence = 0.0
            
            text = self.tokenizer.decode(sequences[0], skip_special_tokens=True)
            return text, confidence
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_clarification(self, question, evidence):
        """
        Generates a specific request for missing information.
        """
        # Truncate evidence to avoid context overflow and repetition issues
        max_evidence_len = 500
        short_evidence = evidence[:max_evidence_len] + "..." if len(evidence) > max_evidence_len else evidence
        
        prompt = f"""Question: {question}
            Evidence Found: {short_evidence}

            Task: The evidence provided is insufficient to answer the question.
            Generate a polite, single-sentence explanation of what specific information is missing.
            If you cannot pinpoint the missing info, simply say "The available evidence is insufficient."
            Do NOT repeat the question.

            Explanation:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=60, do_sample=False)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Fallback if model produces empty/repetitive content
        if not response or len(response) < 5 or response.lower() == question.lower():
             return "The evidence found was not sufficient to answer this question reliably."
        
        return response