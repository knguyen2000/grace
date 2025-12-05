import torch
from transformers import pipeline

class NliVerifier:
    def __init__(self, model_name="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"):
        """
        Initializes the NLI verifier pipeline.
        """
        device = 0 if torch.cuda.is_available() else -1
        print(f"Loading NLI model {model_name} on device {device}...")
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            truncation=True
        )
        # Map model labels to our internal labels
        self.label_map = {
            "ENTAILMENT": "entails",
            "CONTRADICTION": "contradicts",
            "NEUTRAL": "unknown"
        }
        print("NLI Verifier loaded.")

    def verify(self, evidence, answer):
        """
        Verifies if the evidence (premise) entails, contradicts, or
        is neutral to the answer (hypothesis)
        """
        if not evidence or not answer:
            return "unknown", 0.0

        result = self.pipe(f"{evidence} </s></s> {answer}")[0]
        
        label = self.label_map.get(result['label'], "unknown")
        score = result['score']

        return label, score
