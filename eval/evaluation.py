# eval/evaluation.py

def evaluate_batch(results, mode="graph"):
    action_key = f"{mode}_action"
    correct_key = f"{mode}_correct"
    correct = sum(r[correct_key] for r in results if r.get(action_key) == 'answer')
    total_answered = sum(1 for r in results if r.get(action_key) == 'answer')
    total = len(results)
    coverage = total_answered / total if total > 0 else 0
    accuracy = correct / total_answered if total_answered > 0 else 0
    return {"accuracy": accuracy, "coverage": coverage}
