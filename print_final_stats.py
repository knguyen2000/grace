import json
import collections

def print_stats():
    try:
        with open('outputs/ce_final_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: outputs/ce_final_results.json not found.")
        return

    # Calculate stats for GRACE (Model 4)
    # The JSON contains a list of results for GRACE
    
    total = len(data)
    correct = sum(1 for r in data if r.get('is_correct'))
    abstains = sum(1 for r in data if r.get('final_answer', '').startswith("Abstained"))
    answered = total - abstains
    
    # Net Utility
    net_utility = sum(r.get('net_utility', 0) for r in data) / total if total > 0 else 0
    
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Abstains: {abstains}")
    print(f"Answered: {answered}")
    print(f"Net Utility: {net_utility:.3f}")
    
    if answered > 0:
        acc_answered = (correct - (sum(1 for r in data if r.get('is_correct') and r.get('final_answer', '').startswith("Abstained")))) / answered
        print(f"Accuracy (Answered): {acc_answered:.2%}")
    else:
        print("Accuracy (Answered): N/A")

    # Check GraphRAG stats if available in the JSON?
    # The JSON structure is a list of dicts for GRACE results only?
    # run_evaluation.py saves `final_results` which corresponds to `preds_grace`.
    # It doesn't save all models' full details in this JSON, only GRACE's detailed log.
    
    # But wait, run_evaluation.py prints the table for ALL models.
    # I can't reconstruct the full table from this JSON alone.
    # But I can see if GRACE is still 100% abstain.
    
if __name__ == "__main__":
    print_stats()
