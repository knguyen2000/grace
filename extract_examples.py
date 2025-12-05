import json

def main():
    with open('outputs/ce_final_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Successful Answer
    success = [r for r in data if r['graph_action'] == 'answer' and r['graph_correct']]
    
    # 2. Correct Abstain
    correct_abstain = [r for r in data if r['graph_action'] == 'abstain' and r.get('abstain_correct') == True]
    
    # 3. Failure (Wrong Answer)
    failure = [r for r in data if r['graph_action'] == 'answer' and not r['graph_correct']]

    with open('examples.md', 'w', encoding='utf-8') as f:
        f.write("# Examples\n\n")
        
        if success:
            ex = success[0]
            f.write("## Success Case\n")
            f.write(f"**Question:** {ex['question']}\n")
            f.write(f"**Answer:** {ex['final_answer']}\n")
            f.write(f"**Gold:** {ex['gold']}\n")
            f.write(f"**Action:** {ex['joint_action']}\n")
            f.write(f"**Net Utility:** {ex['net_utility']}\n\n")

        if correct_abstain:
            ex = correct_abstain[0]
            f.write("## Correct Abstention\n")
            f.write(f"**Question:** {ex['question']}\n")
            f.write(f"**Result:** {ex['final_answer']}\n")
            f.write(f"**Gold:** {ex['gold']}\n")
            f.write(f"**Baseline Prediction:** {ex.get('baseline_pred', 'N/A')}\n")
            f.write(f"**Action:** {ex['joint_action']}\n")
            f.write(f"**Net Utility:** {ex['net_utility']}\n\n")
            
        if failure:
            ex = failure[0]
            f.write("## Failure Case\n")
            f.write(f"**Question:** {ex['question']}\n")
            f.write(f"**Answer:** {ex['final_answer']}\n")
            f.write(f"**Gold:** {ex['gold']}\n")
            f.write(f"**Action:** {ex['joint_action']}\n")
            f.write(f"**Net Utility:** {ex['net_utility']}\n\n")

if __name__ == "__main__":
    main()
