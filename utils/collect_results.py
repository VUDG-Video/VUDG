import json
import argparse
from tabulate import tabulate

def init_args():
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--input_path', type=str,
                        default='results/Qwen2.5_7b.json',
                        help='Path to input pred file')
    parser.add_argument('--mode', type=str, choices=["sdg", "mdg", "all"], default="all")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init_args()
    domains = {}
    category = {}
    with open(args.input_path, 'r', encoding='utf-8') as f:
        rows = json.load(f)
    for idx, item in enumerate(rows):
        domain = item['domain']
        qa_type = item['question_type']
        pred = item.get('pred', None)
        gt = item['answer']

        if domain not in domains:
            domains[domain] = [0.0, 0]
        if qa_type not in category:
            category[qa_type] = [0.0, 0]

        pred = pred.strip().upper() if pred else ""
        gt = gt.strip().upper()

        if pred == gt:
            domains[domain][0] += 1
            category[qa_type][0] += 1
        
        domains[domain][1] += 1
        category[qa_type][1] += 1


    domains_list = list(domains.keys())
    domain_accs = [domains[d][0] / domains[d][1] for d in domains_list]  # 保留 float 用于计算平均
    avg_acc = sum(domain_accs) / len(domain_accs)  # 域维度的平均（非加权）
    
    accuracies_str = [f"{acc * 100:.2f}" for acc in domain_accs]
    avg_str = f"{avg_acc * 100:.2f}"
    
    headers = ["Domain"] + domains_list + ["Avg"]
    rows = [["Accuracy"] + accuracies_str + [avg_str]]
    print("Domain Acc:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


    # across various question type
    if args.mode == 'all':
        print()
        type_list = list(range(1, 5))
        category_accs = [category[d][0] / category[d][1] for d in type_list]
        avg_cat_acc = sum(v[0] for v in category.values()) / sum(v[1] for v in category.values())
        cat_acc_str = [f"{acc * 100:.2f}" for acc in category_accs]
        avg_cat_str = f"{avg_cat_acc * 100:.2f}"

        headers = ["Category"] + [str(t) for t in type_list] + ["Avg"]
        rows = [["Accuracy"] + cat_acc_str + [avg_cat_str]]
        print("Category Acc:")
        print(tabulate(rows, headers=headers, tablefmt="grid"))