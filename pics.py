import os
import re
import matplotlib.pyplot as plt

datasets = ['douban', 'twitter1_youtube', 'twitter_foursquare']

def extract_results(file_path):
    results = {}
    with open(file_path, 'r') as file:
        content = file.read()

        results['MRR'] = float(re.search(r'MRR: ([\d.]+)', content).group(1))
        results['Acc'] = float(re.search(r'Acc: ([\d.]+)', content).group(1))
        results['AUC'] = float(re.search(r'AUC: ([\d.]+)', content).group(1))
        results['Precision@1'] = float(re.search(r'Precision@1: ([\d.]+)', content).group(1))
        results['Precision@10'] = float(re.search(r'Precision@10: ([\d.]+)', content).group(1))
    return results

def plot_ratio_mmr_p10():
    fig, axs = plt.subplots(2, len(datasets), figsize=(16, 10))

    for i in range(len(datasets)):
        train_ratios = []
        mrrs = []
        p10s = []
        output_dir = f'output/{datasets[i]}'
        for train_ratio in [x / 10.0 for x in range(1, 10)]:
            file_name = f'FedWoNeg_tr={train_ratio}.txt'
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                results = extract_results(file_path)
                train_ratios.append(train_ratio)
                mrrs.append(results['MRR'])
                p10s.append(results['Precision@10'])
            else:
                print(f"Warning: {file_path} not found.")

        axs[0, i].plot(train_ratios, mrrs, marker='o', label='MRR')
        axs[0, i].set_xlabel('Train Ratio')
        axs[0, i].set_ylabel('MRR')
        axs[0, i].set_title(f'MRR vs Train Ratio for {datasets[i]}')
        axs[0, i].grid(True)

        axs[1, i].plot(train_ratios, p10s, marker='o', label='Precision@10')
        axs[1, i].set_xlabel('Train Ratio')
        axs[1, i].set_ylabel('Precision@10')
        axs[1, i].set_title(f'Precision@10 vs Train Ratio for {datasets[i]}')
        axs[1, i].grid(True)

    plt.savefig('pics/ratio_mrr_p10.png')
    plt.tight_layout()

def plot_margin_dim_alpha():
    dataset = datasets[1]
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))

    output_dir = f'output/{dataset}'
    p10s = []

    for i in range(11):
        margin = i / 10.0
        filename = f'FedWoNeg_margin={margin}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            p10s.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs[0].plot([x / 10.0 for x in range(11)], p10s, marker='o', label='Precision@10')
    axs[0].set_xlabel('Margin')
    axs[0].set_ylabel('Precision@10')
    axs[0].set_title(f'Precision@10 vs Margin for {dataset}')
    axs[0].grid(True)

    p10s = []

    for i in [2**i for i in range(4, 10)]:
        dim = i
        filename = f'FedWoNeg_dim={dim}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            p10s.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs[1].plot([2**i for i in range(4, 10)], p10s, marker='o', label='Precision@10')
    axs[1].set_xlabel('Dim')
    axs[1].set_ylabel('Precision@10')
    axs[1].set_title(f'Precision@10 vs Dim for {dataset}')
    axs[1].grid(True)

    p10s = []

    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        alpha = i
        filename = f'FedWoNeg_alpha={alpha}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            p10s.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs[2].plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], p10s, marker='o', label='Precision@10')
    axs[2].set_xlabel('Alpha')
    axs[2].set_ylabel('Precision@10')
    axs[2].set_title(f'Precision@10 vs Alpha for {dataset}')
    axs[2].grid(True)

    plt.savefig('pics/margin_dim_alpha.png')
    plt.tight_layout()


if __name__ == '__main__':
    plot_ratio_mmr_p10()
    plot_margin_dim_alpha()