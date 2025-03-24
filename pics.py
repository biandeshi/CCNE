import os
import re
import matplotlib.pyplot as plt

datasets = ['douban', 'twitter1_youtube', 'twitter_foursquare']
results = {}
train_file = 'FedRdm'

def extract_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

        results['MRR'] = float(re.search(r'MRR: ([\d.]+)', content).group(1))
        results['Acc'] = float(re.search(r'Acc: ([\d.]+)', content).group(1))
        results['AUC'] = float(re.search(r'AUC: ([\d.]+)', content).group(1))
        results['Precision@1'] = float(re.search(r'Precision@1: ([\d.]+)', content).group(1))
        results['Precision@5'] = float(re.search(r'Precision@5: ([\d.]+)', content).group(1))
        results['Precision@10'] = float(re.search(r'Precision@10: ([\d.]+)', content).group(1))
        results['Precision@20'] = float(re.search(r'Precision@20: ([\d.]+)', content).group(1))
        results['Precision@30'] = float(re.search(r'Precision@30: ([\d.]+)', content).group(1))
        results['Precision@40'] = float(re.search(r'Precision@40: ([\d.]+)', content).group(1))
        results['Precision@50'] = float(re.search(r'Precision@50: ([\d.]+)', content).group(1))
        results['Precision@60'] = float(re.search(r'Precision@60: ([\d.]+)', content).group(1))
        results['Precision@70'] = float(re.search(r'Precision@70: ([\d.]+)', content).group(1))
        results['Precision@80'] = float(re.search(r'Precision@80: ([\d.]+)', content).group(1))
        results['Precision@90'] = float(re.search(r'Precision@90: ([\d.]+)', content).group(1))
        results['Precision@100'] = float(re.search(r'Precision@100: ([\d.]+)', content).group(1))
    return results

def plot_p1_2_p100():
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    p_values = [results['Precision@1'], results['Precision@5']]
    p_values.append([results[f'Precision@{i * 10}'] for i in range(1, 11)])

    axs.plot([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], p_values, marker='o', label='Precision@K')
    axs.set_xlabel('K')
    axs.set_ylabel('Precision@K')
    axs.set_title('Precision@K vs K')
    axs.grid(True)

    plt.tight_layout()
    plt.savefig('pics/ratio_mrr_p10.png')

def plot_ratio_mmr_p10():
    fig, axs = plt.subplots(2, len(datasets), figsize=(16, 10))

    for i in range(len(datasets)):
        train_ratios = []
        mrrs = []
        p10s = []
        output_dir = f'output/{datasets[i]}'
        for train_ratio in [x / 10.0 for x in range(1, 10)]:
            file_name = f'{train_file}_tr={train_ratio}.txt'
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

def plot_margin_dim():
    dataset = datasets[1]
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

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

    plt.savefig('pics/margin_dim.png')
    plt.tight_layout()


if __name__ == '__main__':
    plot_ratio_mmr_p10()
    plot_margin_dim()