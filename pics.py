import os
import re
import matplotlib.pyplot as plt

train_file = ['FedGAN', 'CCNE']
datasets = ['douban', 'twitter1_youtube']
makers = ['.', 'x', 's']

def extract_results(file_path):
    results = {}
    with open(file_path, 'r') as file:
        content = file.read()

        results['MRR'] = float(re.search(r'MRR: ([\d.]+)', content).group(1))
        results['Acc'] = float(re.search(r'Acc: ([\d.]+)', content).group(1))
        results['AUC'] = float(re.search(r'AUC: ([\d.]+)', content).group(1))
        results['Precision@1'] = float(re.search(r'Precision@1: ([\d.]+)', content).group(1))
        results['Precision@5'] = float(re.search(r'Precision@5: ([\d.]+)', content).group(1))
        results['Precision@10'] = float(re.search(r'Precision@10: ([\d.]+)', content).group(1))
        results['Precision@15'] = float(re.search(r'Precision@15: ([\d.]+)', content).group(1))
        results['Precision@20'] = float(re.search(r'Precision@20: ([\d.]+)', content).group(1))
        results['Precision@25'] = float(re.search(r'Precision@25: ([\d.]+)', content).group(1))
        results['Precision@30'] = float(re.search(r'Precision@30: ([\d.]+)', content).group(1))
    return results

def plot_pK():
    fig, axs = plt.subplots(1, len(datasets), figsize=(10, 5))

    for i in range(len(datasets)):
        for j in range(len(train_file)):
            file = f'output/{datasets[i]}/{train_file[j]}_tr=0.8.txt'
            results = extract_results(file)
            p_values = [results[f'Precision@{k}'] for k in [1, 5, 10, 15, 20, 25, 30]]


            axs[i].plot([1, 5, 10, 15, 20, 25, 30], p_values, marker=makers[j], label=train_file[j])
        axs[i].set_xlabel('K')
        axs[i].set_ylabel('Precision@K')
        axs[i].grid(True)

    plt.tight_layout()
    plt.legend()
    plt.savefig('pics/pK.png')

def plot_ratio_mmr_pk():
    fig, axs = plt.subplots(2, len(datasets), figsize=(10, 10))

    output_dir = f'output/douban'
    for i in range(3):
        file = train_file[i]
        mrrs = []
        pk = []
        train_ratios = []
        for train_ratio in [x / 10.0 for x in range(1, 10)]:
            file_name = f'{file}_tr={train_ratio}.txt'
            file_path = os.path.join(output_dir, file_name)

            if os.path.exists(file_path):
                results = extract_results(file_path)
                train_ratios.append(train_ratio)
                mrrs.append(results['MRR'])
                pk.append(results['Precision@10'])
            else:
                print(f"Warning: {file_path} not found.")

        axs[0].plot(train_ratios, mrrs, marker=makers[i], label=file)
        axs[0].set_xlabel('Train Ratio')
        axs[0].set_ylabel('MRR')
        axs[0].grid(True)

        axs[1].plot(train_ratios, pk, marker=makers[i], label=file)
        axs[1].set_xlabel('Train Ratio')
        axs[1].set_ylabel('Precision@10')
        axs[1].grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig('pics/ratio_mrr_p10.png')

def plot_margin_dim():
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    output_dir = f'output/douban'
    pks = []

    for i in range(11):
        margin = i / 10.0
        filename = f'FedWoNeg_margin={margin}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            pks.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs[0].plot([x / 10.0 for x in range(11)], pks, marker='o', label='Precision@10')
    axs[0].set_xlabel('Margin')
    axs[0].set_ylabel('Precision@10')
    axs[0].set_title(f'Precision@10 vs Margin')
    axs[0].grid(True)

    pks = []
    dims = [2**i for i in range(4, 9)] + [400, 512]
    for dim in dims:
        filename = f'FedWoNeg_dim={dim}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            pks.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs[1].plot(dims, pks, marker='o', label='Precision@10')
    axs[1].set_xlabel('dimension')
    axs[1].set_ylabel('Precision@10')
    axs[1].set_title(f'Precision@10 vs Dim')
    axs[1].grid(True)

    plt.savefig('pics/margin_dim.png')
    plt.tight_layout()


if __name__ == '__main__':
    # plot_ratio_mmr_pk()
    # plot_margin_dim()
    plot_pK()