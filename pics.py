import os
import re
import matplotlib.pyplot as plt


train_file = ['FedGAN', 'CCNE', 'DeepLink', 'PALE', 'CAMU']
datasets = ['douban', 'twitter_youtube']
makers = ['.', 'x', 's', '^', 'D', 'o']

IONE_ratio_douban_MRR = [0.2013, 0.2826, 0.3956, 0.4215, 0.5265, 0.5823, 0.5794, 0.7215, 0.7896]
IONE_ratio_douban_p10 = [0.3416, 0.5126, 0.6145, 0.6208, 0.7456, 0.7911, 0.7741, 0.8906, 0.9294]
IONE_ratio_youtube_MRR = [0.0171, 0.0324, 0.0483, 0.0637, 0.0791, 0.1025, 0.1162, 0.1618, 0.1624]
IONE_ratio_youtube_p10 = [0.0123, 0.0463, 0.0793, 0.1032, 0.1339, 0.1607, 0.1922, 0.2481, 0.2446]

DeepLink_ratio_douban_MRR = [0.1675, 0.2865, 0.3698, 0.4225, 0.4763, 0.4983, 0.4986, 0.5016, 0.5041]
DeepLink_ratio_douban_p10 = [0.3768, 0.5532, 0.6625, 0.7429, 0.7986, 0.8263, 0.8378, 0.8239, 0.8496]
DeepLink_ratio_youtube_MRR = [0.0332, 0.0517, 0.0609, 0.0623, 0.0691, 0.0729, 0.0798, 0.0856, 0.1103]
DeepLink_ratio_youtube_p10 = [0.0732, 0.1123, 0.1346, 0.1356, 0.1486, 0.1501, 0.1723, 0.1629, 0.1976]

PALE_ratio_douban_MRR = [0.2006, 0.3351, 0.4212, 0.4814, 0.5152, 0.5327, 0.5516, 0.5741, 0.5591]
PALE_ratio_douban_p10 = [0.4506, 0.6269, 0.7312, 0.7759, 0.8179, 0.8321, 0.8498, 0.8432, 0.8316]
PALE_ratio_youtube_MRR = [0.0346, 0.0519, 0.0616, 0.0678, 0.0713, 0.0741, 0.0861, 0.0913, 0.1082]
PALE_ratio_youtube_p10 = [0.0794, 0.1158, 0.1201, 0.1513, 0.1582, 0.1623, 0.1799, 0.1942, 0.2211]

CAMU_ratio_douban_MRR = [0.2213, 0.3321, 0.4213, 0.4659, 0.5123, 0.5105, 0.5597, 0.5269, 0.5523]
CAMU_ratio_douban_p10 = [0.4512, 0.6345, 0.7512, 0.7945, 0.8214, 0.8549, 0.8493, 0.8513, 0.8613]
CAMU_ratio_youtube_MRR = [0.0366, 0.0563, 0.0632, 0.0794, 0.0761, 0.0823, 0.0903, 0.1002, 0.1113]
CAMU_ratio_youtube_p10 = [0.0774, 0.1261, 0.1413, 0.1623, 0.1602, 0.1801, 0.1891, 0.2003, 0.2184]

MRR_douban = [DeepLink_ratio_douban_MRR, PALE_ratio_douban_MRR, CAMU_ratio_douban_MRR]
MRR_youtube = [DeepLink_ratio_youtube_MRR, PALE_ratio_youtube_MRR, CAMU_ratio_youtube_MRR]
p10_douban = [DeepLink_ratio_douban_p10, PALE_ratio_douban_p10, CAMU_ratio_douban_p10]
p10_youtube = [DeepLink_ratio_youtube_p10, PALE_ratio_youtube_p10, CAMU_ratio_youtube_p10]

MRR_dict = {'douban': MRR_douban, 'twitter_youtube': MRR_youtube}
p10_dict = {'douban': p10_douban, 'twitter_youtube': p10_youtube}

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
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    p = [1, 5, 10, 15, 20, 25, 30]
    lines = []
    labels = []
    
    for i in range(len(datasets)):
        for j in range(len(train_file)):
            if j > 1:
                break
            else:
                file = f'output/{datasets[i]}/{train_file[j]}_tr=0.2.txt'
                results = extract_results(file)
                p_values = [results[f'Precision@{k}'] for k in p]

            line , = axs[i].plot(p, p_values, marker=makers[j], label = 'FAUA' if j == 0 else train_file[j])
            lines.append(line)
            labels.append(train_file[j])
        axs[i].set_xlabel('K')
        axs[i].set_ylabel('Precision@K')
        axs[i].legend()
        axs[i].set_title(f'({i + 1})')
        axs[i].grid(True)

    # fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)
    plt.subplots_adjust(bottom=0.1)
    plt.tight_layout()
    plt.savefig(f'pics/pK.png')

def plot_ratio_mrr_p10():
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    train_ratios = [x / 10.0 for x in range(1, 10)]
    for i in range(len(datasets)):
        output_dir = f'output/{datasets[i]}'
        
        for j in range(len(train_file)):
            mrr = []
            p10 = []

            if j < 2:
                # Get MRR and p10 for each train ratio
                for train_ratio in train_ratios:
                    file_name = f'{train_file[j]}_tr={train_ratio}.txt'
                    file_path = os.path.join(output_dir, file_name)

                    if os.path.exists(file_path):
                        results = extract_results(file_path)
                        mrr.append(results['MRR'])
                        p10.append(results['Precision@10'])
                    else:
                        print(f"Warning: {file_path} not found.")
            else:
                # mrr = MRR_dict[datasets[i]][j - 2]
                # p10 = p10_dict[datasets[i]][j - 2]
                break

            axs[i, 0].plot(train_ratios, mrr, marker=makers[j], label='FAUA' if j == 0 else train_file[j])
            axs[i, 1].plot(train_ratios, p10, marker=makers[j], label='FAUA' if j == 0 else train_file[j])
        axs[i, 0].set_xlabel('Train Ratio')
        axs[i, 0].set_ylabel('MRR')
        axs[i, 0].set_title(f'({i * 2 + 1})')
        axs[i, 0].legend()
        axs[i, 0].grid(True)

        axs[i, 1].set_xlabel('Train Ratio')
        axs[i, 1].set_ylabel('Precision@10')
        axs[i, 1].set_title(f'({i * 2 + 2})')
        axs[i, 1].legend()
        axs[i, 1].grid(True)

    # fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    # plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(f'pics/ratio_mrr_p10.png')

def plot_ratio_mrr(dataset):
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))

    train_ratios = [x / 10.0 for x in range(1, 10)]
    for i in range(len(train_file)):
        if i < 2:
            output_dir = f'output/{dataset}'
            mrr = []
            for train_ratio in train_ratios:
                file_name = f'{train_file[i]}_tr={train_ratio}.txt'
                file_path = os.path.join(output_dir, file_name)

                if os.path.exists(file_path):
                    results = extract_results(file_path)
                    mrr.append(results['MRR'])
                else:
                    print(f"Warning: {file_path} not found.")
        else:
            mrr = MRR_dict[dataset][i - 2]
        axs.plot([x / 10.0 for x in range(1, 10)], mrr, marker=makers[i], label=train_file[i])

    axs.set_xlabel('Train Ratio')
    axs.set_ylabel('MRR')
    axs.set_title(dataset)
    axs.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pics/{dataset}/ratio_mrr.png', format='png', dpi=1000)

def plot_ratio_pk(dataset):
    fig, axs = plt.subplots(1, 1, figsize=(8, 8))

    train_ratios = [x / 10.0 for x in range(1, 10)]
    for i in range(len(train_file)):
        if i < 2:
            output_dir = f'output/{dataset}'
            p10 = []
            for train_ratio in train_ratios:
                file_name = f'{train_file[i]}_tr={train_ratio}.txt'
                file_path = os.path.join(output_dir, file_name)

                if os.path.exists(file_path):
                    results = extract_results(file_path)
                    p10.append(results['Precision@10'])
                else:
                    print(f"Warning: {file_path} not found.")
        else:
            p10 = p10_dict[dataset][i - 2]
        axs.plot([x / 10.0 for x in range(1, 10)], p10, marker=makers[i], label=train_file[i])

    axs.set_xlabel('Train Ratio')
    axs.set_ylabel('Precision@10')
    axs.set_title(dataset)
    axs.grid(True)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'pics/{dataset}/ratio_p10.png', format='png', dpi=1000)
    

def plot_dim():
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    output_dir = f'output/{datasets[1]}'
    pks = []

    dims = [2**i for i in range(4, 9)] + [400, 512]
    for dim in dims:
        filename = f'{train_file[0]}_dim={dim}.txt'
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            results = extract_results(file_path)
            pks.append(results['Precision@10'])
        else:
            print(f"Warning: {file_path} not found.")

    axs.plot(dims, pks, marker='o', label='Precision@10')
    axs.set_xlabel('dimension')
    axs.set_ylabel('Precision@10')
    axs.set_title(f'Precision@10 vs Dim')
    axs.grid(True)

    plt.savefig('pics/P10_2_dim.png')
    plt.tight_layout()


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})
    plot_dim()
    plot_ratio_mrr_p10()
    plot_pK()
    # for dataset in datasets:
        # plot_pK(dataset)
        # plot_ratio_pk(dataset)
        # plot_ratio_mrr(dataset)