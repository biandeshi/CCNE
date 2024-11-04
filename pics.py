import matplotlib.pyplot as plt

ratio = [i/10 for i in range(1, 10)]
p10 = [0.4945, 0.6525, 0.7663, 0.8137, 0.8605, 0.8772, 0.9137, 0.9286, 0.9286]
mrr = [0.2429, 0.3489, 0.4715, 0.5330, 0.5741, 0.5957, 0.6217, 0.6405, 0.6425]

plt.plot(ratio, p10)
plt.xlabel('ratio')
plt.ylabel('precision@10')
plt.grid(True)
plt.savefig('pics/ratio_p10.png')
plt.cla()

plt.plot(ratio, mrr)
plt.xlabel('ratio')
plt.ylabel('MRR')
plt.grid(True)
plt.savefig('pics/ratio_MRR.png')