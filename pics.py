import matplotlib.pyplot as plt

epoch = [1, 2, 3, 4, 5]
accuracy = [0.8, 0.9, 0.7, 0.6, 0.5]

plt.plot(epoch, accuracy)
plt.grid(True)
plt.savefig('plot.png')