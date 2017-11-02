import numpy as np
import matplotlib.pyplot as plt

# k bandits: [(mean, std-dev), ..]
bandits = [(-3, 1), (1, 1), (2, 1), (4, 1), (0, 1)]
k = len(bandits)
e = 0.1

learning_rates = [0.1, 0.01, 0.001]
colors = ["-r", "-b", "-g"]
for lr_color in zip(learning_rates, colors):
    lr = lr_color[0]

    Q = [0 for _ in range(k)]
    N = [0 for _ in range(k)]

    # Plot
    plot_xs = []
    plot_ys = []

    for n in range(10_000):
        lr *= 0.999
        i = np.random.randint(0, k) if np.random.uniform() < e else np.argmax(Q)
        r = np.random.normal(bandits[i][0], bandits[i][1])
        N[i] += 1
        Q[i] += lr * (r - Q[i])

        # Plot
        plot_xs.append(n)
        plot_ys.append(np.average(Q, weights=N))

    print(Q)
    print(N)

    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.plot(plot_xs, plot_ys, lr_color[1])

plt.show()
