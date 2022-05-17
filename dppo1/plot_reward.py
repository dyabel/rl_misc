import numpy as np
import matplotlib.pyplot as plt
reward1 = np.load('dppo_1.npy')
reward3 = np.load('dppo_3.npy')
plt.figure()
plt.plot(np.arange(len(reward1)), reward1)
plt.plot(np.arange(len(reward3)), reward3)
plt.savefig('reward.png')