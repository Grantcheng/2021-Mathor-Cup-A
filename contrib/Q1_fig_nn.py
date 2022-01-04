import pickle

import matplotlib.pyplot as plt

idx = 1
with open(f'raw/Q1_history_nn_{idx}.pkl', 'rb') as f:
    history = pickle.load(f)

fig = plt.figure()
plt.plot(history['val_loss'])
plt.xlabel('Steps')
plt.ylabel('Huber loss on valid set')
fig.savefig(f'results/Q1_history_nn_{idx}.svg')
plt.close(fig)
