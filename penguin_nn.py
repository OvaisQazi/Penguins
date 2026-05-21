import json
import numpy as np
import matplotlib.pyplot as plt
from functions import nn, predict, n0, n1, n2

# loading data
with open('penguin-data.json', 'r') as f:
    datadict = json.load(f)

# extract data
data = np.array(datadict['data'])

# subtract mean per feature (column)
data -= data.mean(axis = 0)
data /= data.std(axis = 0)

# extract labels
lbls = np.array(datadict['target'])

# print to check if correctly loaded
print("Dataset loaded.")
print(f"Classes : {datadict['target_names']}")
print(f"Features : {datadict['feature_names']}")
print(f"Sample : {data[0]}")

# dict for mapping labels
mplbldict = {0: np.array([1, -1, -1]),
             1: np.array([-1, 1, -1]),
             2: np.array([-1, -1, 1])}

# building label matrix
labels = np.array([mplbldict[lbl] for lbl in lbls])

print("\nLabel encoding:")
for k, v in mplbldict.items():
    print(f"{datadict['target_names'][k]:10s} ({k}) → {v}")

# splitting train and test data
# np.random.seed(69) # if you want to not change the randomness
N = data.shape[0]
index = np.random.permutation(N)
split = int(0.9 * N)

train_index, test_index = index[:split], index[split:]
x_train, y_train = data[train_index], labels[train_index]
x_test, y_test = data[test_index], labels[test_index]

print(f"\nTraining samples : {len(train_index)}\nTesting samples : {len(test_index)}")

# setting random weights
s = n0 * n1 + n1 + n1 * n2 + n2
wvec = np.random.randn(s) * 0.1

print(f"\nWeight vector size: {s}")
print(f"A1: {n1}x{n0}={n0*n1}  b1: {n1}  A2: {n2}x{n1}={n1*n2}  b2: {n2}")

# training loop

lr = 0.05 # learning rate
iter = 5000 # iteration
ss_sgd = 1e-6 # step size for sgd

loss_history, grad_norm_history, wvec = nn(lr, iter, ss_sgd, wvec, x_train, y_train)

# testing

y_test_int = np.array([np.argmax(y) for y in y_test])
y_pred_int = predict(wvec, x_test)

print("\nTrue vs predicted labels")
for i, (true, pred) in enumerate(zip(y_test_int, y_pred_int)):
    status = "✔" if true == pred else "✘"
    print(f"{i+1:3d}: True: {datadict['target_names'][true]:10s} | Pred: {datadict['target_names'][pred]:10s} {status}")

accuracy = np.mean(y_pred_int == y_test_int)
print(f"\nTest accuracy: {accuracy * 100:.1f}%")

# Per-class breakdown
for k, name in enumerate(datadict['target_names']):
    mask = y_test_int == k
    acc_k = np.mean(y_pred_int[mask] == k)
    print(f"  {name:10s}: {acc_k * 100:.1f}%  ({mask.sum()} samples)")

# ploting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
 
steps = np.arange(len(loss_history)) * 100
 
ax1.plot(steps, loss_history, color='steelblue')
ax1.set_xlabel('SGD iteration')
ax1.set_ylabel('Mean squared loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)
 
ax2.plot(steps, grad_norm_history, color='coral')
ax2.set_xlabel('SGD iteration')
ax2.set_ylabel('||gradient||')
ax2.set_title('Gradient Norm')
ax2.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\nPlot saved to training_curves.png")