# %%
import os
import sys

# Move up one directory from 'notebooks' to 'root_dir' and add it to python path
#root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
#if root_dir not in sys.path:
#    sys.path.append(root_dir)


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy 

from .pc_layer import PCLayer
from .pc_trainer import PCTrainer

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(f'using {device}')

# %%
# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x))])
train_dataset = datasets.MNIST('../data', download=True, train=True, transform=transform)
test_dataset = datasets.MNIST('../data', download=True, train=False, transform=transform)

batch_size = 500
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print(f'# train images: {len(train_dataset)} and # test images: {len(test_dataset)}')

# %%
input_size = 28*28  # 28x28 images
hidden_size = 256
output_size = 10    # 10 classes
activation_fn = nn.ReLU
loss_fn = lambda output, _target: 0.5 * (output - _target).pow(2).sum() # this loss function holds to the error of the output layer of the model


model = nn.Sequential(
    nn.Linear(input_size, hidden_size),
    PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, hidden_size),
    PCLayer(),
    activation_fn(),
    nn.Linear(hidden_size, output_size)
)
model.train()   # set the model to training mode
model.to(device)

# %%
# number of inference iterations where the latent states x are updated. Inference does not run till convergence but for a fixed number of iterations
T = 20                              

# options for the update of the latent state x
optimizer_x_fn = optim.SGD          # optimizer for latent state x, SGD perform gradient descent. Other alternative are Adam, RMSprop, etc. 
optimizer_x_kwargs = {'lr': 0.01}   # optimizer parameters for latent state x to pass to the optimizer. The best learning rate will depend on the task and the optimiser. 
                                    # Other parameters such as momentum, weight_decay could also be set here with additional elements, e.g., "momentum": 0.9, "weight_decay": 0.01

# options for the update of the parameters p
update_p_at = 'last'                # update parameters p at the last iteration, can be set to 'all' to implement ipc (https://arxiv.org/abs/2212.00720)
optimizer_p_fn = optim.Adam         # optimizer for parameters p
optimizer_p_kwargs = {'lr': 0.001}  # optimizer parameters for parameters p, 0.001 is a good starting point for Adam, but it should be adjusted for the task

trainer = PCTrainer(model, 
    T = T, 
    optimizer_x_fn = optimizer_x_fn,
    optimizer_x_kwargs = optimizer_x_kwargs,
    update_p_at = update_p_at,   
    optimizer_p_fn = optimizer_p_fn,
    optimizer_p_kwargs = optimizer_p_kwargs,
)

# %%
# get classification accuracy of the model
def test(model, dataset, batch_size=1000):
    model.eval()
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, predicted = torch.max(pred, -1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
    model.train()
    return round(correct / total, 4)

# %%
epochs = 10

test_acc = np.zeros(epochs + 1)
test_acc[0] = test(model, test_dataset)
for epoch in range(epochs):
    # Initialize the tqdm progress bar
    with tqdm(train_loader, desc=f'Epoch {epoch+1} - Test accuracy: {test_acc[epoch]:.3f}') as pbar:
        for data, label in pbar:
            data, label = data.to(device), label.to(device)
            # convert labels to one-hot encoding
            label = F.one_hot(label, num_classes=output_size).float()
            trainer.train_on_batch(
                inputs=data,
                loss_fn=loss_fn,
                loss_fn_kwargs={'_target': label}
            )
    test_acc[epoch + 1] = test(model, test_dataset)
    pbar.set_description(f'Epoch {epoch + 1} - Test accuracy: {test_acc[epoch + 1]:.3f}')

plt.plot(test_acc)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.show()

# %%



