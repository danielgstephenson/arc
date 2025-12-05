from math import pi
from pickle import TRUE
from typing import KeysView
import torch
from torch import Tensor, nn, tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import block, r_, sin, cos
import os
import sys
import json
from collections import OrderedDict
import matplotlib.pyplot as plt

data_path = '../data.csv'
checkpoint_path = './checkpoint.pt'
old_checkpoint_path = './checkpoint.pt'
json_path = './parameters.json'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float32)
torch.set_printoptions(sci_mode=False)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concatenate(([[0,0]],move_vector_array)).tolist()
action_vectors = torch.tensor(action_array,dtype=torch.float32).to(device)
action_profiles = torch.tensor([
	action_array[i] + action_array[j]
	for i in range(9) for j in range(9)
]).to(device)

class ValueModel(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		n0 = 16
		k = 20
		self.h0 = nn.Linear(n0, k)
		self.h1 = nn.Linear(n0 + 1*k, k)
		self.h2 = nn.Linear(n0 + 2*k, k)
		self.h3 = nn.Linear(n0 + 3*k, k)
		self.h4 = nn.Linear(n0 + 4*k, 1)
	def forward(self, state: Tensor) -> Tensor:
		y0: Tensor = self.activation(self.h0(state))
		x1 = torch.cat((state, y0), dim=1)
		y1: Tensor = self.activation(self.h1(x1))
		x2 = torch.cat((x1, y1), dim=1)
		y2: Tensor = self.activation(self.h2(x2))
		x3 = torch.cat((x2, y2), dim=1)
		y3: Tensor = self.activation(self.h3(x3))
		x4 = torch.cat((x3, y3), dim=1)
		return self.h4(x4)
	def __call__(self, *args, **kwds) -> Tensor:
		return super().__call__(*args, **kwds)
	def maximin(self, states: Tensor) -> Tensor:
		n = states.shape[0]
		x = states.reshape(-1,16)
		values = self(x)
		value_matrices = values.reshape(n,9,9)
		mins = torch.amin(value_matrices,2)
		return torch.amax(mins,1).unsqueeze(1)

def save_checkpoint():
	checkpoint = { 
		'state_dict': model.state_dict(),
	}
	torch.save(checkpoint,checkpoint_path)
	ordered_dict = OrderedDict()
	for key, value in model.state_dict().items():
		if isinstance(value, Tensor):
			ordered_dict[key] = value.detach().cpu().tolist()
	with open(json_path,'w') as file: 
		json.dump(ordered_dict, file, indent=4)

print('Loading Data...')
df = pd.read_csv(data_path, header=None)
data = torch.tensor(df.to_numpy(),dtype=torch.float32)
dataset = TensorDataset(data)
batch_size = 10000
batch_count = (len(dataset) // batch_size)
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
print('Data Loaded')

old_model = ValueModel().to(device).eval()
model = ValueModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if os.path.exists(checkpoint_path):
	checkpoint = torch.load(checkpoint_path, weights_only=False)
	model.load_state_dict(checkpoint['state_dict'])
if os.path.exists(old_checkpoint_path):
	old_checkpoint = torch.load(old_checkpoint_path, weights_only=False)
	old_model.load_state_dict(old_checkpoint['state_dict'])

# os.system('clear')
discount = 0.99
step = 1
best_mse = 1000000
step_epoch = 0
best_count = 0
max_best_count = 100
best_state_dict = model.state_dict()

# Verify the maximin process.
# Start with a larger timestep?


print('Training...')
for epoch in range(10000000):
	step_epoch += 1
	for batch, batch_data in enumerate(dataloader):
		data: Tensor = batch_data[0].to(device)
		n = data.shape[0]
		optimizer.zero_grad()
		state = data[:,0:16]
		output = model(state)
		reward = data[:,16].unsqueeze(1)
		future_states = data[:,17:]
		with torch.no_grad():
			future_value = old_model.maximin(future_states)
		target = (1-discount)*reward + discount*future_value
		loss = F.mse_loss(output, target, reduction='mean')
		loss.backward()
		optimizer.step()
		batch_mse = loss.detach().cpu().numpy()
		if batch_mse < best_mse:
			best_mse = batch_mse
			best_state_dict = model.state_dict()
			best_count = 0
		else:
			best_count += 1
		message = ''
		message += f'Step {step}, '
		message += f'Epoch {step_epoch}, '
		message += f'Batch {batch+1:03} / {batch_count}, '
		message += f'Loss: {batch_mse:.4f}, '
		message += f'Best: {best_mse:.4f}, '
		message += f'Count: {best_count:03} / {max_best_count}'
		print(message)
		if best_count >= max_best_count:
			old_model.load_state_dict(best_state_dict)
			# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
			best_mse = 1000000
			best_count = 0
			step += 1
			step_epoch = 0
			save_checkpoint()
			break
