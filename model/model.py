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
import json
from collections import OrderedDict
import matplotlib.pyplot as plt

# data_path = '/teamspace/studios/this_studio/arc/data.csv'
# checkpoint_path = '/teamspace/studios/this_studio/arc/model/checkpoint.pt'
# old_checkpoint_path = '/teamspace/studios/this_studio/arc/model/checkpoint0.pt'
# json_path = '/teamspace/studios/this_studio/arc/model/parameters.json'
data_path = '../data.csv'
checkpoint_path = './checkpoint.pt'
old_checkpoint_path = './checkpoint0.pt'
json_path = './parameters.json'

print('Loading Data...')
df = pd.read_csv(data_path, header=None)
print('Data Loaded')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concatenate(([[0,0]],move_vector_array)).tolist()
action_vectors = torch.tensor(action_array,dtype=torch.float64).to(device)
action_profiles = torch.tensor([
	action_array[i] + action_array[j]
	for i in range(9) for j in range(9)
]).to(device)

class ActionValueModel(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		n0 = 20
		n1 = 100
		n2 = 100
		n3 = 100
		n4 = 100
		n5 = 1
		self.h0 = nn.Linear(n0, n1)
		self.h1 = nn.Linear(n1, n2)
		self.h2 = nn.Linear(n2, n3)
		self.h3 = nn.Linear(n3, n4)
		self.h4 = nn.Linear(n4, n5)
	def forward(self, state_actions: Tensor) -> Tensor:
		layer0: Tensor = self.activation(self.h0(state_actions))
		layer1: Tensor = self.activation(self.h1(layer0))
		layer2: Tensor = self.activation(self.h2(layer1))
		layer3: Tensor = self.activation(self.h3(layer2))
		return self.h4(layer3)
	def __call__(self, *args, **kwds) -> Tensor:
		return super().__call__(*args, **kwds)
	def value(self, states: Tensor) -> Tensor:
		n = states.shape[0]
		s = states.repeat_interleave(81,0)
		a = action_profiles.repeat(n,1)
		state_actions = torch.cat((s,a),dim=1) 
		action_values = self(state_actions).reshape(-1,9,9)
		mins = torch.amin(action_values,2)
		return torch.amax(mins,1).unsqueeze(1)

def score(states: Tensor) -> Tensor:
	state0 = states[:,0:8]
	state1 = states[:,8:16]
	torsoPos0 = state0[:,0:2]
	torsoPos1 = state1[:,0:2]
	dist0 = torch.sqrt(torch.sum(torsoPos0**2,dim=1))
	dist1 = torch.sqrt(torch.sum(torsoPos1**2,dim=1))
	bladePos0 = state0[:,4:6]
	bladePos1 = state1[:,4:6]
	gapVec0 = torsoPos0 - bladePos1
	gapVec1 = torsoPos1 - bladePos0
	gap0 = torch.sqrt(torch.sum(gapVec0**2,dim=1))
	gap1 = torch.sqrt(torch.sum(gapVec1**2,dim=1))
	death0 = 1*(gap0 < 1.5)
	death1 = 1*(gap1 < 1.5)
	score = dist1 - dist0 + 100*(death1 - death0)
	return score.unsqueeze(1)

def save_checkpoint():
	checkpoint = { 
		'state_dict': model.state_dict(),
		'best_mse': best_mse
	}
	torch.save(checkpoint,checkpoint_path)
	ordered_dict = OrderedDict()
	for key, value in model.state_dict().items():
		if isinstance(value, Tensor):
			ordered_dict[key] = value.detach().cpu().tolist()
	with open(json_path,'w') as file: 
		json.dump(ordered_dict, file, indent=4)

old_model = ActionValueModel().to(device).eval()
model = ActionValueModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_mse = 1000000
if os.path.exists(checkpoint_path):
	checkpoint = torch.load(checkpoint_path, weights_only=False)
	model.load_state_dict(checkpoint['state_dict'])
	best_mse = checkpoint['best_mse']
if os.path.exists(old_checkpoint_path):
	old_checkpoint = torch.load(old_checkpoint_path, weights_only=False)
	old_model.load_state_dict(old_checkpoint['state_dict'])

print('Pre-Calculating Values')
base_data: Tensor = torch.tensor(df.to_numpy(),dtype=torch.float64)
precal_batch_size = 10000
precal_batch_count = (base_data.shape[0] // precal_batch_size) + 1
future_state = base_data[:,20:36]
precal_dataset = TensorDataset(future_state)
precal_dataloader = DataLoader(precal_dataset, batch_size=10000, shuffle=False)
batch_outputs = []
with torch.no_grad():
	for batch, data_array in enumerate(precal_dataloader):
		future_state = data_array[0].to(device)
		print(f'Pre-Calculate Batch {batch} / {precal_batch_count}')
		future_value = old_model.value(future_state).detach().cpu()
		batch_outputs.append(future_value)
future_value = torch.cat(batch_outputs,dim=0)

data = torch.cat((base_data,future_value),dim=1)
dataset = TensorDataset(data)
batch_size = 100000
batch_count = (len(dataset) // batch_size) + 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

# os.system('clear')
dt = 0.2
discount = 0.01
gamma = 1 - dt*discount
epoch_count = 100000

print('Training...')
for epoch in range(epoch_count):
	total_loss = 0
	for batch, batch_data in enumerate(dataloader):
		data: Tensor = batch_data[0].to(device)
		n = data.shape[0]
		optimizer.zero_grad()
		state_actions = data[:,0:20]
		output = model(state_actions)
		state = data[:,0:16]
		present_score = score(state)
		future_state = data[:,20:36] 
		future_score = score(future_state)
		reward = future_score - present_score
		future_value = data[:,36].unsqueeze(1)
		target = reward + gamma*future_value
		loss = F.mse_loss(output, target, reduction='sum')
		loss.backward()
		optimizer.step()
		batch_loss = loss.detach().cpu().numpy()
		batch_mse = batch_loss / n
		total_loss += batch_loss
		# if batch % 1 == 0:
		# 	message = ''
		# 	message += f'Epoch {epoch+1}, '
		# 	message += f'Batch {batch+1:02d} / {batch_count}, '
		# 	message += f'Loss: {batch_mse}'
		# 	print(message)
	epoch_mse = total_loss / len(dataset)
	if epoch_mse < best_mse:
		best_mse = epoch_mse
		save_checkpoint()
	message = ''
	message += f'Epoch {epoch+1}, '
	message += f'Loss: {epoch_mse:08f}, '
	message += f'Best: {best_mse:08f} '
	print(message)
