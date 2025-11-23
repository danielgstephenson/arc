from math import pi
from pickle import TRUE
from typing import KeysView
import torch
from torch import Tensor, nn, tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import block, r_, sin, cos
import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt

os.system('clear')
# data_path = '/teamspace/studios/this_studio/arc/data.csv'
# checkpoint_path = '/teamspace/studios/this_studio/arc/model/checkpoint.pt'
# json_path = '/teamspace/studios/this_studio/arc/model/parameters.json'
data_path = '../data.csv'
checkpoint_path = 'checkpoint.pt'
json_path = 'parameters.json'

print('loading data...')
df = pd.read_csv(data_path, header=None)
print('data loaded.')

# Example of how to unpack one row of the data
16*(1+9*9) # 1 current state, 9*9 possible next states
x = torch.tensor(df.iloc[0,:].to_numpy())
x.shape
y = x.unfold(0,16,16)
y.shape

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False)

class MyDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		self.data: Tensor = torch.tensor(df.to_numpy(),dtype=torch.float64)
		self.count = df.shape[0]

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		return self.data[idx]

dataset = MyDataset(df)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concatenate(([[0,0]],move_vector_array)).tolist()
action_vectors = torch.tensor(action_array,dtype=torch.float64).to(device)
action_pairs = torch.tensor([[i,j] for i in range(9) for j in range(9)]).to(device)

class ActionValueModel(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		# self.activation = torch.nn.ReLU()
		n0 = 16
		n1 = 200
		n2 = 200
		n3 = 200
		n4 = 200
		n5 = 81
		self.h0 = nn.Linear(n0, n1)
		self.h1 = nn.Linear(n1, n2)
		self.h2 = nn.Linear(n2, n3)
		self.h3 = nn.Linear(n3, n4)
		self.h4 = nn.Linear(n4, n5)
	def forward(self, state: Tensor) -> Tensor:
		layer0: Tensor = self.activation(self.h0(state))
		layer1: Tensor = self.activation(self.h1(layer0))
		layer2: Tensor = self.activation(self.h2(layer1))
		layer3: Tensor = self.activation(self.h3(layer2))
		return self.h4(layer3)
	def __call__(self, *args, **kwds) -> Tensor:
		return super().__call__(*args, **kwds)
	def value(self, state: Tensor) -> Tensor:
		action_values = self(state).reshape(-1,9,9)
		mins = torch.amin(action_values,2)
		return torch.amax(mins,1)
	
old_model = ActionValueModel().to(device)
model = ActionValueModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def score(state: Tensor) -> Tensor:
	state0 = state[:,0:8]
	state1 = state[:,8:16]
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

# os.system('clear')

if os.path.exists(checkpoint_path):
	print('loading checkpoint...')
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['state_dict'])
	old_model.load_state_dict(checkpoint['state_dict'])

def save_checkpoint():
	state_dict = model.state_dict()
	checkpoint = { 'state_dict': state_dict }
	torch.save(checkpoint,checkpoint_path)
	ordered_dict = OrderedDict()
	for key, value in state_dict.items():
		if isinstance(value, Tensor):
			ordered_dict[key] = value.detach().cpu().tolist()
	with open(json_path,'w') as file:
	 	json.dump(ordered_dict, file, indent=4)

discount = 0
batch_size = 2000 # 100000
batch_count = (len(dataset) // batch_size) + 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)
epoch_count = 500000
step_count = 100000

for step in range(step_count):
	for epoch in range(epoch_count):
		for batch, cpu_data in enumerate(dataloader):
			optimizer.zero_grad()
			data: Tensor = cpu_data.to(device)
			present = data[:,0:16]
			output = model(present)
			present_score = score(present)
			present_scores = present_score.repeat(1,81)
			future = data[:,16:].reshape(-1,16)
			future_scores = score(future).reshape(-1,81) 
			rewards = future_scores - present_scores
			with torch.no_grad():
				future_values = old_model.value(future).reshape(-1,81)
			target = rewards + discount*future_values
			loss = F.mse_loss(output, target)
			loss.backward()
			optimizer.step()
			L = loss.detach().cpu().numpy()
			print(f'Step {step+1}, Epoch {epoch+1} / {epoch_count}, Batch {batch+1:02d} / {batch_count}, Loss: {L}')
			if batch % 100 == 0: save_checkpoint()
	save_checkpoint()
	old_model.load_state_dict(model.state_dict())


# batch_size = 3
# data = dataset[0:batch_size].to(device)
# present = data[:,0:16]
# output = model(present)
# present_score = score(present)
# present_scores = present_score.repeat(1,81)
# future = data[:,16:].reshape(-1,16)
# future_scores = score(future).reshape(-1,81) 
# rewards = future_scores - present_scores
# with torch.no_grad():
# 	future_values = old_model.value(future).reshape(-1,81)
# target = rewards + discount*future_values


plot_cpu_data = next(iter(dataloader))
optimizer.zero_grad()
plot_data: Tensor = cpu_data.to(device)
plot_data = dataset[0:batch_size].to(device)
present = plot_data[:,0:16]
output = model(present)
present_score = score(present)
present_scores = present_score.repeat(1,81)
future = plot_data[:,16:].reshape(-1,16)
future_scores = score(future).reshape(-1,81) 
rewards = future_scores - present_scores
with torch.no_grad():
	future_values = old_model.value(future).reshape(-1,81)
target = rewards + discount*future_values
x = output.detach().cpu().numpy()
y = target.detach().cpu().numpy()
x.shape
plt.ion()
plt.clf()
plt.axline(xy1=(0,0),xy2=(1,1),color=(0,0.8,0))
plt.scatter(x,y)
