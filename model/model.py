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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concatenate(([[0,0]],move_vector_array)).tolist()
action_vectors = torch.tensor(action_array,dtype=torch.float64).to(device)
action_pairs = torch.tensor([[i,j] for i in range(9) for j in range(9)]).to(device)

def get_distance(a: Tensor, b: Tensor):
	return torch.sqrt((b[:,0]-a[:,0])**2+(b[:,1]-a[:,1])**2)

def score(state0: Tensor, state1: Tensor) -> Tensor:
	position0 = state0[:,0:2]
	position1 = state1[:,0:2]
	dist0 = torch.sqrt(torch.sum(position0**2,dim=1))
	dist1 = torch.sqrt(torch.sum(position1**2,dim=1))
	return (dist1 - dist0).unsqueeze(1)

class MyDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		self.state00: Tensor = torch.tensor(df.iloc[:, 0:8].to_numpy(),dtype=torch.float64)
		self.state10: Tensor = torch.tensor(df.iloc[:, 8:16].to_numpy(),dtype=torch.float64)
		self.action0: Tensor = torch.tensor(df.iloc[:, 16:18].to_numpy(),dtype=torch.float64)
		self.action1: Tensor = torch.tensor(df.iloc[:, 18:20].to_numpy(),dtype=torch.float64)
		self.state01: Tensor = torch.tensor(df.iloc[:, 20:28].to_numpy(),dtype=torch.float64)
		self.state11: Tensor = torch.tensor(df.iloc[:, 28:36].to_numpy(),dtype=torch.float64)
		self.count = df.shape[0]

	def __len__(self):
		return self.count

	def __getitem__(self, idx):
		s00 = self.state00[idx]
		s10 = self.state10[idx]
		a0 = self.action0[idx]
		a1 = self.action1[idx]
		s01 = self.state01[idx]
		s11 = self.state11[idx]
		return s00, s10, a0, a1, s01, s11

dataset = MyDataset(df)

class ActionValueModel(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		# self.activation = torch.nn.ReLU()
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
	def forward(self, s0: Tensor, s1: Tensor, a0: Tensor, a1: Tensor) -> Tensor:
		input = torch.cat((s0,s1,a0,a1),dim=1)
		return self.core(input)
	def core(self, input: Tensor) -> Tensor:
		layer0: Tensor = self.activation(self.h0(input))
		layer1: Tensor = self.activation(self.h1(layer0))
		layer2: Tensor = self.activation(self.h2(layer1))
		layer3: Tensor = self.activation(self.h3(layer2))
		return self.h4(layer3)
	def value(self, s0: Tensor, s1: Tensor) -> Tensor:
		return torch.vmap(self.maximin)(s0, s1).unsqueeze(1)
	def maximin(self, s0: Tensor, s1: Tensor) -> Tensor:
		action_values = self.action_value_matrix(s0, s1)
		mins = torch.min(action_values,dim=1).values
		return torch.max(mins)
	def action_value_matrix(self, s0: Tensor, s1: Tensor) -> Tensor:
		s0 = s0.repeat(81,1)
		s1 = s1.repeat(81,1)
		i0 = action_pairs[:,0]
		i1 = action_pairs[:,1]
		a0 = action_vectors[i0,:]
		a1 = action_vectors[i1,:]
		input = torch.cat((s0,s1,a0,a1),dim=1)
		output = self.core(input)
		return output.reshape(9,9)

# os.system('clear')

old_model = ActionValueModel().to(device)
model = ActionValueModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

discount = 0.1
dt = 0.04
batch_size = 10000 # 100000
batch_count = (len(dataset) // batch_size) + 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)
epoch_count = 500000
step_count = 100000

for step in range(step_count):
	for epoch in range(epoch_count):
		for batch, (s00, s10, a0, a1, s01, s11) in enumerate(dataloader):
			if batch + 1 == batch_count: break
			optimizer.zero_grad()
			s00: Tensor = s00.to(device)
			s10: Tensor = s10.to(device)
			a0: Tensor = a0.to(device)
			a1: Tensor = a1.to(device)
			s01: Tensor = s01.to(device)
			s11: Tensor = s11.to(device)
			output: Tensor = model(s00,s10,a0,a1)
			score0 = score(s00,s10)
			score1 = score(s01,s11)
			reward = score1 - score0
			with torch.no_grad():
				target = reward
				# target = reward + (1 - dt*discount)*old_model.value(s01,s11)
			loss = F.mse_loss(output, target)
			loss.backward()
			optimizer.step()
			L = loss.detach().cpu().numpy()
			print(f'Step {step+1}, Epoch {epoch+1} / {epoch_count}, Batch {batch+1:02d} / {batch_count}, Loss: {L}')
			if batch % 100 == 0: save_checkpoint()
	save_checkpoint()
	old_model.load_state_dict(model.state_dict())

(s00, s10, a0, a1, s01, s11) = next(iter(dataloader))
s00: Tensor = s00.to(device)
s10: Tensor = s10.to(device)
a0: Tensor = a0.to(device)
a1: Tensor = a1.to(device)
s01: Tensor = s01.to(device)
s11: Tensor = s11.to(device)
output: Tensor = model(s00,s10,a0,a1).squeeze(1)
score0 = score(s00,s10)
score1 = score(s01,s11)
reward = score1 - score0
with torch.no_grad():
	target = reward + (1-dt*discount)*old_model.value(s01,s11)
target = target.squeeze(1)
x = output.detach().cpu().numpy()
y = target.detach().cpu().numpy()
plt.ion()
plt.clf()
plt.axline(xy1=(0,0),xy2=(1,1),color=(0,0.8,0))
plt.scatter(x,y)
