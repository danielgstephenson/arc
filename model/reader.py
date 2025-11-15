from math import pi
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import sin, cos
import os

os.system('clear')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float32)
torch.set_printoptions(sci_mode=False)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concat(([[0,0]],move_vector_array)).tolist()

actions = torch.tensor(action_array,dtype=torch.float32).to(device)
action_pairs = torch.tensor([
	[action_array[i], action_array[j]]
	for j in range(9) for i in range(9)
],dtype=torch.float32).to(device)

rotations = torch.tensor([
	[
		[np.round(cos(i/4*pi),6),np.round(-sin(i/4*pi),6)],
		[np.round(sin(i/4*pi),6),np.round( cos(i/4*pi),6)]
	]
	for i in range(8)
], dtype=torch.float32)

reflections = torch.tensor([
	[
		[1,0],
		[0,1]
	],
	[
		[0,1],
		[1,0]
	],
], dtype=torch.float32)

transforms = torch.stack(tuple([
	torch.matmul(rotations[i],reflections[j])
	for i in range(8) for j in range(2)
])).to(device)

def get_distance(a: Tensor, b: Tensor):
	return torch.sqrt((b[:,0]-a[:,0])**2+(b[:,1]-a[:,1])**2)

df = pd.read_csv('data.csv', header=None)

class MyDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		assert df.shape[1] == 35, 'df.shape[1] != 35'
		s0 = torch.tensor(df.iloc[:, 0:16].to_numpy(),dtype=torch.float32)
		s1 = torch.tensor(df.iloc[:, 19:35].to_numpy(),dtype=torch.float32)
		self.old_state = torch.stack(tuple(s0[:,i:(i+2)] for i in range(8)))
		self.new_state = torch.stack(tuple(s1[:,i:(i+2)] for i in range(8)))
		self.dt = torch.tensor(df.iloc[:, 16].to_numpy(),dtype=torch.float32)
		a0 = torch.tensor(df.iloc[:, 17].to_numpy(),dtype=torch.int)
		a1 = torch.tensor(df.iloc[:, 18].to_numpy(),dtype=torch.int)
		self.action0 = actions[a0]
		self.action1 = actions[a1]

	def get_jump_prob(self):
		dist0 = get_distance(self.old_state[0],self.new_state[0])
		dist1 = get_distance(self.old_state[4],self.new_state[4])
		n0 = torch.sum(dist0 > 10)
		n1 = torch.sum(dist1 > 10)
		n = len(self)
		return (n0 + n1) / n

	def __len__(self):
		return len(self.dt)

	def __getitem__(self, idx):
		old_state = self.old_state[:,idx]
		dt = self.dt[idx]
		action0 = self.action0[idx]
		action1 = self.action1[idx]
		new_state = self.new_state[:,idx]
		return old_state, dt, action0, action1, new_state

dataset = MyDataset(df)

batch_size = 4
dataloader = DataLoader(dataset, batch_size, shuffle=True)

for x in dataset[0]:
	print(x)

dataset.get_jump_prob()

class ActionValue(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		n0 = 20
		n1 = 50
		n2 = 50
		n3 = 50
		n4 = 50
		self.h1 = nn.Linear(n0, n1)
		self.h2 = nn.Linear(n1, n2)
		self.h3 = nn.Linear(n2, n3)
		self.h4 = nn.Linear(n3, n4)
		self.out = nn.Linear(n4, 1)
	def forward(self, state: Tensor, action0:Tensor, action1:Tensor) -> Tensor:
		state = state.reshape(-1,16)
		input = torch.cat((state, action0, action1),dim=1)
		layer1 = self.activation(self.h1(input))
		layer2 = self.activation(self.h2(layer1))
		layer3 = self.activation(self.h3(layer2))
		layer4 = self.activation(self.h4(layer3))
		output = self.out(layer4)
		return output

# os.system('clear')

Q = ActionValue().to(device)
optimizer = torch.optim.Adam(Q.parameters(), lr=0.01)

def norm (v: Tensor) -> Tensor:
	return torch.linalg.vector_norm()

discount = 15

old_state, dt, action0, action1, new_state= next(iter(dataloader))
old_state: Tensor = old_state.to(device)
dt: Tensor = dt.unsqueeze(1).to(device)
action0: Tensor = action0.to(device)
action1: Tensor = action0.to(device)
new_state: Tensor = new_state.to(device)
flipAgents = np.random.random() < 0.5
if(flipAgents):
	old_state = old_state[:,[4,5,6,7,0,1,2,3],:]
	old_state = old_state[:,[4,5,6,7,0,1,2,3],:]
	action0, action1 = action1, action0
transform = transforms[np.random.choice(16)].to(device)
old_state = torch.einsum('ij,bkj->bki',transform,old_state)
new_state = torch.einsum('ij,bkj->bki',transform,new_state)
action0 = torch.einsum('ij,bj->bi',transform,action0)
action1 = torch.einsum('ij,bj->bi',transform,action1)
dist0: Tensor = torch.linalg.vector_norm(old_state[:,0,:],dim=1)
dist1: Tensor = torch.linalg.vector_norm(old_state[:,4,:],dim=1)
reward = (dist1 - dist0).unsqueeze(1)
futureValue = reward # actual future value still needs to be implemented
targets = dt*reward + F.relu(1-dt*discount)*reward
optimizer.zero_grad()
outputs = Q(old_state, action0, action1)
loss = F.huber_loss(outputs, targets)
loss.backward()
optimizer.step()


action_pairs.shape

new_state.shape
new_state.repeat(9,9,1,1,1).shape

# action

# g = torch.vmap(f,in_dims=(0,1),out_dims=(0,1))
# g(action_pairs)
	