from math import pi
from pickle import TRUE
import torch
from torch import Tensor, nn, tensor
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import r_, sin, cos
import os

os.system('clear')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=False)

move_vector_array = [
	[np.round(cos(i/4*pi),6), np.round(sin(i/4*pi),6)] 
	for i in range(8)]
action_array = np.concat(([[0,0]],move_vector_array)).tolist()
action_vectors = torch.tensor(action_array,dtype=torch.float64).to(device)
action_pairs = torch.tensor([[i,j] for i in range(9) for j in range(9)]).to(device)

def get_distance(a: Tensor, b: Tensor):
	return torch.sqrt((b[:,0]-a[:,0])**2+(b[:,1]-a[:,1])**2)

df = pd.read_csv('data.csv', header=None)
# df.iloc[0,:]

def reward(state0: Tensor, state1: Tensor) -> Tensor:
	p0 = state0[:,0:2]
	p1 = state1[:,0:2]
	dist0 = torch.sqrt(torch.sum(p0**2,dim=1))
	dist1 = torch.sqrt(torch.sum(p1**2,dim=1))
	return (dist1 - dist0).unsqueeze(1)

class MyDataset(Dataset):
	def __init__(self, df: pd.DataFrame):
		assert df.shape[1] == 35, 'df.shape[1] != 35'
		self.state00 = torch.tensor(df.iloc[:, 0:8].to_numpy(),dtype=torch.float64)
		self.state10 = torch.tensor(df.iloc[:, 8:16].to_numpy(),dtype=torch.float64)
		self.dt = torch.tensor(df.iloc[:, 16].to_numpy(),dtype=torch.float64).unsqueeze(1)
		self.action0 = action_vectors[df.iloc[:, 17].to_numpy()]
		self.action1 = action_vectors[df.iloc[:, 18].to_numpy()]
		self.state01 = torch.tensor(df.iloc[:, 19:27].to_numpy(),dtype=torch.float64)
		self.state11 = torch.tensor(df.iloc[:, 27:35].to_numpy(),dtype=torch.float64)
		self.reward = reward(self.state01,self.state11)

	def get_jump_prob(self):
		p00 = self.state00[:,0:2]
		p10 = self.state10[:,0:2]
		p01 = self.state01[:,0:2]
		p11 = self.state11[:,0:2]
		dp0 = p01 - p00
		dp1 = p11 - p10
		dist0 = torch.sqrt(torch.sum(dp0**2,dim=1))
		dist1 = torch.sqrt(torch.sum(dp1**2,dim=1))
		n0 = torch.sum(dist0 > 10)
		n1 = torch.sum(dist1 > 10)
		n = len(self)
		return (n0 + n1) / n

	def __len__(self):
		return len(self.dt)

	def __getitem__(self, idx):
		s00 = self.state00[idx]
		s10 = self.state10[idx]
		dt = self.dt[idx]
		a0 = self.action0[idx]
		a1 = self.action1[idx]
		r = self.reward[idx]
		s01 = self.state01[idx]
		s11 = self.state01[idx]
		return s00, s10, dt, a0, a1, r, s01, s11

dataset = MyDataset(df)
dataset.get_jump_prob()

class ActionValue(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
		n0 = 20
		n1 = 100
		n2 = 100
		n3 = 100
		n4 = 100
		self.h1 = nn.Linear(n0, n1)
		self.h2 = nn.Linear(n1, n2)
		self.h3 = nn.Linear(n2, n3)
		self.h4 = nn.Linear(n3, n4)
		self.out = nn.Linear(n4, 1)
	def forward(self, s0: Tensor, s1: Tensor, a0: Tensor, a1: Tensor):
		input = torch.cat((s0,s1,a0,a1),dim=1)
		return self.core(input)
	def core(self, input: Tensor) -> Tensor:
		layer1: Tensor = self.activation(self.h1(input))
		layer2: Tensor = self.activation(self.h2(layer1))
		layer3: Tensor = self.activation(self.h3(layer2))
		layer4: Tensor = self.activation(self.h4(layer3))
		return self.out(layer4)
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
	def zero_(self):
		with torch.no_grad():
			for param in self.parameters():
				if param.requires_grad:  # Only modify learnable parameters
					param.zero_()


os.system('clear')


old_model = ActionValue().to(device)
old_model.zero_()
model = ActionValue().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
batch_size = 5000 # 10000
batch_count = (len(dataset) // batch_size) + 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

if os.path.exists('checkpoint.pt'):
	checkpoint = torch.load('checkpoint.pt')
	old_model.load_state_dict(checkpoint['state_dict'])
	model.load_state_dict(checkpoint['state_dict'])

def save_checkpoint():
	state_dict = model.state_dict()
	olds_state_dict = old_model.state_dict()
	checkpoint = { 
		'state_dict': state_dict,
		'old_state_dict': olds_state_dict
	}
	torch.save(checkpoint,'checkpoint.pt')

discount = 0.03
max_epoch = 10000000
max_step = 10000000
print_batch = np.round(len(dataset) / 100)
for step in range(max_step):
	for epoch in range(max_epoch):
		losses = np.array([])
		for batch, (s00, s10, dt, a0, a1, r, s01, s11) in enumerate(dataloader):
			optimizer.zero_grad()
			s00: Tensor = s00.to(device)
			s10: Tensor = s10.to(device)
			dt: Tensor = dt.to(device)
			a0: Tensor = a0.to(device)
			a1: Tensor = a1.to(device)
			r: Tensor = r.to(device)
			s01: Tensor = s01.to(device)
			s11: Tensor = s11.to(device)
			with torch.no_grad():
				v = old_model.value(s01, s11)
			target: Tensor = dt*r + (1-dt*discount)*v
			output: Tensor = model(s00,s10,a0,a1)
			loss = F.huber_loss(output, target)
			losses = np.append(losses,loss.detach().cpu().item())
			loss.backward()
			optimizer.step()
			L = np.format_float_positional(loss,6)
			print(f'Step {step+1}, Epoch {epoch+1}, Batch {batch} / {batch_count}, Loss: {L}')
		save_checkpoint()
		mean_loss = np.mean(losses)
		print('Mean Loss:', np.format_float_positional(mean_loss,6))
		if (mean_loss < 0.01):
			old_model.load_state_dict(model.state_dict())
			save_checkpoint()	
			optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
			break
