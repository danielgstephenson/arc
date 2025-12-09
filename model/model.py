from math import pi
import torch
from torch import Tensor, nn, tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import block, indices, r_, sin, cos
import os
import time
import io
import contextlib
import random

data_path = '../data.bin'
checkpoint_path = './checkpoint0.pt'
old_checkpoint_path = './checkpoint0.pt'
json_path = './parameters.json'

os.path.getmtime(data_path)

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
        k = 100
        self.h0 = nn.Linear(n0, k)
        self.h1 = nn.Linear(n0 + 1*k, k)
        self.h2 = nn.Linear(n0 + 2*k, k)
        self.h3 = nn.Linear(n0 + 3*k, k)
        self.h4 = nn.Linear(n0 + 4*k, k)
        self.h5 = nn.Linear(n0 + 5*k, k)
        self.h6 = nn.Linear(n0 + 6*k, 1)
    def forward(self, state: Tensor) -> Tensor:
        y0: Tensor = self.activation(self.h0(state))
        x1 = torch.cat((state, y0), dim=1)
        y1: Tensor = self.activation(self.h1(x1))
        x2 = torch.cat((x1, y1), dim=1)
        y2: Tensor = self.activation(self.h2(x2))
        x3 = torch.cat((x2, y2), dim=1)
        y3: Tensor = self.activation(self.h3(x3))
        x4 = torch.cat((x3, y3), dim=1)
        y4: Tensor = self.activation(self.h4(x4))
        x5 = torch.cat((x4, y4), dim=1)
        y5: Tensor = self.activation(self.h5(x5))
        x6 = torch.cat((x5, y5), dim=1)
        return self.h6(x6)
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
def get_reward(states: Tensor)->Tensor:
    pos0 = states[:,0:2]
    pos1 = states[:,8:10]
    dist0 = torch.sqrt(torch.sum(pos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(pos1**2,dim=1))
    return (dist1 - dist0).unsqueeze(1)

def save_checkpoint():
    with contextlib.redirect_stdout(io.StringIO()):
        checkpoint = { 'state_dict': best_state_dict }
        torch.save(checkpoint,checkpoint_path)
        # example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
        # example_input_tuple = (example_input,)
        # onnx_program = torch.onnx.export(model, example_input_tuple, dynamo=True)
        # if onnx_program is not None:
        #     onnx_program.save('model.onnx')

print('Loading Data...')
data_array = np.fromfile(data_path,dtype=np.float32,count=-1,offset=0).reshape(-1, 82*16)
data = torch.tensor(data_array,dtype=torch.float32)
dataset = TensorDataset(data)
batch_size = 20000
batch_count = (len(dataset) // batch_size)
dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
print('Data Loaded')

states = data[:,0:16]
n = states.shape[0]
idxs = torch.tensor([i for i in range(n)])
idx = random.randint(0,n)
state = states[idx,:]
others = states[idxs != idx, :]
differences = others - state.unsqueeze(0).repeat(n-1,1)
distances = torch.sqrt(torch.sum(differences**2,dim=1))
sup_distances = torch.max(torch.abs(differences),dim=1)[0]
sup_distances.shape
min_index = torch.argmin(sup_distances)
sup_distances[min_index]
state - others[min_index]
# torch.cat((state.unsqueeze(1),others[min_index].unsqueeze(1)),dim=1)

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
discount = 0.9
step = 1
best_mse = 1000000
step_epoch = 0
best_count = 0
best_state_dict = model.state_dict()

print('Training...')
for epoch in range(10000000):
    for batch, batch_data in enumerate(dataloader):
        data: Tensor = batch_data[0].to(device)
        n = data.shape[0]
        optimizer.zero_grad()
        states = data[:,0:16]
        output = model(states)
        reward = get_reward(output)
        potential_futures = data[:,16:]
        with torch.no_grad():
            n = potential_futures.shape[0]
            x = potential_futures.reshape(-1,16)
            potential_values = get_reward(x) # old_model(x)
            value_matrices = potential_values.reshape(n,9,9)
            mins = torch.amin(value_matrices,2)
            future_value = torch.amax(mins,1).unsqueeze(1)
        target = (1-discount)*reward + discount*future_value
        loss = F.mse_loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()
        batch_mse = loss.detach().cpu().numpy()
        if batch_mse < best_mse:
            best_mse = batch_mse
            best_state_dict = model.state_dict()
            save_checkpoint()
            best_count = 0
        else:
            best_count += 1
        message = ''
        message += f'Epoch {epoch+1}, '
        message += f'Batch {batch+1:03} / {batch_count}, '
        message += f'Loss: {batch_mse:.5f}, '
        message += f'Best: {best_mse:.5f}, '
        message += f'Count: {best_count:03}'
        print(message)