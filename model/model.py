from math import isnan, pi
import torch
from torch import Tensor, chunk, nn, tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import sin, cos
import os
import io
import contextlib
import socketio
from typing import Any
import math

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
    return (dist1 - 0.1*dist0).unsqueeze(1)

def save_onnx(model: nn.Module, filePath: str):
    with contextlib.redirect_stdout(io.StringIO()):
        example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
        example_input_tuple = (example_input,)
        onnx_program = torch.onnx.export(model, example_input_tuple, dynamo=True)
        if onnx_program is not None:
            onnx_program.save(filePath)

steps = 50
models: list[Any] = [get_reward]
optimizers: list[Any] = [0]
for step in range(1,steps):
    model = ValueModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    models.append(model)
    optimizers.append(optimizer)
    checkpoint_path = f'./checkpoints/checkpoint{step}.pt'
    if os.path.exists(checkpoint_path):
        print(f'Loading {checkpoint_path} ...')
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

for step in range(1,steps):
    fileName = f'./onnx/model{step}.onnx'
    print(f'saving {fileName} ...')
    save_onnx(models[step],fileName)

# os.system('clear')
discount = 1
self_noise = 0
other_noise = 0.05
sio = socketio.SimpleClient()
sio.connect('http://localhost:3000')
sio.emit('requestData')

print('Training...')
for batch in range(10000000000):
    event = sio.receive()
    sio.emit('requestData')
    data_bytearray = bytearray(event[1])
    data = torch.frombuffer(data_bytearray,dtype=torch.float32)
    data = data.reshape(-1, 82*16).to(device)
    n = data.shape[0]
    message = f'Batch: {batch}, Losses:'
    for step in range(1,steps):
        model: ValueModel = models[step]
        optimizer: torch.optim.Optimizer = optimizers[step]
        optimizer.zero_grad()
        states = data[:,0:16]
        output = model(states)
        reward = get_reward(output)
        potential_futures = data[:,16:]
        with torch.no_grad():
            n = potential_futures.shape[0]
            x = potential_futures.reshape(-1,16)        
            old_model = models[step-1]
            potential_values = old_model(x)
            value_matrices = potential_values.reshape(n,9,9)
            means = torch.mean(value_matrices,2)
            mins = torch.amin(value_matrices,2)
            action_values = other_noise*means + (1-other_noise)*mins
            max_value = torch.amax(action_values,1).unsqueeze(1)
            average_value = torch.mean(action_values,1).unsqueeze(1)
            future_state_value = self_noise*average_value + (1-self_noise)*max_value
        target = (1-discount)*reward + discount*future_state_value
        loss = F.mse_loss(output, target, reduction='mean')
        loss_value = loss.detach().cpu().numpy()
        if math.isnan(loss_value):
            continue
        message += f' {loss_value:.1f}'
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        optimizer.step()
        checkpoint = { 'state_dict': model.state_dict() }
        try:
            torch.save(checkpoint, f'./checkpoints/checkpoint{step}.pt')
        except KeyboardInterrupt:
            print('\nKeyboardInterrupt detected. Saving checkpoint...')
            torch.save(checkpoint, f'./checkpoints/checkpoint{step}.pt')
            print('Checkpoint saved.')
            raise
    print(message)