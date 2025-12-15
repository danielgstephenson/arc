from math import isnan, pi
import torch
from torch import Tensor, chunk, nn, tensor
from torch.optim import Adam
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
        inputSize = 16
        k = 50
        self.hiddenCount = 15
        self.hiddenLayers = nn.ModuleList([nn.Linear(inputSize + i*k, k) for i in range(self.hiddenCount)])
        self.outputLayer = nn.Linear(inputSize + self.hiddenCount*k, 1)
    def forward(self, state: Tensor) -> Tensor:
        x = state
        for i in range(self.hiddenCount):
            h = self.hiddenLayers[i]
            y: Tensor = self.activation(h(x))
            x = torch.cat((x,y),dim=1)
        return self.outputLayer(x)
    def __call__(self, *args, **kwds) -> Tensor:
        return super().__call__(*args, **kwds)
    
def get_reward(states: Tensor)->Tensor:
    pos0 = states[:,0:2]
    pos1 = states[:,8:10]
    dist0 = torch.sqrt(torch.sum(pos0**2,dim=1))
    dist1 = torch.sqrt(torch.sum(pos1**2,dim=1))
    close = torch.tensor(5)
    dist0 = torch.maximum(dist0,close)
    swing0 = torch.sqrt(torch.sum(states[:,6:8]**2,dim=1))
    reward = dist1 - dist0 + 0.25*swing0
    return reward.unsqueeze(1)

def save_onnx(model: nn.Module, path: str):
    with contextlib.redirect_stdout(io.StringIO()):
        example_input = torch.tensor([[i for i in range(16)]],dtype=torch.float32).to(device)
        example_input_tuple = (example_input,)
        onnx_program = torch.onnx.export(model, example_input_tuple, dynamo=True)
        if onnx_program is not None:
            onnx_program.save(path)

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, path: str):
    checkpoint = { 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    try:
        torch.save(checkpoint, path)
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt detected. Saving checkpoint...')
        torch.save(checkpoint, path)
        print('Checkpoint saved.')
        raise

print('Loading checkpoints...')
steps = 50
models: list[ValueModel] = []
optimizers: list[torch.optim.Optimizer] = []
for step in range(steps):
    model = ValueModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    models.append(model)
    optimizers.append(optimizer)
    checkpoint_path = f'./checkpoints/checkpoint{step}.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

learning_rate = 0.0001
for optimizer in optimizers:
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

# for step in range(1,steps):
#      fileName = f'./onnx/model{step}.onnx'
#      print(f'saving {fileName} ...')
#      save_onnx(models[step],fileName)

# os.system('clear')
discount = 0.95
self_noise = 0.1
other_noise = 0.01
sio = socketio.SimpleClient()
sio.connect('http://localhost:3000')
sio.emit('requestData')
event = sio.receive()
bytes = bytearray(event[1])
data = torch.frombuffer(bytes,dtype=torch.float32).reshape(-1, 82*16).to(device)
sio.emit('requestData')

print('Training...')
for batch in range(10000000000):
    event = sio.receive()
    sio.emit('requestData')
    bytes = bytearray(event[1])
    new_data = torch.frombuffer(bytes,dtype=torch.float32).reshape(-1, 82*16).to(device)
    data = torch.cat((data[-9000:,:],new_data),dim=0)
    n = data.shape[0]
    message = f'Batch: {batch}, Losses:'
    for step in range(15):
        model = models[step]
        optimizer = optimizers[step]
        optimizer.zero_grad()
        states = data[:,0:16]
        output = model(states)
        reward = get_reward(states)
        potential_futures = data[:,16:]
        with torch.no_grad():
            n = potential_futures.shape[0]
            x = potential_futures.reshape(-1,16)        
            old_model = get_reward if step == 0 else models[step-1]
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
        message += f' {loss_value:.2f}'
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        save_checkpoint(model, optimizer, f'./checkpoints/checkpoint{step}.pt')
    print(message)