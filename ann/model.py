import asyncio
import socketio
import os
print('importing torch...')
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

n0 = 8 # input size 
n1 = 100 # first hidden layer size
n2 = 100 # second hidden layer size
n3 = 100 # third hidden layer size


class Core(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.ReLU()
		self.h1 = torch.nn.Linear(n0, n1)
		self.h2 = torch.nn.Linear(n1, n2)
		self.h3 = torch.nn.Linear(n2, n3)
		self.out = torch.nn.Linear(n3, 1)
	def forward(self,state: torch.Tensor) -> torch.Tensor:
		layer1 = self.activation(self.h1(state))
		layer2 = self.activation(self.h2(layer1))
		layer3 = self.activation(self.h3(layer2))
		output = self.out(layer3)
		return output

class Evaluator(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.core = Core()
	def forward(self,state: torch.Tensor) -> torch.Tensor:
		state0 = state
		state1 = state[:,[4,5,6,7,0,1,2,3]]
		result = self.core(state0) - self.core(state1)
		output = result.squeeze(1) if result.dim() > 1 else state
		return output
	def serialize(self):
		state_dict = self.state_dict()
		weight = [
			state_dict[key].cpu().tolist()
			for key in ['core.h1.weight','core.h2.weight','core.h3.weight','core.out.weight']
		]
		bias = [
			state_dict[key].cpu().tolist()
			for key in ['core.h1.bias','core.h2.bias','core.h3.bias','core.out.bias']
		]
		return {'weight': weight, 'bias': bias}
	
model = Evaluator()
if os.path.exists('checkpoint.pt'):
	print('loading checkpoint...')
	checkpoint = torch.load('checkpoint.pt')
	model.load_state_dict(checkpoint)
torch.save(model.state_dict(),'checkpoint.pt')
states = torch.tensor([
	[1,1,1,1,2,2,2,2],
	[2,2,2,2,1,1,1,1]
], dtype=torch.float64)
values = model.forward(states)
print('values',values.cpu().tolist())

sio = socketio.AsyncClient(handle_sigint=False)

@sio.event
async def connect():
	print('connected')
	await sio.emit('parameters',model.serialize())

@sio.event
async def disconnect():
    print('disconnected from server')

async def main():
	try:
		await sio.connect('http://localhost:3000', wait=True)
		await sio.wait()
	except asyncio.CancelledError:
		await sio.disconnect()
		print('Shutdown Complete')

try:
	asyncio.run(main())
except KeyboardInterrupt:
	print('KeyboardInterrupt')