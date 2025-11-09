import asyncio
import socketio
import os
print('importing torch...')
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = " + str(device))
torch.set_default_device(device)
torch.set_default_dtype(torch.float64)

n0 = 16 # input size 
n1 = 100 # first hidden layer size
n2 = 100 # second hidden layer size
n3 = 100 # third hidden layer size

class Core(torch.nn.Module):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.activation = torch.nn.LeakyReLU(negative_slope=0.01)
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
		self.flip = [8, 9,10,11,12,13,14,15,0, 1, 2, 3, 4, 5, 6, 7]
		self.negX = torch.tensor([(-1)**(i+1) for i in range(16)])
		self.negY = torch.tensor([(-1)**i for i in range(16)])
		self.negXY = torch.tensor([-1 for _ in range(16)])
		self.ready = True
		self.state_list = []
		self.value_list = []
	def forward(self,state: torch.Tensor) -> torch.Tensor:
		s0 = state
		s1 = state * self.negX
		s2 = state * self.negY
		s3 = state * self.negXY
		output = self.base(s0) + self.base(s1) + self.base(s2) + self.base(s3)
		return output
	def base(self,state: torch.Tensor) -> torch.Tensor:
		s0 = state
		s1 = state[:,self.flip]
		result = self.core(s0) - self.core(s1)
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
# test_state = torch.tensor([[i * 1.0 for i in range(16)]])
# test_value = model(test_state)
# print('test_value',test_value)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def learn():
	if len(model.state_list) == 0: return
	model.ready = False
	states = torch.tensor(model.state_list, dtype=torch.float64)
	values = torch.tensor(model.value_list, dtype=torch.float64)
	model.state_list = []
	model.value_list = []
	model.train()
	optimizer.zero_grad()
	predictions = model(states)
	loss = F.mse_loss(predictions, values)
	loss.backward()
	optimizer.step()
	torch.save(model.state_dict(),'checkpoint.pt')
	model.ready = True

sio = socketio.AsyncClient(handle_sigint=False)

@sio.event
async def connect():
	print('connected')
	await sio.emit('parameters',model.serialize())

@sio.event
async def observe(data):
	model.state_list.append(data['state'])
	model.value_list.append(data['value'])
	if(model.ready):
		learn()
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