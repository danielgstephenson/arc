import socketio

print('model')

sio = socketio.Client()

@sio.event
def connected():
	print('connected')

sio.connect('http://localhost:3000')
sio.wait()
