import asyncio
import socketio

sio = socketio.AsyncClient(handle_sigint=False)

@sio.event
async def connect():
	print('connected')

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