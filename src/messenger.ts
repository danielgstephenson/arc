import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { Simulation } from './simulation'
import { Fighter } from './entities/fighter'
import { Vec2 } from 'planck'
import { normalize } from './math'
import { ChildProcessWithoutNullStreams, spawn } from 'child_process'

export class Messenger {
  server: Server
  io: SocketIoServer
  simulation: Simulation
  model: ChildProcessWithoutNullStreams

  constructor (server: Server) {
    this.server = server
    this.simulation = new Simulation()
    this.io = new SocketIoServer(server.httpServer)
    this.model = spawn('python', ['./model.py'])
    this.model.stdin.write('hello')
    this.setupIo()
  }

  setupIo (): void {
    this.io.on('connection', socket => {
      socket.emit('connected')
      console.log(socket.id, 'connected')
      socket.on('input', (vector: Vec2) => {
        const moveDir = normalize(vector)
        this.simulation.playerForce = Vec2.mul(Fighter.movePower, moveDir)
        socket.emit('summary', this.simulation.summary)
      })
    })
  }
}
