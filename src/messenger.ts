import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { Simulation } from './simulation'
import { Fighter } from './entities/fighter'
import { Vec2 } from 'planck'
import { normalize } from './math'
import { Model } from './model'

export class Messenger {
  server: Server
  io: SocketIoServer
  simulation = new Simulation()
  model = new Model()

  constructor (server: Server) {
    this.server = server
    this.simulation = new Simulation()
    this.io = new SocketIoServer(server.httpServer)
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
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
      socket.on('parameters', parameters => {
        for (const name in parameters) {
          console.log(name)
        }
        this.model.weight = parameters.weight as number[][][]
        this.model.bias = parameters.bias as number[][]
        const states = [
          [1, 1, 1, 1, 2, 2, 2, 2]
        ]
        const values = states.map(state => {
          return this.model.evaluate(state)
        })
        console.log('values', values)
      })
    })
  }
}
