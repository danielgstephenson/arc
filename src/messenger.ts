import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { Simulation } from './simulation/simulation'
import { Trial } from './simulation/trial'

export class Messenger {
  server: Server
  io: SocketIoServer
  simulation: Simulation

  constructor (server: Server) {
    console.log('messenger')
    this.simulation = new Trial()
    this.server = server
    this.io = new SocketIoServer(server.httpServer)
    this.setupIo()
  }

  setupIo (): void {
    this.io.on('connection', socket => {
      socket.emit('connected')
      console.log(socket.id, 'connected')
      socket.on('input', (action: number) => {
        if (this.simulation.player != null) {
          this.simulation.player.action = action
        }
        socket.emit('summary', this.simulation.summary)
      })
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
      socket.on('parameters', parameters => {
        this.simulation.model.weight = parameters.weight as number[][][]
        this.simulation.model.bias = parameters.bias as number[][]
        // const testState = range(16)
        // const testValue = this.simulation.model.evaluate(testState)
        // console.log('test_value', testValue)
      })
    })
  }
}
