import { Server } from './server'
import { Server as SocketIoServer } from 'socket.io'
import { Simulation } from './simulation/simulation'
import { Imagination } from './simulation/imagination'
import { Trial } from './simulation/trial'
import { DataGenerator } from './dataGenerator'

export class Messenger {
  server: Server
  io: SocketIoServer
  simulation: Simulation

  constructor (server: Server) {
    console.log('messenger')
    this.io = new SocketIoServer(server.httpServer)
    // const dataGenerator = new DataGenerator(this.io)
    // this.simulation = dataGenerator.imagination
    this.simulation = new Trial()
    this.server = server
    this.setupIo()
  }

  setupIo (): void {
    this.io.on('connection', socket => {
      console.log(socket.id, 'connected')
      if (!(this.simulation instanceof Imagination)) {
        socket.emit('renderScale', this.server.config.renderScale)
      }
      socket.on('input', (action: number) => {
        if (this.simulation.player != null) {
          this.simulation.player.action = action
        }
        socket.emit('summary', this.simulation.summary)
      })
      socket.on('disconnect', () => {
        console.log(socket.id, 'disconnected')
      })
    })
  }
}
