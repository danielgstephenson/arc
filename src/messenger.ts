import { Server } from './server'

export class Messenger {
  server: Server

  constructor (server: Server) {
    this.server = server
  }
}
