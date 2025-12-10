import { Vec2 } from 'planck'
import { randomDir, range, sample } from './math'
import { Imagination } from './simulation/imagination'
import { Server as SocketIoServer } from 'socket.io'

export class DataGenerator {
  imagination = new Imagination()
  io: SocketIoServer
  filePath = './model/data.bin'
  count = 0

  constructor (io: SocketIoServer) {
    this.io = io
    this.imagination.timeStep = 0.2
    console.log('DataGenerator')
    const fighter0 = this.imagination.addFighter(new Vec2(0, +10))
    const fighter1 = this.imagination.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.io.on('connection', socket => {
      socket.on('requestData', () => {
        socket.emit('data', this.generate())
      })
    })
  }

  generate (): Buffer<ArrayBuffer> {
    const data: number[] = []
    range(1000).forEach(_ => {
      this.reset()
      const fighters = [...this.imagination.fighters.values()]
      const fighterStates = [
        fighters[0].getState(),
        fighters[1].getState()
      ]
      const datum = [...fighterStates[0], ...fighterStates[1]]
      range(9).forEach(a0 => {
        range(9).forEach(a1 => {
          const a = [a0, a1]
          range(2).forEach(i => {
            fighters[i].setState(fighterStates[i])
            fighters[i].action = a[i]
          })
          this.imagination.step()
          const newFighterStates = [
            fighters[0].getState(),
            fighters[1].getState()
          ]
          datum.push(...newFighterStates[0], ...newFighterStates[1])
        })
      })
      data.push(...datum)
    })
    const float32Array = new Float32Array(data)
    const buffer = Buffer.from(float32Array.buffer)
    return buffer
  }

  reset (): void {
    const fighters = [...this.imagination.fighters.values()]
    const fighterPositions = range(2).map(_ => {
      const spawnDistance = sample([5, 10, 15, 20, 30, 50])
      const distance = spawnDistance * Math.random()
      return Vec2.mul(distance, randomDir())
    })
    const fighterVelocities = range(2).map(_ => {
      const speed = 7 * Math.random()
      return Vec2.mul(speed, randomDir())
    })
    const weaponPositions = range(2).map(i => {
      const spawnReach = 10
      const reach = spawnReach * Math.random()
      return Vec2.combine(1, fighterPositions[i], reach, randomDir())
    })
    const weaponVelocities = range(2).map(_ => {
      const weaponSpeed = 20 * Math.random()
      return Vec2.mul(weaponSpeed, randomDir())
    })
    range(2).forEach(i => {
      fighters[i].body.setPosition(fighterPositions[i])
      fighters[i].body.setLinearVelocity(fighterVelocities[i])
      fighters[i].weapon.body.setPosition(weaponPositions[i])
      fighters[i].weapon.body.setLinearVelocity(weaponVelocities[i])
      fighters[i].action = 0
    })
    this.imagination.step()
  }
}
