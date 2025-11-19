import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range, round, sample } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import { actionVectors } from '../actionVectors'
import fs from 'fs-extra'

export class DataGenerator extends Simulation {
  writeStream: fs.WriteStream
  oldState0 = range(8).map(_ => 0)
  oldState1 = range(8).map(_ => 0)
  filePath = './data.csv'
  deathCount = 0
  maxStep = 4

  constructor () {
    super()
    console.log('DataGenerator')
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.restart()
    this.writeStream = fs.createWriteStream(this.filePath, { flags: 'a' })
    // this.player = fighter0
  }

  preStep (dt: number): void {
    super.preStep(dt)
    const fighters = [...this.fighters.values()]
    this.oldState0 = this.getState(fighters[0])
    this.oldState1 = this.getState(fighters[1])
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    this.deathCount = 0
    fighters.forEach(fighter => {
      if (fighter.dead) {
        this.deathCount += 1
        this.respawn(fighter)
      }
    })
    this.writeData(dt)
    this.restart()
  }

  writeData (dt: number): void {
    const fighters = [...this.fighters.values()]
    const actionVector0 = actionVectors[fighters[0].action]
    const actionVector1 = actionVectors[fighters[1].action]
    const s00 = this.oldState0
    const s10 = this.oldState0
    const a0 = [actionVector0.x, actionVector0.y]
    const a1 = [actionVector1.x, actionVector1.y]
    const s01 = this.getState(fighters[0])
    const s11 = this.getState(fighters[1])
    const data = [...s00, ...s10, ...a0, ...a1, ...s01, ...s11, this.deathCount]
    const roundData = data.map(x => round(x, 6))
    const dataString = roundData.join(',') + '\n'
    this.writeStream.write(dataString)
  }

  restart (): void {
    super.restart()
    const fighters = [...this.fighters.values()]
    const spawnDistance = 40
    const spawnReach = 10
    fighters.forEach(fighter => {
      this.respawn(fighter)
      const fighterDistance = spawnDistance * Math.random()
      const fighterPosition = Vec2.mul(fighterDistance, randomDir())
      fighter.body.setPosition(fighterPosition)
      const fighterSpeed = 7 * Math.random()
      const fighterVelocity = Vec2.mul(fighterSpeed, randomDir())
      fighter.body.setLinearVelocity(fighterVelocity)
      const reach = spawnReach * Math.random()
      const weaponPosition = Vec2.combine(1, fighterPosition, reach, randomDir())
      fighter.weapon.body.setPosition(weaponPosition)
      const weaponSpeed = 20 * Math.random()
      const weaponVelocity = Vec2.mul(weaponSpeed, randomDir())
      fighter.weapon.body.setLinearVelocity(weaponVelocity)
      fighter.action = sample(range(actionVectors.length))
    })
  }

  respawn (fighter: Fighter): void {
    super.respawn(fighter)
    const spawnRadius = Math.min(30, Arena.size) - Blade.radius
    fighter.spawnPoint = Vec2.mul(spawnRadius, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }
}
