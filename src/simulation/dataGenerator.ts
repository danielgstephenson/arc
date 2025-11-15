import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range, round, sample } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import { actionSpace } from '../actionSpace'
import fs from 'fs-extra'

export class DataGenerator extends Simulation {
  writeStream: fs.WriteStream
  oldState = range(17).map(_ => 0)
  newState = range(17).map(_ => 0)
  filePath = './data.csv'
  currentStep = 0
  maxStep = 4

  constructor () {
    super()
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.restart()
    this.writeStream = fs.createWriteStream(this.filePath, { flags: 'a' })
    // this.player = fighters[0]
  }

  getState (dt: number): number[] {
    const fighters = [...this.fighters.values()]
    const fp0 = fighters[0].body.getPosition()
    const fv0 = fighters[0].body.getLinearVelocity()
    const wp0 = fighters[0].weapon.body.getPosition()
    const wv0 = fighters[0].weapon.body.getLinearVelocity()
    const fp1 = fighters[1].body.getPosition()
    const fv1 = fighters[1].body.getLinearVelocity()
    const wp1 = fighters[1].weapon.body.getPosition()
    const wv1 = fighters[1].weapon.body.getLinearVelocity()
    return [
      fp0.x, fp0.y, fv0.x, fv0.y, wp0.x, wp0.y, wv0.x, wv0.y,
      fp1.x, fp1.y, fv1.x, fv1.y, wp1.x, wp1.y, wv1.x, wv1.y
    ]
  }

  preStep (dt: number): void {
    super.preStep(dt)
    const fighters = [...this.fighters.values()]
    this.oldState = this.model.getState(fighters[0], fighters[1])
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
    this.newState = this.model.getState(fighters[0], fighters[1])
    this.writeData(dt)
    this.currentStep += 1
    if (this.currentStep > this.maxStep) {
      this.restart()
    }
  }

  writeData (dt: number): void {
    const fighters = [...this.fighters.values()]
    const data = [
      ...this.oldState,
      dt, fighters[0].action, fighters[1].action,
      ...this.newState
    ]
    const roundData = data.map(x => round(x, 6))
    const dataString = roundData.join(',') + '\n'
    this.writeStream.write(dataString)
  }

  restart (): void {
    super.restart()
    this.currentStep = 0
    const fighters = [...this.fighters.values()]
    const spawnDistance = 10
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
      fighter.action = sample(range(actionSpace.length))
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
