import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range, sample } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import { actionVectors } from '../actionVectors'
import fs from 'fs-extra'

export class DataGenerator extends Simulation {
  writeStream: fs.WriteStream
  filePath = './data.csv'
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
    this.reset()
    this.writeStream = fs.createWriteStream(this.filePath, { flags: 'a' })
    this.generate()
  }

  generate (): void {
    this.step()
    const fighters = [...this.fighters.values()]
    const spawnDistance = 40
    const spawnReach = 10
    const fighterPositions = range(2).map(_ => {
      const distance = spawnDistance * Math.random()
      return Vec2.mul(distance, randomDir())
    })
    const fighterVelocities = range(2).map(_ => {
      const speed = 7 * Math.random()
      return Vec2.mul(speed, randomDir())
    })
    const weaponPositions = range(2).map(i => {
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
    const s0 = this.getState(fighters[0])
    const s1 = this.getState(fighters[1])
    const data = [...s0, ...s1]
    range(9).forEach(a0 => {
      range(9).forEach(a1 => {
        const a = [a0, a1]
        range(2).forEach(i => {
          fighters[i].body.setPosition(fighterPositions[i])
          fighters[i].body.setLinearVelocity(fighterVelocities[i])
          fighters[i].weapon.body.setPosition(weaponPositions[i])
          fighters[i].weapon.body.setLinearVelocity(weaponVelocities[i])
          fighters[i].action = a[i]
        })
        this.step()
        const s0 = this.getState(fighters[0])
        const s1 = this.getState(fighters[1])
        data.push(...s0, ...s1)
      })
    })
    // const roundData = data.map(x => round(x, 8))
    const dataString = data.join(',') + '\n'
    this.writeStream.write(dataString)
    setTimeout(() => this.generate(), 0)
  }

  preStep (dt: number): void {
    super.preStep(dt)
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    fighters.forEach((fighter, i) => {
      if (fighter.dead) {
        // this.respawn(fighter)
      }
    })
  }

  respawn (fighter: Fighter): void {
    super.respawn(fighter)
    const spawnRadius = Math.min(50, Arena.size) - Blade.radius
    fighter.spawnPoint = Vec2.mul(spawnRadius, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }

  reset (): void {
    super.reset()
    const fighters = [...this.fighters.values()]
    const rand = Math.random()
    const spawnDistance = rand < 0.5 ? 20 : 60
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
}
