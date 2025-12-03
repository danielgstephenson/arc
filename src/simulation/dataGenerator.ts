import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range, sample } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import fs from 'fs-extra'

export class DataGenerator extends Simulation {
  writeStream: fs.WriteStream
  filePath = './data.csv'

  constructor () {
    super()
    console.log('DataGenerator')
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.writeStream = fs.createWriteStream(this.filePath, { flags: 'a' })
    this.generate()
  }

  generate (): void {
    this.step()
    const fighters = [...this.fighters.values()]
    const spawnReach = 10
    const fighterPositions = range(2).map(_ => {
      const spawnDistance = sample([5, 10, 20, 50])
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
    const s00 = this.getState(fighters[0])
    const s10 = this.getState(fighters[1])
    const data = [...s00, ...s10]
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
        const s01 = this.getState(fighters[0])
        const s11 = this.getState(fighters[1])
        data.push(...s01, ...s11)
      })
    })
    const fixedDecimals = data.map(x => x.toFixed(7))
    const dataString = fixedDecimals.join(',') + '\n'
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
        this.respawn(fighter)
      }
    })
  }

  respawn (fighter: Fighter): void {
    super.respawn(fighter)
    const spawnRadius = Math.min(40, Arena.size) - Blade.radius
    fighter.spawnPoint = Vec2.mul(spawnRadius, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }
}
