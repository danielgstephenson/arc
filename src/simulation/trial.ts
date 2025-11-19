import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range, sample } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import { actionVectors } from '../actionVectors'

export class Trial extends Simulation {
  filePath = './data.csv'
  currentStep = 0
  maxStep = 4
  bot: Fighter
  player: Fighter

  constructor () {
    super()
    console.log('Trial')
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.restart()
    this.player = fighter0
    this.bot = fighter1
  }

  preStep (dt: number): void {
    super.preStep(dt)
    const s0 = this.getState(this.bot)
    const s1 = this.getState(this.player)
    const a1 = this.player.action
    this.bot.action = this.model.getAction(s0, s1, a1)
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
    this.currentStep += 1
  }

  restart (): void {
    super.restart()
    this.currentStep = 0
    const fighters = [...this.fighters.values()]
    const spawnDistance = 20
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
