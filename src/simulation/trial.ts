import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'
import { Model } from '../model'

export class Trial extends Simulation {
  model = new Model()
  timeStep = 0.04
  currentStep = 0
  maxStep = 4
  bot: Fighter
  player: Fighter
  maxTorsoSpeed = 0
  maxWeaponSpeed = 0

  constructor () {
    super()
    console.log('Trial')
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.reset()
    this.player = fighter0
    this.bot = fighter1
    this.start()
  }

  preStep (dt: number): void {
    super.preStep(dt)
    this.bot.action = this.model.action
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
    const botState = this.bot.getState()
    const playerState = this.player.getState()
    const state = [...botState, ...playerState]
    void this.model.update(state)
    this.currentStep += 1
  }

  reset (): void {
    super.reset()
    this.currentStep = 0
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      fighter.spawnPoint = Vec2.mul(0.5, randomDir())
      fighter.body.setPosition(fighter.spawnPoint)
      this.respawn(fighter)
    })
  }

  respawn (fighter: Fighter): void {
    super.respawn(fighter)
    const spawnRadius = Math.min(15, Arena.size) - Blade.radius
    const position = fighter.body.getPosition()
    const dir = Vec2.normalize(position)
    fighter.spawnPoint = Vec2.combine(1, position, spawnRadius, dir)
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }
}
