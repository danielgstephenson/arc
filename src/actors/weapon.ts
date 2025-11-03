import { Vec2 } from 'planck'
import { Actor } from './actor'
import { Fighter } from './fighter'
import { Blade } from '../features/blade'
import { dirFromTo } from '../math'

export class Weapon extends Actor {
  fighter: Fighter
  blade: Blade
  movePower = 4
  moveDir = new Vec2(0, 0)
  color = 'hsla(220, 50%, 40%, 0.5)'

  constructor (fighter: Fighter) {
    super(fighter.stage, {
      type: 'dynamic',
      bullet: true,
      linearDamping: 0.3,
      fixedRotation: true
    })
    this.label = 'weapon'
    this.fighter = fighter
    this.body.setPosition(this.fighter.body.getPosition())
    this.blade = new Blade(this)
  }

  preStep (dt: number): void {
    super.preStep(dt)
    const bladePoint = this.body.getPosition()
    const torsoPoint = this.fighter.body.getPosition()
    const distance = Vec2.distance(bladePoint, torsoPoint)
    const moveDir = dirFromTo(bladePoint, torsoPoint)
    this.force = Vec2.mul(this.movePower * distance, moveDir)
  }
}
