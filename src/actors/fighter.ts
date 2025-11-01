import { Vec2 } from 'planck'
import { Actor } from './actor'
import { Torso } from '../features/torso'
import { Stage } from '../stage'
import { normalize } from '../math'

export class Fighter extends Actor {
  torso: Torso
  movePower = 5
  moveDir = new Vec2(0, 0)

  constructor (stage: Stage, position: Vec2) {
    super(stage, {
      type: 'dynamic',
      bullet: true,
      linearDamping: 0.7,
      fixedRotation: true
    })
    this.label = 'fighter'
    this.body.setPosition(position)
    this.torso = new Torso(this)
  }

  preStep (dt: number): void {
    super.preStep(dt)
    this.move()
  }

  move (): void {
    this.moveDir = normalize(this.moveDir)
    const force = Vec2.mul(this.moveDir, this.movePower)
    this.body.applyForceToCenter(force)
  }
}
