import { Vec2 } from 'planck'
import { Sim } from '../sim'
import { Actor } from './actor'
import { Torso } from '../features/torso'

export class Fighter extends Actor {
  torso: Torso
  movePower = 5
  swingPower = 5

  constructor (sim: Sim, position: Vec2) {
    super(sim, {
      type: 'dynamic',
      bullet: true,
      linearDamping: 0,
      fixedRotation: true
    })
    this.label = 'fighter'
    this.body.setPosition(position)
    this.torso = new Torso(this)
  }

  preStep (dt: number): void {
    super.preStep(dt)
  }
}
