import { Circle } from 'planck'
import { Feature } from './feature'
import { Fighter } from '../entities/fighter'

export class Torso extends Feature {
  static radius = 0.5
  fighter: Fighter
  alive = true

  constructor (fighter: Fighter) {
    super(fighter, {
      shape: new Circle(Torso.radius),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.fighter = fighter
    this.label = 'torso'
  }
}
