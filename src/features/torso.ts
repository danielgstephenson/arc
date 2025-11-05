import { Circle } from 'planck'
import { Feature } from './feature'
import { Fighter } from '../entities/fighter'

export class Torso extends Feature {
  fighter: Fighter
  radius: number
  alive = true

  constructor (fighter: Fighter, radius = 0.5) {
    super(fighter, {
      shape: new Circle(radius),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.radius = radius
    this.fighter = fighter
    this.label = 'torso'
  }
}
