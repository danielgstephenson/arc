import { Box } from 'planck'
import { Feature } from './feature'
import { Fighter } from '../actors/fighter'

export class Torso extends Feature {
  static width = 4
  static height = 10
  fighter: Fighter
  alive = true

  constructor (fighter: Fighter) {
    super(fighter, {
      shape: new Box(0.5 * Torso.width, 0.5 * Torso.height),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.fighter = fighter
    this.label = 'torso'
  }
}
