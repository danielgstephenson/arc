import { Circle } from 'planck'
import { Feature } from './feature'
import { Weapon } from '../entities/weapon'

export class Blade extends Feature {
  static radius = 1
  weapon: Weapon
  alive = true

  constructor (weapon: Weapon) {
    super(weapon, {
      shape: new Circle(Blade.radius),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.weapon = weapon
    this.label = 'torso'
  }
}
