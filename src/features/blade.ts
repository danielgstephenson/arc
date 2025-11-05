import { Circle } from 'planck'
import { Feature } from './feature'
import { Weapon } from '../entities/weapon'

export class Blade extends Feature {
  weapon: Weapon
  radius: number
  alive = true

  constructor (weapon: Weapon, radius = 1) {
    super(weapon, {
      shape: new Circle(radius),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.radius = radius
    this.weapon = weapon
    this.label = 'torso'
  }
}
