import { Box, Vec2 } from 'planck'
import { Feature } from './feature'
import { Arena } from '../entities/arena'

export class Wall extends Feature {
  arena: Arena
  center: Vec2
  width: number
  height: number

  constructor (environment: Arena, width: number, height: number, center: Vec2) {
    const box = new Box(0.5 * width, 0.5 * height, center)
    super(environment, {
      shape: box,
      density: 1,
      friction: 1,
      restitution: 0
    })
    this.width = width
    this.height = height
    this.center = center
    this.arena = environment
    this.label = 'platform'
  }
}
