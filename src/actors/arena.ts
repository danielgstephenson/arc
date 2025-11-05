import { Vec2 } from 'planck'
import { Actor } from './actor'
import { Wall } from '../features/wall'
import { Simulation } from '../simulation'
import { Boundary } from '../features/boundary'

export class Arena extends Actor {
  boundary: Boundary
  walls: Wall[] = []

  constructor (simulation: Simulation) {
    super(simulation, {
      type: 'static'
    })
    this.label = 'arena'
    const size = 20
    this.boundary = new Boundary(this, [
      new Vec2(-size, -size),
      new Vec2(-size, +size),
      new Vec2(+size, +size),
      new Vec2(+size, -size)
    ])
  }

  addWall (width: number, height: number, center: Vec2): void {
    const wall = new Wall(this, width, height, center)
    this.walls.push(wall)
  }
}
