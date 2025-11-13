import { Vec2 } from 'planck'
import { Entity } from './entity'
import { Simulation } from '../simulation/simulation'
import { Boundary } from '../features/boundary'
import { ArenaSummary } from '../summaries'

export class Arena extends Entity {
  static size = 90000
  boundary: Boundary

  constructor (simulation: Simulation) {
    super(simulation, {
      type: 'static'
    })
    this.label = 'arena'
    this.boundary = new Boundary(this, [
      new Vec2(-Arena.size, -Arena.size),
      new Vec2(-Arena.size, +Arena.size),
      new Vec2(+Arena.size, +Arena.size),
      new Vec2(+Arena.size, -Arena.size)
    ])
  }

  summarize (): ArenaSummary {
    return {
      boundary: this.boundary.vertices
    }
  }
}
