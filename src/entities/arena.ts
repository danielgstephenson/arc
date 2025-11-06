import { Vec2 } from 'planck'
import { Entity } from './entity'
import { Simulation } from '../simulation'
import { Boundary } from '../features/boundary'
import { ArenaSummary } from '../summaries'

export class Arena extends Entity {
  boundary: Boundary

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

  summarize (): ArenaSummary {
    return {
      boundary: this.boundary.vertices
    }
  }
}
