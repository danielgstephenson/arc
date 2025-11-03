import { Chain, Vec2 } from 'planck'
import { Feature } from './feature'
import { Arena } from '../actors/arena'

export class Boundary extends Feature {
  arena: Arena
  vertices: Vec2[]

  constructor (arena: Arena, vertices: Vec2[]) {
    super(arena, {
      shape: new Chain(vertices, true),
      density: 1,
      friction: 0,
      restitution: 0
    })
    this.arena = arena
    this.vertices = vertices
    this.label = 'boundary'
  }
}
