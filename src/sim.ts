import { Vec2, World } from 'planck'
import { Actor } from './actors/actor'

export class Sim {
  world: World
  time: number
  actors = new Map<string, Actor>()
  actorCount = 0
  active = true

  constructor () {
    this.world = new World()
    const gravity = new Vec2(0, -50)
    this.world.setGravity(gravity)
    this.time = performance.now()
    setInterval(() => this.step(), 20)
  }

  step (): void {
    const oldTime = this.time
    this.time = performance.now()
    if (!this.active) return
    const dt = (this.time - oldTime) / 1000
    this.actors.forEach(actor => actor.preStep(dt))
    this.world.step(dt)
    this.actors.forEach(actor => actor.postStep(dt))
  }
}
