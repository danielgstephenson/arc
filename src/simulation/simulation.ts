import { Vec2, World } from 'planck'
import { Arena } from '../entities/arena'
import { Fighter } from '../entities/fighter'
import { Entity } from '../entities/entity'
import { Collider } from '../colllider'
import { SimulationSummary } from '../summaries'
import { DefaultEventsMap, Socket } from 'socket.io'

export class Simulation {
  timeScale = 1
  timeStep = 0.04
  world = new World()
  collider: Collider
  arena: Arena
  summary: SimulationSummary
  entities = new Map<number, Entity>()
  fighters = new Map<number, Fighter>()
  entityCount = 0
  active = true
  player?: Fighter
  ann?: Socket<DefaultEventsMap, DefaultEventsMap, DefaultEventsMap, any>

  constructor () {
    this.collider = new Collider(this)
    this.arena = new Arena(this)
    this.reset()
    this.summary = this.summarize()
  }

  start (): void {
    setInterval(() => this.step(), 1000 * this.timeStep / this.timeScale)
  }

  step (): void {
    if (!this.active) return
    const dt = this.timeStep
    this.preStep(dt)
    this.entities.forEach(entity => entity.preStep(dt))
    this.entities.forEach(entity => entity.body.applyForce(entity.force, Vec2.zero()))
    this.world.step(dt)
    this.entities.forEach(entity => entity.postStep(dt))
    this.summary = this.summarize()
    this.postStep(dt)
  }

  preStep (dt: number): void {}

  postStep (dt: number): void {}

  addFighter (position = Vec2.zero()): Fighter {
    return new Fighter(this, position)
  }

  respawn (fighter: Fighter): void {}

  reset (): void {}

  summarize (): SimulationSummary {
    const fighters = [...this.fighters.values()]
    return {
      arena: this.arena.summarize(),
      fighters: fighters.map(fighter => fighter.summarize()),
      player: this.player?.body.getPosition()
    }
  }
}
