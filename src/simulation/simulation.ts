import { Vec2, World } from 'planck'
import { Arena } from '../entities/arena'
import { Fighter } from '../entities/fighter'
import { Entity } from '../entities/entity'
import { Collider } from '../colllider'
import { SimulationSummary } from '../summaries'
import { Model } from '../model'
import { DefaultEventsMap, Socket } from 'socket.io'

export class Simulation {
  static timeScale = 1
  static timeStep = 0.04 // 0.20
  world = new World()
  model = new Model()
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
    console.log('Simulation')
  }

  start (): void {
    setInterval(() => this.step(), 1000 * Simulation.timeStep / Simulation.timeScale)
  }

  step (): void {
    if (!this.active) return
    const dt = Simulation.timeStep
    this.preStep(dt)
    this.entities.forEach(entity => entity.preStep(dt))
    this.entities.forEach(entity => entity.body.applyForce(entity.force, Vec2.zero()))
    this.world.step(dt)
    this.entities.forEach(entity => entity.postStep(dt))
    this.summary = this.summarize()
    this.postStep(dt)
  }

  getState (fighter: Fighter): number[] {
    const fp = fighter.body.getPosition()
    const fv = fighter.body.getLinearVelocity()
    const wp = fighter.weapon.body.getPosition()
    const wv = fighter.weapon.body.getLinearVelocity()
    return [fp.x, fp.y, fv.x, fv.y, wp.x, wp.y, wv.x, wv.y]
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
