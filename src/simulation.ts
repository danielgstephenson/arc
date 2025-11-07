import { Vec2, World } from 'planck'
import { Arena } from './entities/arena'
import { Fighter } from './entities/fighter'
import { Entity } from './entities/entity'
import { Collider } from './colllider'
import { SimulationSummary } from './summaries'
import { randomDir } from './math'
import { Model } from './model'
import { DefaultEventsMap, Socket } from 'socket.io'

export class Simulation {
  static timeScale = 1
  static timeStep = 0.02
  world = new World()
  model = new Model()
  collider: Collider
  arena: Arena
  time: number
  summary: SimulationSummary
  entities = new Map<number, Entity>()
  fighters = new Map<number, Fighter>()
  entityCount = 0
  active = true
  oldState = [0, 0, 0, 0, 0, 0, 0, 0]
  oldPayoff = 0
  newState = [0, 0, 0, 0, 0, 0, 0, 0]
  player?: Fighter
  ann?: Socket<DefaultEventsMap, DefaultEventsMap, DefaultEventsMap, any>

  constructor () {
    this.collider = new Collider(this)
    this.arena = new Arena(this)
    this.time = performance.now()
    const fighter0 = new Fighter(this, new Vec2(0, +10))
    this.respawn(fighter0)
    const fighter1 = new Fighter(this, new Vec2(0, -10))
    this.respawn(fighter1)
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.summary = this.summarize()
    setInterval(() => this.step(), 1000 * Simulation.timeStep)
  }

  step (): void {
    const oldTime = this.time
    this.time = performance.now()
    if (!this.active) return
    const dt = Simulation.timeScale * (this.time - oldTime) / 1000
    this.preStep(dt)
    this.entities.forEach(entity => entity.preStep(dt))
    this.entities.forEach(entity => entity.body.applyForce(entity.force, Vec2.zero()))
    this.world.step(dt)
    this.entities.forEach(entity => entity.postStep(dt))
    this.postStep(dt)
    this.summary = this.summarize()
  }

  preStep (dt: number): void {
    this.act()
    const fighters = [...this.fighters.values()]
    this.oldState = this.model.getState(fighters[0], fighters[1])
    this.oldPayoff = this.model.objective(fighters[0], fighters[1])
    const p0 = fighters[0].body.getPosition()
    const p1 = fighters[1].body.getPosition()
    const distance0 = Vec2.lengthOf(p0)
    const distance1 = Vec2.lengthOf(p1)
    console.log('payoff', distance1.toFixed(4), distance0.toFixed(4), this.oldPayoff.toFixed(4))
  }

  act (): void {
    const fighters = [...this.fighters.values()]
    const state0 = this.model.getState(fighters[0], fighters[1])
    const state1 = this.model.getState(fighters[1], fighters[0])
    fighters[0].action = this.model.getAction(state0)
    fighters[1].action = this.model.getAction(state1)
  }

  postStep (dt: number): void {
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
    this.newState = this.model.getState(fighters[0], fighters[1])
    // this.sendData()
  }

  sendData (): void {
    if (this.ann == null) return
    const dt = Simulation.timeStep
    const discount = dt * Model.discount
    const state = this.oldState
    const nextValue = this.model.evaluate(this.newState)
    const value = dt * this.oldPayoff + (1 - discount) * nextValue
    this.ann.emit('observe', { state, value })
  }

  respawn (fighter: Fighter): void {
    fighter.spawnPoint = Vec2.mul(12, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }

  summarize (): SimulationSummary {
    const fighters = [...this.fighters.values()]
    return {
      arena: this.arena.summarize(),
      fighters: fighters.map(fighter => fighter.summarize()),
      player: this.player?.body.getPosition()
    }
  }
}
