import { Vec2, World } from 'planck'
import { Arena } from './entities/arena'
import { Fighter } from './entities/fighter'
import { Entity } from './entities/entity'
import { Collider } from './colllider'
import { SimulationSummary } from './summaries'
import { randomDir, range } from './math'
import { Model } from './model'
import { DefaultEventsMap, Socket } from 'socket.io'
import { Blade } from './features/blade'

export class Simulation {
  static timeScale = 2
  static timeStep = 0.04
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
  oldState = range(16).map(_ => 0)
  oldReward = 0
  newState = range(16).map(_ => 0)
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
    setInterval(() => this.step(), 1000 * Simulation.timeStep / Simulation.timeScale)
    // this.player = fighter1
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
    this.oldReward = this.model.getReward(fighters[0], fighters[1])
    const value = this.model.evaluate(this.oldState)
    console.log('value', value)
  }

  act (): void {
    const fighters = [...this.fighters.values()]
    if (fighters[0] !== this.player) {
      const state0 = this.model.getState(fighters[0], fighters[1])
      fighters[0].action = this.model.getAction(state0)
    }
    if (fighters[1] !== this.player) {
      const state1 = this.model.getState(fighters[1], fighters[0])
      fighters[1].action = this.model.getAction(state1)
    }
  }

  postStep (dt: number): void {
    const fighters = [...this.fighters.values()]
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
    this.newState = this.model.getState(fighters[0], fighters[1])
    this.sendData(dt)
    if (Math.random() < 0.04 * Simulation.timeStep) {
      this.restart()
    }
  }

  sendData (dt: number): void {
    if (this.ann == null) return
    const state = this.oldState
    const nextValue = this.model.evaluate(this.newState)
    const A = dt * Model.discount / (1 + Model.discount)
    const value = (1 - A) * this.oldReward + A * nextValue
    this.ann.emit('observe', { state, value })
  }

  respawn (fighter: Fighter): void {
    fighter.spawnPoint = Vec2.mul(Arena.size - Blade.radius, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }

  restart (): void {
    this.fighters.forEach(fighter => {
      this.respawn(fighter)
    })
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
