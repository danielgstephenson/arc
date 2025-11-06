import { Vec2, World } from 'planck'
import { Arena } from './entities/arena'
import { Fighter } from './entities/fighter'
import { Entity } from './entities/entity'
import { Collider } from './colllider'
import { SimulationSummary } from './summaries'
import { randomDir } from './math'

export class Simulation {
  world: World
  collider: Collider
  arena: Arena
  time: number
  summary: SimulationSummary
  entities = new Map<number, Entity>()
  fighters = new Map<number, Fighter>()
  player?: Fighter
  playerForce = Vec2.zero()
  entityCount = 0
  active = true
  timeScale = 1
  timeStep = 0.02

  constructor () {
    this.world = new World()
    this.collider = new Collider(this)
    this.time = performance.now()
    this.arena = new Arena(this)
    const fighter0 = new Fighter(this, new Vec2(0, +10))
    this.respawn(fighter0)
    const fighter1 = new Fighter(this, new Vec2(0, -10))
    this.respawn(fighter1)
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.summary = this.summarize()
    setInterval(() => this.step(), 1000 * this.timeStep)
  }

  step (): void {
    const oldTime = this.time
    this.time = performance.now()
    if (!this.active) return
    const dt = this.timeScale * (this.time - oldTime) / 1000
    this.preStep(dt)
    this.entities.forEach(entity => entity.preStep(dt))
    this.entities.forEach(entity => entity.body.applyForce(entity.force, Vec2.zero()))
    this.world.step(dt)
    this.entities.forEach(entity => entity.postStep(dt))
    this.postStep(dt)
    this.summary = this.summarize()
  }

  preStep (dt: number): void {
    this.movePlayer()
  }

  postStep (dt: number): void {
    const entities = [...this.entities.values()]
    const fighters = entities.filter(a => a instanceof Fighter)
    fighters.forEach(fighter => {
      if (fighter.dead) this.respawn(fighter)
    })
  }

  respawn (fighter: Fighter): void {
    console.log('die')
    fighter.spawnPoint = Vec2.mul(12, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }

  movePlayer (): void {
    if (this.player == null) return
    this.player.force = this.playerForce
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
