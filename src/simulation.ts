import { Vec2, World } from 'planck'
import { Arena } from './entities/arena'
import { Fighter } from './entities/fighter'
import { Renderer } from './renderer'
import { Input } from './input'
import { Entity } from './entities/entity'
import { normalize, randomDir } from './math'
import { Collider } from './colllider'

export class Simulation {
  world: World
  renderer: Renderer
  collider: Collider
  arena: Arena
  input: Input
  player?: Fighter
  time: number
  entities = new Map<number, Entity>()
  entityCount = 0
  active = true
  timeScale = 1
  timeStep = 0.02

  constructor () {
    this.world = new World()
    this.collider = new Collider(this)
    this.input = new Input(this)
    this.time = performance.now()
    this.arena = new Arena(this)
    const player = new Fighter(this, new Vec2(0, +10))
    this.respawn(player)
    this.player = player
    const enemy = new Fighter(this, new Vec2(0, -10))
    this.respawn(enemy)
    enemy.color = 'hsl(120, 100%, 25%)'
    enemy.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
    this.renderer = new Renderer(this)
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
    fighter.spawnPoint = Vec2.mul(10, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }

  movePlayer (): void {
    if (this.player == null) return
    let x = 0
    let y = 0
    if (this.input.isKeyDown('KeyW') || this.input.isKeyDown('ArrowUp')) y += 1
    if (this.input.isKeyDown('KeyS') || this.input.isKeyDown('ArrowDown')) y -= 1
    if (this.input.isKeyDown('KeyA') || this.input.isKeyDown('ArrowLeft')) x -= 1
    if (this.input.isKeyDown('KeyD') || this.input.isKeyDown('ArrowRight')) x += 1
    const moveDir = normalize(new Vec2(x, y))
    this.player.force = Vec2.mul(this.player.movePower, moveDir)
  }
}
