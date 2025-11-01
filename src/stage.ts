import { Vec2, World } from 'planck'
import { Environment } from './actors/environment'
import { Fighter } from './actors/fighter'
import { Renderer } from './renderer'
import { Input } from './input'
import { Actor } from './actors/actor'
import { normalize } from './math'

export class Stage {
  world: World
  renderer: Renderer
  environment: Environment
  input: Input
  player: Fighter
  time: number
  actors = new Map<string, Actor>()
  actorCount = 0
  active = true

  constructor () {
    this.world = new World()
    this.time = performance.now()
    this.environment = new Environment(this)
    this.environment.addPlatform(30, 1, Vec2.zero())
    const spawnPoint = new Vec2(0, 5)
    this.player = new Fighter(this, spawnPoint)
    this.input = new Input(this)
    this.renderer = new Renderer(this)
    setInterval(() => this.step(), 20)
  }

  step (): void {
    const oldTime = this.time
    this.time = performance.now()
    if (!this.active) return
    const dt = (this.time - oldTime) / 1000
    this.movePlayer()
    this.actors.forEach(actor => actor.preStep(dt))
    this.world.step(dt)
    this.actors.forEach(actor => actor.postStep(dt))
  }

  movePlayer (): void {
    let x = 0
    let y = 0
    if (this.input.isKeyDown('KeyW') || this.input.isKeyDown('ArrowUp')) y += 1
    if (this.input.isKeyDown('KeyS') || this.input.isKeyDown('ArrowDown')) y -= 1
    if (this.input.isKeyDown('KeyA') || this.input.isKeyDown('ArrowLeft')) x -= 1
    if (this.input.isKeyDown('KeyD') || this.input.isKeyDown('ArrowRight')) x += 1
    this.player.moveDir = normalize(new Vec2(x, y))
  }
}
