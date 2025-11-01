import { Vec2 } from 'planck'
import { Environment } from './actors/environment'
import { Fighter } from './actors/fighter'
import { Sim } from './sim'
import { Renderer } from './renderer'

export class Stage {
  sim: Sim
  renderer: Renderer
  environment: Environment
  player: Fighter

  constructor () {
    this.sim = new Sim()
    this.renderer = new Renderer(this.sim)
    this.environment = new Environment(this.sim)
    this.environment.addPlatform(30, 1, Vec2.zero())
    const spawnPoint = new Vec2(0, 20)
    this.player = new Fighter(this.sim, spawnPoint)
  }
}
