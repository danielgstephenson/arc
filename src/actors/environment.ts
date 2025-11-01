import { Vec2 } from 'planck'
import { Sim } from '../sim'
import { Actor } from './actor'
import { Platform } from '../features/platform'

export class Environment extends Actor {
  platforms: Platform[] = []

  constructor (sim: Sim) {
    super(sim, {
      type: 'static'
    })
    this.label = 'environment'
  }

  addPlatform (width: number, height: number, center: Vec2): void {
    const platform = new Platform(this, width, height, center)
    this.platforms.push(platform)
  }
}
