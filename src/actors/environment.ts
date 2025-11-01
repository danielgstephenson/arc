import { Vec2 } from 'planck'
import { Actor } from './actor'
import { Platform } from '../features/platform'
import { Stage } from '../stage'

export class Environment extends Actor {
  platforms: Platform[] = []

  constructor (stage: Stage) {
    super(stage, {
      type: 'static'
    })
    this.label = 'environment'
  }

  addPlatform (width: number, height: number, center: Vec2): void {
    const platform = new Platform(this, width, height, center)
    this.platforms.push(platform)
  }
}
