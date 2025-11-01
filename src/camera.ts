import { Vec2 } from 'planck'
import { clamp } from './math'

export class Camera {
  position = new Vec2(0, 0)
  maxZoom = 15
  minZoom = -5
  zoom = 0
  scale = 1

  constructor () {
    this.adjustZoom(0)
  }

  adjustZoom (change: number): void {
    this.zoom = clamp(this.minZoom, this.maxZoom, this.zoom + change)
    this.scale = 0.05 * Math.exp(0.1 * this.zoom)
    // console.log('zoom:', this.zoom)
  }
}
