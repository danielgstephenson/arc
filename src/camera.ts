import { Vec2 } from 'planck'
import { clamp } from './math'

export class Camera {
  position = new Vec2(0, 0)
  maxZoom = 15
  minZoom = -5
  zoom = 0
  scale = 1

  constructor () {
    this.adjustZoom(this.zoom)
  }

  adjustZoom (change: number): void {
    this.zoom = clamp(this.minZoom, this.maxZoom, this.zoom + change)
    this.scale = 0.02 * Math.exp(0.1 * this.zoom)
    console.log('zoom:', this.zoom)
  }
}
