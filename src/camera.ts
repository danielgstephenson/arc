import { Vec2 } from 'planck'

export class Camera {
  position = new Vec2(0, 0)
  scale = 1

  updateScale (zoom: number): void {
    this.scale = 0.02 * Math.exp(0.1 * zoom)
  }
}
