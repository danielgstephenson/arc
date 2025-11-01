import { Environment } from './actors/environment'
import { Fighter } from './actors/fighter'
import { Camera } from './camera'
import { Torso } from './features/torso'
import { Sim } from './sim'

export class Renderer {
  camera = new Camera()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  sim: Sim

  backgroundColor = 'hsl(0,0%,10%)'
  torsoColor = 'hsl(220,100%,40%)'
  platformColor = 'hsl(0,0%,35%)'

  constructor (sim: Sim) {
    this.sim = sim
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    const actors = [...this.sim.actors.values()]
    const fighters = actors.filter(a => a instanceof Fighter)
    console.log('fighters.length', fighters.length)
    fighters.forEach(fighter => this.drawTorso(fighter))
    const environments = actors.filter(a => a instanceof Environment)
    environments.forEach(environment => this.drawPlatforms(environment))
  }

  drawTorso (fighter: Fighter): void {
    this.resetContext()
    this.context.fillStyle = this.torsoColor
    const center = fighter.body.getWorldCenter()
    const x = center.x - 0.5 * Torso.width
    const y = center.y - 0.5 * Torso.height
    this.context.fillRect(x, y, Torso.width, Torso.height)
  }

  drawPlatforms (environment: Environment): void {
    environment.platforms.forEach(platform => {
      this.resetContext()
      this.context.fillStyle = this.platformColor
      const x = platform.center.x - 0.5 * platform.width
      const y = platform.center.y - 0.5 * platform.height
      this.context.fillRect(x, y, platform.width, platform.height)
    })
  }

  setupCanvas (): void {
    this.canvas.width = window.innerWidth
    this.canvas.height = window.innerHeight
  }

  resetContext (): void {
    this.context.resetTransform()
    this.context.translate(0.5 * this.canvas.width, 0.5 * this.canvas.height)
    const vmin = Math.min(this.canvas.width, this.canvas.height)
    this.context.scale(vmin, -vmin)
    this.context.scale(this.camera.scale, this.camera.scale)
    this.context.translate(-this.camera.position.x, -this.camera.position.y)
    this.context.globalAlpha = 1
  }
}
