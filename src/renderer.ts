import { Vec2 } from 'planck'
import { Arena } from './actors/arena'
import { Fighter } from './actors/fighter'
import { Camera } from './camera'
import { Stage } from './stage'
import { Checker } from './checker'
import { Boundary } from './features/boundary'
import { Weapon } from './actors/weapon'
import { dirFromTo } from './math'

export class Renderer {
  camera = new Camera()
  checker = new Checker()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  stage: Stage

  backgroundColor = 'hsl(0,0%,0%)'
  wallColor = 'hsl(0,0%,35%)'

  constructor (stage: Stage) {
    this.stage = stage
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    this.followPlayer()
    this.drawBoundary(this.stage.arena.boundary)
    const actors = [...this.stage.actors.values()]
    const weapons = actors.filter(a => a instanceof Weapon)
    weapons.forEach(weapon => this.drawSpring(weapon))
    weapons.forEach(weapon => this.drawBlade(weapon))
    const fighters = actors.filter(a => a instanceof Fighter)
    fighters.forEach(fighter => this.drawTorso(fighter))
    const environments = actors.filter(a => a instanceof Arena)
    environments.forEach(environment => this.drawWalls(environment))
  }

  drawBoundary (boundary: Boundary): void {
    this.context.fillStyle = this.backgroundColor
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.resetContext()
    this.context.imageSmoothingEnabled = false
    this.context.fillStyle = this.checker.pattern
    this.context.beginPath()
    boundary.vertices.forEach((vertex, i) => {
      if (i === 0) this.context.moveTo(vertex.x, vertex.y)
      else this.context.lineTo(vertex.x, vertex.y)
    })
    this.context.fill()
  }

  drawSpring (weapon: Weapon): void {
    this.resetContext()
    this.context.strokeStyle = 'hsla(0, 100%, 100%, 0.1)'
    this.context.lineWidth = 0.08
    const bladePoint = weapon.body.getWorldCenter()
    const torsoPoint = weapon.fighter.body.getWorldCenter()
    const distance = Vec2.distance(bladePoint, torsoPoint)
    const bladeRadius = weapon.blade.radius
    const torsoRadius = weapon.fighter.torso.radius
    if (distance < bladeRadius + torsoRadius) return
    const dir = dirFromTo(bladePoint, torsoPoint)
    const edgePoint = Vec2.combine(1, bladePoint, bladeRadius, dir)
    this.context.beginPath()
    this.context.moveTo(edgePoint.x, edgePoint.y)
    this.context.lineTo(torsoPoint.x, torsoPoint.y)
    this.context.stroke()
  }

  drawBlade (weapon: Weapon): void {
    this.resetContext()
    this.context.fillStyle = weapon.color
    const center = weapon.body.getWorldCenter()
    this.context.beginPath()
    this.context.arc(center.x, center.y, weapon.blade.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawTorso (fighter: Fighter): void {
    this.resetContext()
    this.context.fillStyle = fighter.color
    const center = fighter.body.getWorldCenter()
    this.context.beginPath()
    this.context.arc(center.x, center.y, fighter.torso.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawWalls (environment: Arena): void {
    environment.walls.forEach(wall => {
      this.resetContext()
      this.context.fillStyle = this.wallColor
      const x = wall.center.x - 0.5 * wall.width
      const y = wall.center.y - 0.5 * wall.height
      this.context.fillRect(x, y, wall.width, wall.height)
    })
  }

  followPlayer (): void {
    if (this.stage.player == null) {
      this.camera.position = Vec2.zero()
      return
    }
    this.camera.position = this.stage.player.body.getWorldCenter()
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
