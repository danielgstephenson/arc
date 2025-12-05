import { Vec2 } from 'planck'
import { Camera } from './camera'
import { Checker } from './checker'
import { dirFromTo } from './math'
import { Torso } from './features/torso'
import { Blade } from './features/blade'
import { ArenaSummary, FighterSummary, SimulationSummary } from './summaries'
import { Arena } from './entities/arena'

export class Renderer {
  camera = new Camera()
  checker = new Checker()
  canvas: HTMLCanvasElement
  context: CanvasRenderingContext2D
  summary: SimulationSummary

  backgroundColor = 'hsl(0,0%,0%)'
  wallColor = 'hsl(0,0%,35%)'

  constructor () {
    this.summary = {
      arena: { boundary: [] },
      fighters: []
    }
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement
    this.context = this.canvas.getContext('2d') as CanvasRenderingContext2D
    this.draw()
  }

  draw (): void {
    window.requestAnimationFrame(() => this.draw())
    this.setupCanvas()
    this.followPlayer()
    this.drawArena(this.summary.arena)
    this.summary.fighters.forEach(fighter => this.drawSpring(fighter))
    this.summary.fighters.forEach(fighter => this.drawBlade(fighter))
    this.summary.fighters.forEach(fighter => this.drawTorso(fighter))
  }

  drawArena (arena: ArenaSummary): void {
    const boundary = arena.boundary
    this.context.fillStyle = this.backgroundColor
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height)
    this.resetContext()
    this.context.imageSmoothingEnabled = false
    this.context.fillStyle = this.checker.pattern
    this.context.beginPath()
    boundary.forEach((vertex, i) => {
      if (i === 0) this.context.moveTo(vertex.x, vertex.y)
      else this.context.lineTo(vertex.x, vertex.y)
    })
    // this.context.fill()
    this.context.strokeStyle = 'hsla(0, 0%, 20%, 1)'
    this.context.lineWidth = 0.2
    this.context.beginPath()
    this.context.arc(0, 0, 5, 0, 2 * Math.PI)
    this.context.arc(0, 0, 10, 0, 2 * Math.PI)
    this.context.arc(0, 0, 20, 0, 2 * Math.PI)
    this.context.moveTo(0, Arena.size)
    this.context.lineTo(0, -Arena.size)
    this.context.moveTo(Arena.size, 0)
    this.context.lineTo(-Arena.size, 0)
    this.context.stroke()
  }

  drawSpring (fighter: FighterSummary): void {
    this.resetContext()
    this.context.strokeStyle = 'hsla(0, 100%, 100%, 0.1)'
    this.context.lineWidth = 0.08
    const distance = Vec2.distance(fighter.weapon, fighter.torso)
    if (distance < Blade.radius + Torso.radius) return
    const dir = dirFromTo(fighter.weapon, fighter.torso)
    const edgePoint = Vec2.combine(1, fighter.weapon, Blade.radius, dir)
    this.context.beginPath()
    this.context.moveTo(edgePoint.x, edgePoint.y)
    this.context.lineTo(fighter.torso.x, fighter.torso.y)
    this.context.stroke()
  }

  drawBlade (fighter: FighterSummary): void {
    this.resetContext()
    const L = fighter.weaponHistory.length
    fighter.weaponHistory.forEach((position, i) => {
      const a = 0.02 * (L - i) / L
      this.context.fillStyle = `hsla(0, 0%, 50%, ${a})`
      this.context.beginPath()
      this.context.arc(position.x, position.y, Blade.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.fillStyle = fighter.weaponColor
    this.context.beginPath()
    this.context.arc(fighter.weapon.x, fighter.weapon.y, Blade.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  drawTorso (fighter: FighterSummary): void {
    this.resetContext()
    this.context.fillStyle = fighter.torsoColor
    const L = fighter.history.length
    fighter.history.forEach((position, i) => {
      const a = 0.02 * (L - i) / L
      this.context.fillStyle = `hsla(0, 0%, 50%, ${a})`
      this.context.beginPath()
      this.context.arc(position.x, position.y, Torso.radius, 0, 2 * Math.PI)
      this.context.fill()
    })
    this.context.fillStyle = fighter.torsoColor
    this.context.beginPath()
    this.context.arc(fighter.torso.x, fighter.torso.y, Torso.radius, 0, 2 * Math.PI)
    this.context.fill()
  }

  followPlayer (): void {
    if (this.summary.player == null) {
      this.camera.position = Vec2.zero()
      return
    }
    this.camera.position = this.summary.player
  }

  setupCanvas (): void {
    this.canvas.width = window.innerWidth / 5
    this.canvas.height = window.innerHeight / 5
    // this.context.imageSmoothingEnabled = false
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
