import { Vec2 } from 'planck'
import { clamp } from './math'

export class Input {
  keyboard = new Map<string, boolean>()
  mousePosition = new Vec2(0, 0)
  mouseButtons = new Map<number, boolean>()
  maxZoom = 15
  minZoom = -100
  zoom = 0

  constructor () {
    window.onkeydown = (event: KeyboardEvent) => this.onkeydown(event)
    window.onkeyup = (event: KeyboardEvent) => this.onkeyup(event)
    window.onwheel = (event: WheelEvent) => this.onwheel(event)
    window.onmousemove = (event: MouseEvent) => this.onmousemove(event)
    window.onmousedown = (event: MouseEvent) => this.onmousedown(event)
    window.onmouseup = (event: MouseEvent) => this.onmouseup(event)
    window.ontouchmove = (event: TouchEvent) => this.ontouchmove(event)
    window.ontouchstart = (event: TouchEvent) => this.ontouchstart(event)
    window.ontouchend = (event: TouchEvent) => this.ontouchend(event)
    window.oncontextmenu = () => {}
  }

  onkeydown (event: KeyboardEvent): void {
    this.keyboard.set(event.code, true)
  }

  onkeyup (event: KeyboardEvent): void {
    this.keyboard.set(event.code, false)
  }

  isKeyDown (key: string): boolean {
    return this.keyboard.get(key) ?? false
  }

  onwheel (event: WheelEvent): void {
    const change = -0.002 * event.deltaY
    this.zoom = clamp(this.minZoom, this.maxZoom, this.zoom + change)
  }

  onmousemove (event: MouseEvent): void {
    this.mousePosition.x = event.clientX - 0.5 * window.innerWidth
    this.mousePosition.y = 0.5 * window.innerHeight - event.clientY
  }

  onmousedown (event: MouseEvent): void {
    this.mouseButtons.set(event.button, true)
    this.mousePosition.x = event.clientX - 0.5 * window.innerWidth
    this.mousePosition.y = 0.5 * window.innerHeight - event.clientY
  }

  onmouseup (event: MouseEvent): void {
    this.mouseButtons.set(event.button, false)
  }

  ontouchmove (event: TouchEvent): void {
    this.mousePosition.x = event.touches[0].clientX - 0.5 * window.innerWidth
    this.mousePosition.y = 0.5 * window.innerHeight - event.touches[0].clientY
  }

  ontouchstart (event: TouchEvent): void {
    this.mouseButtons.set(0, true)
    this.mousePosition.x = event.touches[0].clientX - 0.5 * window.innerWidth
    this.mousePosition.y = 0.5 * window.innerHeight - event.touches[0].clientY
  }

  ontouchend (event: TouchEvent): void {
    this.mouseButtons.set(0, false)
  }

  isMouseButtonDown (button: number): boolean {
    return this.mouseButtons.get(button) ?? false
  }
}
