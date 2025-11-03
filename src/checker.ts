export class Checker {
  pattern: CanvasPattern

  constructor () {
    const canvas: HTMLCanvasElement = document.createElement('canvas')
    const context = canvas.getContext('2d')
    if (context == null) throw new Error('context == null')
    canvas.style.imageRendering = 'pixelated'
    const size = 10 // must be even
    canvas.width = size
    canvas.height = size
    const color1 = 'hsl(0, 0%, 6%)'
    const color2 = 'hsl(0, 0%, 8%)'
    context.fillStyle = color1
    context.fillRect(0, 0, size, size)
    context.fillStyle = color2
    context.fillRect(0, 0, 0.5 * size, 0.5 * size)
    context.fillRect(0.5 * size, 0.5 * size, 0.5 * size, 0.5 * size)
    const pattern = context.createPattern(canvas, 'repeat')
    if (pattern == null) throw new Error('pattern == null')
    this.pattern = pattern
  }
}
