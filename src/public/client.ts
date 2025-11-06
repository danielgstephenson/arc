import { Vec2 } from 'planck'
import { Input } from '../input'
import { Renderer } from '../renderer'
import io from 'socket.io-client'
import { SimulationSummary } from '../summaries'

const renderer = new Renderer()
const input = new Input()

function update (): void {
  renderer.camera.updateScale(input.zoom)
  let x = 0
  let y = 0
  if (input.isKeyDown('KeyW') || input.isKeyDown('ArrowUp')) y += 1
  if (input.isKeyDown('KeyS') || input.isKeyDown('ArrowDown')) y -= 1
  if (input.isKeyDown('KeyA') || input.isKeyDown('ArrowLeft')) x -= 1
  if (input.isKeyDown('KeyD') || input.isKeyDown('ArrowRight')) x += 1
  const vector = new Vec2(x, y)
  socket.emit('input', vector)
}

setInterval(update, 20)

const socket = io()
socket.on('connected', () => {
  console.log('connected')
})
socket.on('summary', (summary: SimulationSummary) => {
  renderer.summary = summary
})
