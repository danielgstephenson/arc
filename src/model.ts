import { Vec2 } from 'planck'
import { angleToDir, dot, range, relu, sample, twoPi } from './math'
import { Fighter } from './entities/fighter'

export class Model {
  static noise = 0.1
  static discount = 0.1
  weight: number[][][] = []
  bias: number[][] = []
  options: Vec2[] = []

  constructor () {
    this.options.push(Vec2.zero())
    range(8).forEach(i => {
      const angle = twoPi * i / 8
      const dir = angleToDir(angle)
      this.options.push(dir)
    })
  }

  core (state: number[]): number {
    if (this.bias.length === 0) return 0
    const layers = [state]
    range(4).forEach(L => {
      const input = layers[L]
      const bias = this.bias[L]
      const weight = this.weight[L]
      const n = bias.length
      const layer = range(n).map(i => {
        const linear = bias[i] + dot(weight[i], input)
        if (L === 4) return (linear)
        return relu(linear)
      })
      layers.push(layer)
    })
    return layers[4][0]
  }

  objective (fighter0: Fighter, fighter1: Fighter): number {
    const p0 = fighter0.body.getPosition()
    const p1 = fighter1.body.getPosition()
    return Vec2.lengthOf(p1) - Vec2.lengthOf(p0)
  }

  evaluate (state: number[]): number {
    const state0 = state
    const state1 = this.flip(state)
    return this.core(state0) - this.core(state1)
  }

  getState (fighter0: Fighter, fighter1: Fighter): number[] {
    const p0 = fighter0.body.getPosition()
    const v0 = fighter0.body.getLinearVelocity()
    const p1 = fighter1.body.getPosition()
    const v1 = fighter1.body.getLinearVelocity()
    return [p0.x, p0.y, v0.x, v0.y, p1.x, p1.y, v1.x, v1.y]
  }

  flip (state: number[]): number[] {
    return [4, 5, 6, 7, 0, 1, 2, 3].map(i => state[i])
  }

  getAction (state: number[]): Vec2 {
    if (Math.random() < Model.noise) {
      return sample(this.options)
    }
    const values = this.options.map(option => {
      const outcome = [...state]
      outcome[2] += option.x
      outcome[3] += option.y
      return this.evaluate(outcome)
    })
    const maxValue = Math.max(...values)
    const bestOptions = this.options.filter((_, i) => {
      return values[i] === maxValue
    })
    return sample(bestOptions)
  }
}
