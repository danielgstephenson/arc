import { Vec2 } from 'planck'
import { angleToDir, dot, leakyRelu, range, sample, twoPi } from './math'
import { Fighter } from './entities/fighter'
import { Simulation } from './simulation'

export class Model {
  static noise = 0.01
  static discount = 0.03
  weight: number[][][] = []
  bias: number[][] = []
  options: Vec2[] = []
  negX = range(16).map(i => (-1) ** (i + 1))
  negY = range(16).map(i => (-1) ** (i))
  negXY = range(16).map(i => -1)

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
        if (L === 3) return linear
        return leakyRelu(linear)
      })
      layers.push(layer)
    })
    return layers[4][0]
  }

  evaluate (state: number[]): number {
    const s0 = state
    const s1 = range(16).map(i => state[i] * this.negX[i])
    const s2 = range(16).map(i => state[i] * this.negY[i])
    const s3 = range(16).map(i => state[i] * this.negXY[i])
    return this.base(s0) + this.base(s1) + this.base(s2) + this.base(s3)
  }

  base (state: number[]): number {
    const state0 = state
    const state1 = this.flip(state)
    return this.core(state0) - this.core(state1)
  }

  getReward (fighter0: Fighter, fighter1: Fighter): number {
    const fp0 = fighter0.body.getPosition()
    const fp1 = fighter1.body.getPosition()
    const fv0 = fighter0.body.getLinearVelocity()
    const fv1 = fighter1.body.getLinearVelocity()
    const wv0 = fighter0.weapon.body.getLinearVelocity()
    const wv1 = fighter1.weapon.body.getLinearVelocity()
    const dist0 = Vec2.lengthOf(fp0)
    const dist1 = Vec2.lengthOf(fp1)
    const swing0 = Vec2.distance(wv0, fv0)
    const swing1 = Vec2.distance(wv1, fv1)
    const C = 10
    return C * (dist1 / C) ** 2 - C * (dist0 / C) ** 2 + 0.01 * (swing0 - swing1)
  }

  getState (fighter0: Fighter, fighter1: Fighter): number[] {
    const fp0 = fighter0.body.getPosition()
    const fv0 = fighter0.body.getLinearVelocity()
    const wp0 = fighter0.weapon.body.getPosition()
    const wv0 = fighter0.weapon.body.getLinearVelocity()
    const fp1 = fighter1.body.getPosition()
    const fv1 = fighter1.body.getLinearVelocity()
    const wp1 = fighter1.weapon.body.getPosition()
    const wv1 = fighter1.weapon.body.getLinearVelocity()
    return [
      fp0.x, fp0.y, fv0.x, fv0.y, wp0.x, wp0.y, wv0.x, wv0.y,
      fp1.x, fp1.y, fv1.x, fv1.y, wp1.x, wp1.y, wv1.x, wv1.y
    ]
  }

  flip (state: number[]): number[] {
    const indices = [...range(8, 15), ...range(0, 7)]
    return indices.map(i => state[i])
  }

  getAction (state: number[]): Vec2 {
    if (Math.random() < Model.noise) {
      return sample(this.options)
    }
    const dt = Simulation.timeStep
    const values = this.options.map(option => {
      const outcome = [...state]
      outcome[2] += option.x * dt * 0.5
      outcome[3] += option.y * dt * 0.5
      return this.evaluate(outcome)
    })
    const maxValue = Math.max(...values)
    const bestOptions = this.options.filter((_, i) => {
      return values[i] === maxValue
    })
    return sample(bestOptions)
  }
}
