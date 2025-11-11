import { Vec2 } from 'planck'
import { angleToDir, average, dirFromTo, dot, leakyRelu, range, sample, twoPi } from './math'
import { Fighter } from './entities/fighter'
import { Simulation } from './simulation'

export class Model {
  static noise = 0.1
  static discount = 0.2
  weight: number[][][] = []
  bias: number[][] = []
  options: Vec2[] = []
  flipFightersIndices = [...range(8, 15), ...range(0, 7)]
  flipXY = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
  negX = range(16).map(i => (-1) ** (i + 1))
  negY = range(16).map(i => (-1) ** (i))

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
    const states: number[][] = []
    states[0] = state
    states[1] = range(16).map(i => state[i] * this.negX[i])
    states[2] = range(16).map(i => state[i] * this.negY[i])
    states[3] = range(16).map(i => state[i] * -1)
    states[4] = this.flipXY.map(i => state[i])
    states[5] = this.flipXY.map(i => state[i] * this.negX[i])
    states[6] = this.flipXY.map(i => state[i] * this.negY[i])
    states[7] = this.flipXY.map(i => state[i] * -1)
    const baseValues = states.map(s => this.base(s))
    return average(baseValues)
  }

  base (state: number[]): number {
    const state0 = state
    const state1 = this.flipFighters(state)
    return this.core(state0) - this.core(state1)
  }

  getScore (state: number[]): number {
    const position = new Vec2(state[0], state[1])
    const distance = Vec2.distance(position, Vec2.zero())
    const velocity = new Vec2(state[2], state[3])
    const speed = Vec2.lengthOf(velocity)
    const B = 100 / (1 + Math.exp(0.6 * (distance - 8)))
    const dirToCenter = dirFromTo(position, Vec2.zero())
    const targetVelocity = Vec2.mul(distance - 5, dirToCenter)
    const squaredVelocityError = Vec2.distanceSquared(velocity, targetVelocity)
    return B * speed - squaredVelocityError
  }

  getReward (state: number[]): number {
    const otherState = this.flipFighters(state)
    return this.getScore(state) - this.getScore(otherState)
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

  flipFighters (state: number[]): number[] {
    return this.flipFightersIndices.map(i => state[i])
  }

  getAction (state: number[]): Vec2 {
    if (Math.random() < Model.noise) {
      return sample(this.options)
    }
    const position = new Vec2(state[0], state[1])
    const distance = Vec2.lengthOf(position)
    const W = 1 / (1 + Math.exp(0.3 * (50 - distance)))
    const values = this.options.map(option => {
      const outcome = [...state]
      outcome[2] += option.x * Simulation.timeStep
      outcome[3] += option.y * Simulation.timeStep
      return W * this.getReward(outcome) + (1 - W) * this.evaluate(outcome)
    })
    const maxValue = Math.max(...values)
    const bestOptions = this.options.filter((_, i) => {
      return values[i] === maxValue
    })
    return sample(bestOptions)
  }
}
