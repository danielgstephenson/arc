import { Vec2 } from 'planck'
import { angleToDir, dot, leakyRelu, range, sample, twoPi } from './math'
import parameters from '../model/parameters.json'
import { actionSpace } from './actionSpace'

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
    this.bias.push(parameters['h1.bias'])
    this.bias.push(parameters['h2.bias'])
    this.bias.push(parameters['h3.bias'])
    this.bias.push(parameters['h4.bias'])
    this.bias.push(parameters['out.bias'])
    this.weight.push(parameters['h1.weight'])
    this.weight.push(parameters['h2.weight'])
    this.weight.push(parameters['h3.weight'])
    this.weight.push(parameters['h4.weight'])
    this.weight.push(parameters['out.weight'])
    range(8).forEach(i => {
      const angle = twoPi * i / 8
      const dir = angleToDir(angle)
      this.options.push(dir)
    })
  }

  core (input: number[]): number {
    if (this.bias.length === 0) return 0
    const layers = [input]
    range(5).forEach(L => {
      const input = layers[L]
      const bias = this.bias[L]
      const weight = this.weight[L]
      const n = bias.length
      const layer = range(n).map(i => {
        const linear = bias[i] + dot(weight[i], input)
        if (L === 4) return linear
        return leakyRelu(linear)
      })
      layers.push(layer)
    })
    return layers[5][0]
  }

  getActionValue (s0: number[], s1: number[], a0: number, a1: number): number {
    const v0 = actionSpace[a0]
    const v1 = actionSpace[a1]
    const input = [...s0, ...s1, v0.x, v0.y, v1.x, v1.y]
    return this.core(input)
  }

  getAction (s0: number[], s1: number[], a1: number): number {
    const actionCount = actionSpace.length
    const actionValues = range(actionCount).map(a0 => {
      return this.getActionValue(s0, s1, a0, a1)
    })
    const maxActionValue = Math.max(...actionValues)
    const optimalActions = range(actionCount).filter(a0 => {
      return actionValues[a0] === maxActionValue
    })
    return sample(optimalActions)
  }
}
