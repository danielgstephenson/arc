import { dot, range, relu } from './math'

export class Model {
  weight: number[][][] = []
  bias: number[][] = []

  core (state: number[]): number {
    const layers = [state]
    range(4).forEach(L => {
      const input = layers[L]
      const bias = this.bias[L]
      const weight = this.weight[L]
      const n = bias.length
      const layer = range(n).map(i => {
        return relu(bias[i] + dot(weight[i], input))
      })
      layers.push(layer)
    })
    return layers[4][0]
  }

  evaluate (state: number[]): number {
    const state0 = state
    const state1 = [4, 5, 6, 7, 0, 1, 2, 3].map(i => state[i])
    return this.core(state0) - this.core(state1)
  }
}
