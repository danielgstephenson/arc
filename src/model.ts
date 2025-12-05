
import { dot, leakyRelu, range, sample } from './math'
import parameters from '../model/parameters.json'
import { actionVectors } from './actionVectors'
import * as ort from 'onnxruntime-node'

export class Model {
  weight: number[][][] = []
  bias: number[][] = []
  flipFightersIndices = [...range(8, 15), ...range(0, 7)]
  flipXY = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
  negX = range(16).map(i => (-1) ** (i + 1))
  negY = range(16).map(i => (-1) ** (i))
  session?: ort.InferenceSession

  constructor () {
    this.loadParameters()
    void this.startSession()
  }

  async startSession (): Promise<void> {
    this.session = await ort.InferenceSession.create('./model/model.onnx')
    console.time('inference')
    let data = Float32Array.from(range(16))
    let tensor = new ort.Tensor('float32', data, [1, 16])
    let feeds = { state: tensor }
    let result = await this.session.run(feeds)
    console.timeEnd('inference')
    console.time('inference')
    data = Float32Array.from(range(16))
    tensor = new ort.Tensor('float32', data, [1, 16])
    feeds = { state: tensor }
    result = await this.session.run(feeds)
    console.timeEnd('inference')
    console.time('inference')
    data = Float32Array.from(range(16))
    tensor = new ort.Tensor('float32', data, [1, 16])
    feeds = { state: tensor }
    result = await this.session.run(feeds)
    console.timeEnd('inference')
    console.log('result', result)
    console.log('linear_4', result.linear_4)
    console.log('data', result.linear_4.data)
    console.log('value', result.linear_4.data[0])
  }

  loadParameters (): void {
    this.bias.push(parameters['h0.bias'])
    this.bias.push(parameters['h1.bias'])
    this.bias.push(parameters['h2.bias'])
    this.bias.push(parameters['h3.bias'])
    this.bias.push(parameters['h4.bias'])
    this.weight.push(parameters['h0.weight'])
    this.weight.push(parameters['h1.weight'])
    this.weight.push(parameters['h2.weight'])
    this.weight.push(parameters['h3.weight'])
    this.weight.push(parameters['h4.weight'])
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
    const v0 = actionVectors[a0]
    const v1 = actionVectors[a1]
    const input = [...s0, ...s1, v0.x, v0.y, v1.x, v1.y]
    return this.core(input)
  }

  getAction (s0: number[], s1: number[], a1: number): number {
    const actionCount = actionVectors.length
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
