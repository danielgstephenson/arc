
import { range, whichMax } from './math'
import * as ort from 'onnxruntime-node'
import { Imagination } from './simulation/imagination'

export class Model {
  imagination = new Imagination()
  session?: ort.InferenceSession
  busy: boolean = false
  action: number = 0

  constructor () {
    void this.startSession()
  }

  async startSession (): Promise<void> {
    this.session = await ort.InferenceSession.create('./model/onnx/model5.onnx')
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
    console.log('result', Object.keys(result))
    console.log('value', result.linear_15.data[0])
  }

  async update (state: number[]): Promise<void> {
    if (this.busy) return
    this.busy = true
    const outcomes = this.imagination.getOutcomes(state)
    const values: number[] = []
    for (const outcome of outcomes) {
      values.push(await this.getValue(outcome))
      // values.push(this.imagination.getValue1(outcome))
    }
    const outcomeMatrix: number[][][] = range(9).map(i => [])
    const valueMatrix: number[][] = range(9).map(i => [])
    range(9).forEach(i => {
      range(9).forEach(j => {
        const v = values.shift()
        if (v != null) valueMatrix[i][j] = v
        const x = outcomes.shift()
        if (x != null) outcomeMatrix[i][j] = x
      })
    })
    const mins = valueMatrix.map(row => Math.min(...row))
    this.action = whichMax(mins)
    this.busy = false
  }

  async getValue (state: number[]): Promise<number> {
    const data = Float32Array.from(state)
    const tensor = new ort.Tensor('float32', data, [1, 16])
    const feeds = { state: tensor }
    if (this.session == null) return 0
    const result = await this.session.run(feeds)
    const value = Number(result.linear_15.data[0])
    return value
  }
}
