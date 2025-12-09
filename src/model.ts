
import { range } from './math'
import * as ort from 'onnxruntime-node'
import { Imagination } from './simulation/imagination'

export class Model {
  imagination = new Imagination()
  session?: ort.InferenceSession

  constructor () {
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
    console.log('value', result.linear_4.data[0])
  }

  // async getValues (fighter: Fighter, otherFighter: Fighter): Promise<number[]> {
  //   if (this.session == null) return range(9).map(i => 0)
  //   const outcomes = this.imagination.getOutcomes(fighter, otherFighter)
  //   const data = Float32Array.from(outcomes.flat())
  //   const tensor = new ort.Tensor('float32', data, [outcomes.length, 16])
  //   const feeds = { state: tensor }
  //   const result = await this.session.run(feeds)
  //   // return the result value
  // }
}
