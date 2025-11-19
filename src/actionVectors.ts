import { Vec2 } from 'planck'
import { angleToDir, range, twoPi } from './math'

export const actionVectors: Vec2[] = []

actionVectors.push(Vec2.zero())
range(8).forEach(i => {
  const angle = twoPi * i / 8
  const dir = angleToDir(angle)
  actionVectors.push(dir)
})
