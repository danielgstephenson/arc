import { Vec2 } from 'planck'
import { angleToDir, range, twoPi } from './math'

export const actionSpace: Vec2[] = []

actionSpace.push(Vec2.zero())
range(8).forEach(i => {
  const angle = twoPi * i / 8
  const dir = angleToDir(angle)
  actionSpace.push(dir)
})
