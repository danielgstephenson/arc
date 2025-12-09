import { Vec2 } from 'planck'
import { Simulation } from './simulation'
import { randomDir, range } from '../math'
import { Fighter } from '../entities/fighter'
import { Arena } from '../entities/arena'
import { Blade } from '../features/blade'

export class Imagination extends Simulation {
  timeStep: number = 0.2

  constructor () {
    super()
    console.log('Imagination')
    const fighter0 = this.addFighter(new Vec2(0, +10))
    const fighter1 = this.addFighter(new Vec2(0, -10))
    fighter0.color = 'hsl(220,100%,40%)'
    fighter0.weapon.color = 'hsla(220, 50%, 40%, 0.5)'
    fighter1.color = 'hsl(120, 100%, 25%)'
    fighter1.weapon.color = 'hsla(120, 100%, 25%, 0.5)'
  }

  getOutcomes (state: number[]): number[][] {
    const fighters = [...this.fighters.values()]
    const fighterStates = [
      range(0, 7).map(i => state[i]),
      range(8, 15).map(i => state[i])
    ]
    const outcomes: number[][] = []
    range(9).forEach(a0 => {
      range(9).forEach(a1 => {
        const a = [a0, a1]
        range(2).forEach(i => {
          const fs = fighterStates[i]
          fighters[i].body.setPosition(new Vec2(fs[0], fs[1]))
          fighters[i].body.setLinearVelocity(new Vec2(fs[2], fs[3]))
          fighters[i].weapon.body.setPosition(new Vec2(fs[4], fs[5]))
          fighters[i].weapon.body.setLinearVelocity(new Vec2(fs[6], fs[7]))
          fighters[i].action = a[i]
        })
        this.step()
        const s01 = fighters[0].getState()
        const s11 = fighters[1].getState()
        outcomes.push([...s01, ...s11])
      })
    })
    return outcomes
  }

  getReward (state: number[]): number {
    const fighterStates = [
      range(0, 7).map(i => state[i]),
      range(8, 15).map(i => state[i])
    ]
    const positions = fighterStates.map(fs => {
      return new Vec2(fs[0], fs[1])
    })
    return Vec2.lengthOf(positions[1]) - Vec2.lengthOf(positions[0])
  }

  getValue1 (state: number[]): number {
    const currentReward = this.getReward(state)
    const outcomes = this.getOutcomes(state)
    const futureRewards = outcomes.map(x => this.getReward(x))
    const rewardMatrix: number[][] = range(9).map(i => [])
    range(9).forEach(i => {
      range(9).forEach(j => {
        const r = futureRewards.shift()
        if (r != null) rewardMatrix[i][j] = r
      })
    })
    const mins = rewardMatrix.map(rewards => Math.min(...rewards))
    const futureReward = Math.max(...mins)
    const discount = 0.9
    return (1 - discount) * currentReward + discount * futureReward
  }

  postStep (dt: number): void {
    super.postStep(dt)
    const fighters = [...this.fighters.values()]
    fighters.forEach((fighter, i) => {
      if (fighter.dead) {
        this.respawn(fighter)
      }
    })
  }

  respawn (fighter: Fighter): void {
    super.respawn(fighter)
    const spawnRadius = Math.min(40, Arena.size) - Blade.radius
    fighter.spawnPoint = Vec2.mul(spawnRadius, randomDir())
    fighter.body.setPosition(fighter.spawnPoint)
    fighter.weapon.body.setPosition(fighter.spawnPoint)
    fighter.body.setLinearVelocity(Vec2.zero())
    fighter.weapon.body.setLinearVelocity(Vec2.zero())
    fighter.dead = false
  }
}
