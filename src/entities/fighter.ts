import { Vec2 } from 'planck'
import { Entity } from './entity'
import { Torso } from '../features/torso'
import { Simulation } from '../simulation/simulation'
import { Weapon } from './weapon'
import { FighterSummary } from '../summaries'
import { actionVectors } from '../actionVectors'

export class Fighter extends Entity {
  static movePower = 4
  torso: Torso
  weapon: Weapon
  spawnPoint: Vec2
  deathPoint: Vec2
  action = 0 // new Vec2(0, 0)
  color = 'hsl(220,100%,40%)'
  dead = false

  constructor (simulation: Simulation, position: Vec2) {
    super(simulation, {
      type: 'dynamic',
      bullet: true,
      linearDamping: 0.7,
      fixedRotation: true
    })
    this.label = 'fighter'
    this.body.setPosition(position)
    this.spawnPoint = position
    this.deathPoint = position
    this.torso = new Torso(this)
    this.weapon = new Weapon(this)
    this.simulation.fighters.set(this.id, this)
  }

  preStep (dt: number): void {
    super.preStep(dt)
    const dir = actionVectors[this.action]
    this.force = Vec2.mul(Fighter.movePower, dir)
  }

  summarize (): FighterSummary {
    return {
      torso: this.body.getPosition(),
      weapon: this.weapon.body.getPosition(),
      torsoColor: this.color,
      weaponColor: this.weapon.color,
      history: this.history,
      weaponHistory: this.weapon.history
    }
  }

  remove (): void {
    super.remove()
    this.simulation.fighters.delete(this.id)
  }
}
