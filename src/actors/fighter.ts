import { Vec2 } from 'planck'
import { Actor } from './actor'
import { Torso } from '../features/torso'
import { Stage } from '../stage'
import { Weapon } from './weapon'

export class Fighter extends Actor {
  torso: Torso
  weapon: Weapon
  spawnPoint: Vec2
  deathPoint: Vec2
  movePower = 4
  moveDir = new Vec2(0, 0)
  color = 'hsl(220,100%,40%)'
  dead = false

  constructor (stage: Stage, position: Vec2) {
    super(stage, {
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
  }
}
