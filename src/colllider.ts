import { Contact, Vec2 } from 'planck'
import { Feature } from './features/feature'
import { Stage } from './stage'
import { Torso } from './features/torso'
import { Blade } from './features/blade'

export class Collider {
  stage: Stage

  constructor (stage: Stage) {
    this.stage = stage
    this.stage.world.on('pre-solve', contact => this.preSolve(contact))
    this.stage.world.on('begin-contact', contact => this.beginContact(contact))
    this.stage.world.on('end-contact', contact => this.endContact(contact))
  }

  beginContact (contact: Contact): void {}

  endContact (contact: Contact): void {}

  preSolve (contact: Contact): void {
    const a = contact.getFixtureA().getUserData() as Feature
    const b = contact.getFixtureB().getUserData() as Feature
    const pairs = [[a, b], [b, a]]
    pairs.forEach(pair => {
      const featureA = pair[0]
      const featureB = pair[1]
      if (featureA instanceof Torso && featureB instanceof Blade) {
        contact.setEnabled(false)
        if (featureA.fighter.id === featureB.weapon.fighter.id) return
        const manifold = contact.getWorldManifold(null)
        if (manifold == null) return
        if (manifold.pointCount === 0) return
        const overlap = -Math.min(...manifold.separations)
        if (overlap > 0) {
          featureA.fighter.deathPoint = new Vec2(manifold.points[0])
          featureA.fighter.dead = true
        }
      }
    })
  }
}
