import { Contact, Vec2 } from 'planck'
import { Feature } from './features/feature'
import { Simulation } from './simulation/simulation'
import { Torso } from './features/torso'
import { Blade } from './features/blade'

export class Collider {
  simulation: Simulation

  constructor (simulation: Simulation) {
    this.simulation = simulation
    this.simulation.world.on('pre-solve', contact => this.preSolve(contact))
    this.simulation.world.on('begin-contact', contact => this.beginContact(contact))
    this.simulation.world.on('end-contact', contact => this.endContact(contact))
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
