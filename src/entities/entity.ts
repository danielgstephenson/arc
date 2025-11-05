import { Body, BodyDef, Fixture, Vec2 } from 'planck'
import { Simulation } from '../simulation'

export class Entity {
  simulation: Simulation
  id: number
  body: Body
  force = Vec2.zero()
  label = 'actor'
  removed = false

  constructor (simulation: Simulation, bodyDef: BodyDef) {
    this.simulation = simulation
    this.simulation.entityCount += 1
    this.id = simulation.entityCount
    this.simulation.entities.set(this.id, this)
    this.body = this.simulation.world.createBody(bodyDef)
    this.body.setUserData(this)
  }

  getFixtures (): Fixture[] {
    const fixtures = []
    for (let fixture = this.body.getFixtureList(); fixture != null; fixture = fixture.getNext()) {
      fixtures.push(fixture)
    }
    return fixtures
  }

  preStep (dt: number): void {}

  postStep (dt: number): void {
    if (this.removed) {
      this.simulation.world.destroyBody(this.body)
      this.simulation.entities.delete(this.id)
    }
  }

  remove (): void {
    this.removed = true
  }
}
