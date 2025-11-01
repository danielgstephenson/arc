import { Body, BodyDef, Fixture } from 'planck'
import { Sim } from '../sim'

export class Actor {
  sim: Sim
  id: string
  body: Body
  label = 'actor'
  removed = false

  constructor (sim: Sim, bodyDef: BodyDef) {
    this.sim = sim
    this.sim.actorCount += 1
    this.id = String(sim.actorCount)
    this.sim.actors.set(this.id, this)
    this.body = this.sim.world.createBody(bodyDef)
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
      this.sim.world.destroyBody(this.body)
      this.sim.actors.delete(this.id)
    }
  }

  remove (): void {
    this.removed = true
  }
}
