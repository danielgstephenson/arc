import { Body, BodyDef, Fixture, Vec2 } from 'planck'
import { Stage } from '../stage'

export class Actor {
  stage: Stage
  id: number
  body: Body
  force = Vec2.zero()
  label = 'actor'
  removed = false

  constructor (stage: Stage, bodyDef: BodyDef) {
    this.stage = stage
    this.stage.actorCount += 1
    this.id = stage.actorCount
    this.stage.actors.set(this.id, this)
    this.body = this.stage.world.createBody(bodyDef)
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
      this.stage.world.destroyBody(this.body)
      this.stage.actors.delete(this.id)
    }
  }

  remove (): void {
    this.removed = true
  }
}
