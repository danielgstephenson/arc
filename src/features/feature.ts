import { Fixture, FixtureDef } from 'planck'
import { Actor } from '../actors/actor'

export class Feature {
  actor: Actor
  fixture: Fixture
  label = 'feature'

  constructor (actor: Actor, fixtureDef: FixtureDef) {
    this.actor = actor
    this.fixture = this.actor.body.createFixture(fixtureDef)
    this.fixture.setUserData(this)
  }
}
