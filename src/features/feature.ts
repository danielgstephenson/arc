import { Fixture, FixtureDef } from 'planck'
import { Entity } from '../entities/entity'

export class Feature {
  entity: Entity
  fixture: Fixture
  label = 'feature'

  constructor (entity: Entity, fixtureDef: FixtureDef) {
    this.entity = entity
    this.fixture = this.entity.body.createFixture(fixtureDef)
    this.fixture.setUserData(this)
  }
}
