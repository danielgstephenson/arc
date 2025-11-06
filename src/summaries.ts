import { Vec2 } from 'planck'

export interface ArenaSummary {
  boundary: Vec2[]
}

export interface FighterSummary {
  torso: Vec2
  weapon: Vec2
  torsoColor: string
  weaponColor: string
}

export interface SimulationSummary {
  arena: ArenaSummary
  fighters: FighterSummary[]
  player?: Vec2
}
