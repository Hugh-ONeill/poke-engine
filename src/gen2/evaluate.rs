use super::damage_calc::type_effectiveness_modifier;
use super::items::Items;
use super::state::PokemonVolatileStatus;
use crate::choices::{Choices, MoveCategory};
use crate::state::{Pokemon, PokemonStatus, State};

const POKEMON_ALIVE: f32 = 60.0;
const POKEMON_HP: f32 = 100.0;

const POKEMON_ATTACK_BOOST: f32 = 30.0;
const POKEMON_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPECIAL_ATTACK_BOOST: f32 = 30.0;
const POKEMON_SPECIAL_DEFENSE_BOOST: f32 = 15.0;
const POKEMON_SPEED_BOOST: f32 = 30.0;

const POKEMON_BOOST_MULTIPLIER_6: f32 = 3.3;
const POKEMON_BOOST_MULTIPLIER_5: f32 = 3.15;
const POKEMON_BOOST_MULTIPLIER_4: f32 = 3.0;
const POKEMON_BOOST_MULTIPLIER_3: f32 = 2.5;
const POKEMON_BOOST_MULTIPLIER_2: f32 = 2.0;
const POKEMON_BOOST_MULTIPLIER_1: f32 = 1.0;
const POKEMON_BOOST_MULTIPLIER_0: f32 = 0.0;
const POKEMON_BOOST_MULTIPLIER_NEG_1: f32 = -1.0;
const POKEMON_BOOST_MULTIPLIER_NEG_2: f32 = -2.0;
const POKEMON_BOOST_MULTIPLIER_NEG_3: f32 = -2.5;
const POKEMON_BOOST_MULTIPLIER_NEG_4: f32 = -3.0;
const POKEMON_BOOST_MULTIPLIER_NEG_5: f32 = -3.15;
const POKEMON_BOOST_MULTIPLIER_NEG_6: f32 = -3.3;

const POKEMON_FROZEN: f32 = -40.0;
const POKEMON_ASLEEP: f32 = -25.0;
const POKEMON_PARALYZED: f32 = -25.0;
const POKEMON_TOXIC: f32 = -30.0;
const POKEMON_POISONED: f32 = -10.0;
const POKEMON_BURNED: f32 = -25.0;

const LEECH_SEED: f32 = -30.0;
const SUBSTITUTE: f32 = 40.0;
const CONFUSION: f32 = -20.0;

const REFLECT: f32 = 20.0;
const LIGHT_SCREEN: f32 = 20.0;
const SAFE_GUARD: f32 = 5.0;

const SPIKES: f32 = -12.0;

// extras for under-used gen2 state
const POKEMON_RECOVERY_MOVE: f32 = 8.0;
const HOPELESS_MATCHUP: f32 = -50.0;
const SPEED_TIER_BONUS: f32 = 20.0;
const ENCORE_PENALTY: f32 = -30.0;
const DISABLE_PENALTY: f32 = -12.0;
const RECHARGE_PENALTY: f32 = -35.0;
const PARTIALLY_TRAPPED_PENALTY: f32 = -15.0;
const FORESIGHT_PENALTY: f32 = -10.0;
// PERISH4..PERISH1 = [4..1] turns left; closer to 0 hurts more
const PERISH_PENALTY: [f32; 4] = [-12.0, -25.0, -50.0, -80.0];
const FUTURE_SIGHT_INCOMING: f32 = -25.0;

fn evaluate_burned(pokemon: &Pokemon) -> f32 {
    // burn is not as punishing in certain situations

    let mut multiplier = 0.0;
    for mv in pokemon.moves.into_iter() {
        if mv.choice.category == MoveCategory::Physical {
            multiplier += 1.0;
        }
    }

    // don't make burn as punishing for special attackers
    if pokemon.special_attack > pokemon.attack {
        multiplier /= 2.0;
    }

    multiplier * POKEMON_BURNED
}

fn get_boost_multiplier(boost: i8) -> f32 {
    match boost {
        6 => POKEMON_BOOST_MULTIPLIER_6,
        5 => POKEMON_BOOST_MULTIPLIER_5,
        4 => POKEMON_BOOST_MULTIPLIER_4,
        3 => POKEMON_BOOST_MULTIPLIER_3,
        2 => POKEMON_BOOST_MULTIPLIER_2,
        1 => POKEMON_BOOST_MULTIPLIER_1,
        0 => POKEMON_BOOST_MULTIPLIER_0,
        -1 => POKEMON_BOOST_MULTIPLIER_NEG_1,
        -2 => POKEMON_BOOST_MULTIPLIER_NEG_2,
        -3 => POKEMON_BOOST_MULTIPLIER_NEG_3,
        -4 => POKEMON_BOOST_MULTIPLIER_NEG_4,
        -5 => POKEMON_BOOST_MULTIPLIER_NEG_5,
        -6 => POKEMON_BOOST_MULTIPLIER_NEG_6,
        _ => panic!("Invalid boost value: {}", boost),
    }
}

fn has_sleep_talk(pokemon: &Pokemon) -> bool {
    for mv in pokemon.moves.into_iter() {
        if mv.id == crate::choices::Choices::SLEEPTALK {
            return true;
        }
    }
    false
}

fn has_recovery_move(pokemon: &Pokemon) -> bool {
    for mv in pokemon.moves.into_iter() {
        match mv.id {
            Choices::REST
            | Choices::RECOVER
            | Choices::SOFTBOILED
            | Choices::MILKDRINK
            | Choices::MOONLIGHT
            | Choices::MORNINGSUN
            | Choices::SYNTHESIS => return true,
            _ => {}
        }
    }
    false
}

/// How well can this pokemon's physical/special moves hit the defender?
/// Returns (physical_threat, special_threat) in [0.0, 1.0] range.
/// 0.0 = all moves immune, 1.0 = at least one move is super effective.
fn threat_vs(attacker: &Pokemon, defender: &Pokemon) -> (f32, f32) {
    let mut best_phys: f32 = 0.0;
    let mut best_spec: f32 = 0.0;

    for mv in attacker.moves.into_iter() {
        if mv.id == crate::choices::Choices::NONE { continue; }
        let eff = type_effectiveness_modifier(&mv.choice.move_type, defender);
        match mv.choice.category {
            MoveCategory::Physical => best_phys = best_phys.max(eff),
            MoveCategory::Special => best_spec = best_spec.max(eff),
            _ => {}
        }
    }

    // cap at 1.0 (super effective = full value, anything beyond is bonus but not needed)
    (best_phys.min(1.0), best_spec.min(1.0))
}

fn evaluate_pokemon(pokemon: &Pokemon) -> f32 {
    let mut score = 0.0;
    score += POKEMON_HP * pokemon.hp as f32 / pokemon.maxhp as f32;

    match pokemon.status {
        PokemonStatus::BURN => score += evaluate_burned(pokemon),
        PokemonStatus::FREEZE => score += POKEMON_FROZEN,
        PokemonStatus::SLEEP => {
            // sleep is much less punishing with Sleep Talk
            if has_sleep_talk(pokemon) {
                score += POKEMON_ASLEEP * 0.4;
            } else {
                score += POKEMON_ASLEEP;
            }
        }
        PokemonStatus::PARALYZE => score += POKEMON_PARALYZED,
        PokemonStatus::TOXIC => score += POKEMON_TOXIC,
        PokemonStatus::POISON => score += POKEMON_POISONED,
        PokemonStatus::NONE => {}
    }

    if pokemon.item != Items::NONE {
        score += 10.0;
    }

    if has_recovery_move(pokemon) {
        score += POKEMON_RECOVERY_MOVE;
    }

    if score < 0.0 {
        score = 0.0;
    }

    score += POKEMON_ALIVE;

    score
}

pub fn evaluate(state: &State) -> f32 {
    let mut score = 0.0;
    let mut side_one_alive_count: f32 = 0.0;
    let mut side_two_alive_count: f32 = 0.0;

    // get active pokemon for matchup-aware boost evaluation
    let s1_active = &state.side_one.pokemon[state.side_one.active_index];
    let s2_active = &state.side_two.pokemon[state.side_two.active_index];

    // how well can each side's moves hit the other?
    let (s1_phys_threat, s1_spec_threat) = threat_vs(s1_active, s2_active);
    let (s2_phys_threat, s2_spec_threat) = threat_vs(s2_active, s1_active);

    let mut iter = state.side_one.pokemon.into_iter();
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            side_one_alive_count += 1.0;
            score += evaluate_pokemon(pkmn);
            if iter.pokemon_index == state.side_one.active_index {
                for vs in state.side_one.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score += LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score += SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score += CONFUSION,
                        PokemonVolatileStatus::ENCORE => score += ENCORE_PENALTY,
                        PokemonVolatileStatus::DISABLE => score += DISABLE_PENALTY,
                        PokemonVolatileStatus::MUSTRECHARGE => score += RECHARGE_PENALTY,
                        PokemonVolatileStatus::PARTIALLYTRAPPED => score += PARTIALLY_TRAPPED_PENALTY,
                        PokemonVolatileStatus::FORESIGHT => score += FORESIGHT_PENALTY,
                        PokemonVolatileStatus::PERISH4 => score += PERISH_PENALTY[0],
                        PokemonVolatileStatus::PERISH3 => score += PERISH_PENALTY[1],
                        PokemonVolatileStatus::PERISH2 => score += PERISH_PENALTY[2],
                        PokemonVolatileStatus::PERISH1 => score += PERISH_PENALTY[3],
                        _ => {}
                    }
                }
                // attack boosts only matter if we can hit the opponent
                score += get_boost_multiplier(state.side_one.attack_boost)
                    * POKEMON_ATTACK_BOOST * s1_phys_threat;
                score += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s1_spec_threat;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
    }
    let mut iter = state.side_two.pokemon.into_iter();
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            side_two_alive_count += 1.0;
            score -= evaluate_pokemon(pkmn);

            if iter.pokemon_index == state.side_two.active_index {
                for vs in state.side_two.volatile_statuses.iter() {
                    match vs {
                        PokemonVolatileStatus::LEECHSEED => score -= LEECH_SEED,
                        PokemonVolatileStatus::SUBSTITUTE => score -= SUBSTITUTE,
                        PokemonVolatileStatus::CONFUSION => score -= CONFUSION,
                        PokemonVolatileStatus::ENCORE => score -= ENCORE_PENALTY,
                        PokemonVolatileStatus::DISABLE => score -= DISABLE_PENALTY,
                        PokemonVolatileStatus::MUSTRECHARGE => score -= RECHARGE_PENALTY,
                        PokemonVolatileStatus::PARTIALLYTRAPPED => score -= PARTIALLY_TRAPPED_PENALTY,
                        PokemonVolatileStatus::FORESIGHT => score -= FORESIGHT_PENALTY,
                        PokemonVolatileStatus::PERISH4 => score -= PERISH_PENALTY[0],
                        PokemonVolatileStatus::PERISH3 => score -= PERISH_PENALTY[1],
                        PokemonVolatileStatus::PERISH2 => score -= PERISH_PENALTY[2],
                        PokemonVolatileStatus::PERISH1 => score -= PERISH_PENALTY[3],
                        _ => {}
                    }
                }
                score -= get_boost_multiplier(state.side_two.attack_boost)
                    * POKEMON_ATTACK_BOOST * s2_phys_threat;
                score -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s2_spec_threat;
                score -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
    }

    score += state.side_one.side_conditions.reflect as f32 * REFLECT;
    score += state.side_one.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score += state.side_one.side_conditions.safeguard as f32 * SAFE_GUARD;
    score += state.side_one.side_conditions.spikes as f32 * SPIKES * side_one_alive_count;

    score -= state.side_two.side_conditions.reflect as f32 * REFLECT;
    score -= state.side_two.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score -= state.side_two.side_conditions.safeguard as f32 * SAFE_GUARD;
    score -= state.side_two.side_conditions.spikes as f32 * SPIKES * side_two_alive_count;

    // hopeless matchup: an active that can't damage the opponent at all is
    // dead weight this turn (must switch). gen2 has no abilities, so this
    // collapses purely to type effectiveness.
    if s1_active.hp > 0 && s1_phys_threat == 0.0 && s1_spec_threat == 0.0 {
        score += HOPELESS_MATCHUP;
    }
    if s2_active.hp > 0 && s2_phys_threat == 0.0 && s2_spec_threat == 0.0 {
        score -= HOPELESS_MATCHUP;
    }

    // speed-tier advantage: outspeeding only matters if you can actually hit.
    // gen2 has no Trick Room, so faster always moves first (paralysis aside).
    let s1_max_threat = s1_phys_threat.max(s1_spec_threat);
    let s2_max_threat = s2_phys_threat.max(s2_spec_threat);
    if s1_active.hp > 0 && s2_active.hp > 0 {
        if s1_active.speed > s2_active.speed && s1_max_threat > 0.0 {
            score += SPEED_TIER_BONUS * s1_max_threat;
        } else if s2_active.speed > s1_active.speed && s2_max_threat > 0.0 {
            score -= SPEED_TIER_BONUS * s2_max_threat;
        }
    }

    // future sight pending damage. Side struct's future_sight = (turns_left, originator).
    // turns_left > 0 means damage is incoming on THIS side from the originator.
    if state.side_one.future_sight.0 > 0 {
        score += FUTURE_SIGHT_INCOMING;
    }
    if state.side_two.future_sight.0 > 0 {
        score -= FUTURE_SIGHT_INCOMING;
    }

    score
}
