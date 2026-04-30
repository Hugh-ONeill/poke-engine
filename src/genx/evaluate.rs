use super::abilities::Abilities;
use super::items::Items;
use super::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::choices::MoveCategory;
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, State};

const POKEMON_ALIVE: f32 = 30.0;
const POKEMON_HP: f32 = 100.0;
const USED_TERA: f32 = -75.0;

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
const AURORA_VEIL: f32 = 40.0;
const SAFE_GUARD: f32 = 5.0;
const TAILWIND: f32 = 7.0;
const HEALING_WISH: f32 = 30.0;

const STEALTH_ROCK: f32 = -10.0;
const SPIKES: f32 = -7.0;
const TOXIC_SPIKES: f32 = -7.0;
const STICKY_WEB: f32 = -25.0;

// Weather / terrain — applied to the active mon on each side.
// Speed-doubling weather abilities are large because outspeeding flips matchups outright.
// Type/passive numbers are smaller — meaningful but not dominant.
const WEATHER_TYPE_BOOSTED: f32 = 8.0;
const WEATHER_TYPE_SUPPRESSED: f32 = -8.0;
const WEATHER_SPEED_ABILITY: f32 = 25.0;
const WEATHER_PASSIVE_DAMAGE: f32 = -8.0;
const WEATHER_PASSIVE_HEAL: f32 = 6.0;
const WEATHER_ABILITY_MINOR: f32 = 5.0;

const TERRAIN_GRASSY_HEAL: f32 = 5.0;
const TERRAIN_MISTY_STATUS_BLOCK: f32 = 12.0;
const TERRAIN_MISTY_DRAGON_RESIST: f32 = 4.0;
const TERRAIN_ELECTRIC_SLEEP_BLOCK: f32 = 8.0;
const TERRAIN_PSYCHIC_PRIORITY_BLOCK: f32 = 4.0;
const TERRAIN_TYPE_BOOSTED: f32 = 5.0;

// Pending side-level effects (stored on the side that benefits).
const WISH_PENDING: f32 = 15.0;
const FUTURE_SIGHT_PENDING: f32 = 18.0;

// Perish song — counter on active. PERISH1 ≈ KO next turn (~POKEMON_HP + POKEMON_ALIVE).
const PERISH_1: f32 = -120.0;
const PERISH_2: f32 = -50.0;
const PERISH_3: f32 = -20.0;
const PERISH_4: f32 = -10.0;

// Active volatile statuses beyond the existing LEECHSEED/SUBSTITUTE/CONFUSION.
const ENCORE: f32 = -25.0;
const TAUNT: f32 = -15.0;
const YAWN: f32 = -25.0;          // mirrors POKEMON_ASLEEP — landing next turn
const SALTCURE: f32 = -15.0;
const SALTCURE_WEAK: f32 = -30.0; // water/steel take 1/4
const DESTINY_BOND: f32 = 5.0;
const DISABLE: f32 = -10.0;
const TORMENT: f32 = -10.0;
const HEAL_BLOCK: f32 = -15.0;
const OCTOLOCK: f32 = -20.0;
const PARTIALLY_TRAPPED: f32 = -8.0;
const INGRAIN: f32 = 8.0;
const AQUA_RING: f32 = 6.0;
const MAGNET_RISE: f32 = 5.0;
const TAR_SHOT: f32 = -5.0;
const GLAIVE_RUSH: f32 = -30.0;
const SLOW_START: f32 = -25.0;
const FOCUS_ENERGY: f32 = 5.0;
const LASER_FOCUS: f32 = 8.0;
const NIGHTMARE_VS: f32 = -15.0;  // only relevant when asleep — but ENGINE only sets it then
const CURSE_ON_ACTIVE: f32 = -25.0;

fn evaluate_poison(pokemon: &Pokemon, base_score: f32) -> f32 {
    match pokemon.ability {
        Abilities::POISONHEAL => 15.0,
        Abilities::GUTS
        | Abilities::MARVELSCALE
        | Abilities::QUICKFEET
        | Abilities::TOXICBOOST
        | Abilities::MAGICGUARD => 10.0,
        _ => base_score,
    }
}

fn evaluate_burned(pokemon: &Pokemon) -> f32 {
    // burn is not as punishing in certain situations

    // guts, marvel scale, quick feet will result in a positive evaluation
    match pokemon.ability {
        Abilities::GUTS | Abilities::MARVELSCALE | Abilities::QUICKFEET => {
            return -2.0 * POKEMON_BURNED
        }
        _ => {}
    }

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

fn evaluate_hazards(pokemon: &Pokemon, side: &Side) -> f32 {
    let mut score = 0.0;
    let pkmn_is_grounded = pokemon.is_grounded();
    if pokemon.item != Items::HEAVYDUTYBOOTS {
        if pokemon.ability != Abilities::MAGICGUARD {
            score += side.side_conditions.stealth_rock as f32 * STEALTH_ROCK;
            if pkmn_is_grounded {
                score += side.side_conditions.spikes as f32 * SPIKES;
                score += side.side_conditions.toxic_spikes as f32 * TOXIC_SPIKES;
            }
        }
        if pkmn_is_grounded {
            score += side.side_conditions.sticky_web as f32 * STICKY_WEB;
        }
    }

    score
}

fn evaluate_active_volatiles(pokemon: &Pokemon, side: &Side) -> f32 {
    let mut score = 0.0;
    for vs in side.volatile_statuses.iter() {
        match vs {
            PokemonVolatileStatus::LEECHSEED => score += LEECH_SEED,
            PokemonVolatileStatus::SUBSTITUTE => score += SUBSTITUTE,
            PokemonVolatileStatus::CONFUSION => score += CONFUSION,
            PokemonVolatileStatus::PERISH1 => score += PERISH_1,
            PokemonVolatileStatus::PERISH2 => score += PERISH_2,
            PokemonVolatileStatus::PERISH3 => score += PERISH_3,
            PokemonVolatileStatus::PERISH4 => score += PERISH_4,
            PokemonVolatileStatus::ENCORE => score += ENCORE,
            PokemonVolatileStatus::TAUNT => score += TAUNT,
            PokemonVolatileStatus::DISABLE => score += DISABLE,
            PokemonVolatileStatus::TORMENT => score += TORMENT,
            PokemonVolatileStatus::HEALBLOCK => score += HEAL_BLOCK,
            PokemonVolatileStatus::OCTOLOCK => score += OCTOLOCK,
            PokemonVolatileStatus::YAWN => score += YAWN,
            PokemonVolatileStatus::SALTCURE => {
                if pokemon.has_type(&PokemonType::WATER)
                    || pokemon.has_type(&PokemonType::STEEL)
                {
                    score += SALTCURE_WEAK;
                } else {
                    score += SALTCURE;
                }
            }
            PokemonVolatileStatus::PARTIALLYTRAPPED => score += PARTIALLY_TRAPPED,
            PokemonVolatileStatus::TARSHOT => score += TAR_SHOT,
            PokemonVolatileStatus::GLAIVERUSH => score += GLAIVE_RUSH,
            PokemonVolatileStatus::SLOWSTART => score += SLOW_START,
            PokemonVolatileStatus::NIGHTMARE => score += NIGHTMARE_VS,
            PokemonVolatileStatus::CURSE => score += CURSE_ON_ACTIVE,
            PokemonVolatileStatus::INGRAIN => score += INGRAIN,
            PokemonVolatileStatus::AQUARING => score += AQUA_RING,
            PokemonVolatileStatus::MAGNETRISE => score += MAGNET_RISE,
            PokemonVolatileStatus::DESTINYBOND => score += DESTINY_BOND,
            PokemonVolatileStatus::FOCUSENERGY => score += FOCUS_ENERGY,
            PokemonVolatileStatus::LASERFOCUS => score += LASER_FOCUS,
            _ => {}
        }
    }
    score
}

// Wish heals the side that has it pending; future_sight is stored on the caster's
// side and lands on the opponent. Both score positive for the side they're set on.
fn evaluate_pending_effects(side: &Side) -> f32 {
    let mut score = 0.0;
    if side.future_sight.0 > 0 {
        score += FUTURE_SIGHT_PENDING;
    }
    if side.wish.0 > 0 {
        score += WISH_PENDING;
    }
    score
}

fn evaluate_weather_for_active(pokemon: &Pokemon, weather: Weather) -> f32 {
    let mut score = 0.0;
    match weather {
        Weather::SUN | Weather::HARSHSUN => {
            if pokemon.has_type(&PokemonType::FIRE) { score += WEATHER_TYPE_BOOSTED; }
            if pokemon.has_type(&PokemonType::WATER) { score += WEATHER_TYPE_SUPPRESSED; }
            match pokemon.ability {
                Abilities::CHLOROPHYLL => score += WEATHER_SPEED_ABILITY,
                Abilities::SOLARPOWER => score += WEATHER_ABILITY_MINOR,
                Abilities::FLOWERGIFT => score += WEATHER_ABILITY_MINOR,
                Abilities::LEAFGUARD => score += WEATHER_ABILITY_MINOR,
                Abilities::DRYSKIN => score += WEATHER_PASSIVE_DAMAGE,
                _ => {}
            }
        }
        Weather::RAIN | Weather::HEAVYRAIN => {
            if pokemon.has_type(&PokemonType::WATER) { score += WEATHER_TYPE_BOOSTED; }
            if pokemon.has_type(&PokemonType::FIRE) { score += WEATHER_TYPE_SUPPRESSED; }
            match pokemon.ability {
                Abilities::SWIFTSWIM => score += WEATHER_SPEED_ABILITY,
                Abilities::RAINDISH => score += WEATHER_PASSIVE_HEAL,
                Abilities::DRYSKIN => score += WEATHER_PASSIVE_HEAL,
                Abilities::HYDRATION => score += WEATHER_ABILITY_MINOR,
                _ => {}
            }
        }
        Weather::SAND => {
            if pokemon.has_type(&PokemonType::ROCK) { score += WEATHER_TYPE_BOOSTED; }
            let immune_to_chip = pokemon.has_type(&PokemonType::ROCK)
                || pokemon.has_type(&PokemonType::GROUND)
                || pokemon.has_type(&PokemonType::STEEL)
                || matches!(pokemon.ability,
                    Abilities::MAGICGUARD | Abilities::OVERCOAT
                    | Abilities::SANDVEIL | Abilities::SANDFORCE | Abilities::SANDRUSH);
            if !immune_to_chip { score += WEATHER_PASSIVE_DAMAGE; }
            match pokemon.ability {
                Abilities::SANDRUSH => score += WEATHER_SPEED_ABILITY,
                Abilities::SANDFORCE => score += WEATHER_ABILITY_MINOR,
                Abilities::SANDVEIL => score += WEATHER_ABILITY_MINOR,
                _ => {}
            }
        }
        Weather::HAIL | Weather::SNOW => {
            if pokemon.has_type(&PokemonType::ICE) { score += WEATHER_TYPE_BOOSTED; }
            if weather == Weather::HAIL {
                let immune_to_chip = pokemon.has_type(&PokemonType::ICE)
                    || matches!(pokemon.ability,
                        Abilities::MAGICGUARD | Abilities::OVERCOAT
                        | Abilities::ICEBODY | Abilities::SNOWCLOAK | Abilities::SLUSHRUSH);
                if !immune_to_chip { score += WEATHER_PASSIVE_DAMAGE; }
            }
            match pokemon.ability {
                Abilities::SLUSHRUSH => score += WEATHER_SPEED_ABILITY,
                Abilities::ICEBODY => score += WEATHER_PASSIVE_HEAL,
                Abilities::SNOWCLOAK => score += WEATHER_ABILITY_MINOR,
                _ => {}
            }
        }
        Weather::NONE => {}
    }
    score
}

fn evaluate_terrain_for_active(pokemon: &Pokemon, terrain: Terrain) -> f32 {
    if !pokemon.is_grounded() { return 0.0; }
    let mut score = 0.0;
    match terrain {
        Terrain::ELECTRICTERRAIN => {
            score += TERRAIN_ELECTRIC_SLEEP_BLOCK;
            if pokemon.has_type(&PokemonType::ELECTRIC) { score += TERRAIN_TYPE_BOOSTED; }
            if pokemon.ability == Abilities::SURGESURFER { score += WEATHER_SPEED_ABILITY; }
        }
        Terrain::GRASSYTERRAIN => {
            score += TERRAIN_GRASSY_HEAL;
            if pokemon.has_type(&PokemonType::GRASS) { score += TERRAIN_TYPE_BOOSTED; }
            if pokemon.ability == Abilities::GRASSPELT { score += WEATHER_ABILITY_MINOR; }
        }
        Terrain::MISTYTERRAIN => {
            score += TERRAIN_MISTY_STATUS_BLOCK;
            // dragon damage halved against grounded non-fairy mons
            if !pokemon.has_type(&PokemonType::FAIRY) {
                score += TERRAIN_MISTY_DRAGON_RESIST;
            }
        }
        Terrain::PSYCHICTERRAIN => {
            score += TERRAIN_PSYCHIC_PRIORITY_BLOCK;
            if pokemon.has_type(&PokemonType::PSYCHIC) { score += TERRAIN_TYPE_BOOSTED; }
        }
        Terrain::NONE => {}
    }
    score
}

fn evaluate_pokemon(pokemon: &Pokemon) -> f32 {
    let mut score = 0.0;
    score += POKEMON_HP * pokemon.hp as f32 / pokemon.maxhp as f32;

    match pokemon.status {
        PokemonStatus::BURN => score += evaluate_burned(pokemon),
        PokemonStatus::FREEZE => score += POKEMON_FROZEN,
        PokemonStatus::SLEEP => score += POKEMON_ASLEEP,
        PokemonStatus::PARALYZE => score += POKEMON_PARALYZED,
        PokemonStatus::TOXIC => score += evaluate_poison(pokemon, POKEMON_TOXIC),
        PokemonStatus::POISON => score += evaluate_poison(pokemon, POKEMON_POISONED),
        PokemonStatus::NONE => {}
    }

    if pokemon.item != Items::NONE {
        score += 10.0;
    }

    // without this a low hp pokemon could get a negative score and incentivize the other side
    // to keep it alive
    if score < 0.0 {
        score = 0.0;
    }

    score += POKEMON_ALIVE;

    score
}

pub fn evaluate(state: &State) -> f32 {
    let mut score = 0.0;

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn);
            score += evaluate_hazards(pkmn, &state.side_one);
            if iter.pokemon_index == state.side_one.active_index {
                score += evaluate_active_volatiles(pkmn, &state.side_one);

                score += get_boost_multiplier(state.side_one.attack_boost) * POKEMON_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
        if pkmn.terastallized {
            s1_used_tera = true;
        }
    }
    if s1_used_tera {
        score += USED_TERA;
    }
    let mut iter = state.side_two.pokemon.into_iter();
    let mut s2_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score -= evaluate_pokemon(pkmn);
            score -= evaluate_hazards(pkmn, &state.side_two);

            if iter.pokemon_index == state.side_two.active_index {
                score -= evaluate_active_volatiles(pkmn, &state.side_two);

                score -= get_boost_multiplier(state.side_two.attack_boost) * POKEMON_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST;
                score -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.speed_boost) * POKEMON_SPEED_BOOST;
            }
        }
        if pkmn.terastallized {
            s2_used_tera = true;
        }
    }
    if s2_used_tera {
        score -= USED_TERA;
    }

    score += state.side_one.side_conditions.reflect as f32 * REFLECT;
    score += state.side_one.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score += state.side_one.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    score += state.side_one.side_conditions.safeguard as f32 * SAFE_GUARD;
    score += state.side_one.side_conditions.tailwind as f32 * TAILWIND;
    score += state.side_one.side_conditions.healing_wish as f32 * HEALING_WISH;

    score -= state.side_two.side_conditions.reflect as f32 * REFLECT;
    score -= state.side_two.side_conditions.light_screen as f32 * LIGHT_SCREEN;
    score -= state.side_two.side_conditions.aurora_veil as f32 * AURORA_VEIL;
    score -= state.side_two.side_conditions.safeguard as f32 * SAFE_GUARD;
    score -= state.side_two.side_conditions.tailwind as f32 * TAILWIND;
    score -= state.side_two.side_conditions.healing_wish as f32 * HEALING_WISH;

    score += evaluate_pending_effects(&state.side_one);
    score -= evaluate_pending_effects(&state.side_two);

    let weather = state.weather.weather_type;
    if weather != Weather::NONE {
        let s1_active = state.side_one.get_active_immutable();
        let s2_active = state.side_two.get_active_immutable();
        if s1_active.hp > 0 {
            score += evaluate_weather_for_active(s1_active, weather);
        }
        if s2_active.hp > 0 {
            score -= evaluate_weather_for_active(s2_active, weather);
        }
    }

    let terrain = state.terrain.terrain_type;
    if terrain != Terrain::NONE {
        let s1_active = state.side_one.get_active_immutable();
        let s2_active = state.side_two.get_active_immutable();
        if s1_active.hp > 0 {
            score += evaluate_terrain_for_active(s1_active, terrain);
        }
        if s2_active.hp > 0 {
            score -= evaluate_terrain_for_active(s2_active, terrain);
        }
    }

    score
}
