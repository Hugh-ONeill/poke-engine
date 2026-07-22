use super::abilities::Abilities;
use super::damage_calc::type_effectiveness_modifier;
use super::items::Items;
use super::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::choices::{Choices, MoveCategory};
use crate::state::{Pokemon, PokemonStatus, PokemonType, Side, State};

/// Runtime switch: `CB_EVAL_BASELINE=1` reverts evaluate() to the UPSTREAM
/// v0.0.47 feature set and constants — i.e. exactly the eval foul-play runs.
///
/// Why runtime and not a cargo feature: the A/B harness spawns a fresh process
/// per game, so an env var picks the arm per process and BOTH arms run one
/// identical build. A compile-time feature would need a rebuild between arms,
/// which swaps the .so under a running series and confounds the comparison.
///
/// Read once and cached: evaluate() is on the hot path of every rollout, and
/// both arms pay the same (already-initialized) atomic load, so the switch
/// cannot bias the throughput comparison in either direction.
fn baseline_eval() -> bool {
    static BASELINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BASELINE.get_or_init(|| {
        std::env::var("CB_EVAL_BASELINE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

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

const STEALTH_ROCK: f32 = -15.0;
const SPIKES: f32 = -9.0;
// upstream v0.0.47 values, used under CB_EVAL_BASELINE
const STEALTH_ROCK_BASE: f32 = -10.0;
const SPIKES_BASE: f32 = -7.0;
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

// Paradox booster (Protosynthesis/Quark Drive). Non-speed boost is 1.3x ≈ +0.6 stage,
// speed boost is 1.5x = +1 stage. Credited at fractions of the existing stage constants.
const PARADOX_ATK: f32 = 18.0;  // 0.6 * POKEMON_ATTACK_BOOST
const PARADOX_DEF: f32 = 9.0;   // 0.6 * POKEMON_DEFENSE_BOOST
const PARADOX_SPA: f32 = 18.0;  // 0.6 * POKEMON_SPECIAL_ATTACK_BOOST
const PARADOX_SPD: f32 = 9.0;   // 0.6 * POKEMON_SPECIAL_DEFENSE_BOOST
const PARADOX_SPE: f32 = 30.0;  // 1.0 * POKEMON_SPEED_BOOST

// Tera bonuses applied to the active mon while terastallized. Offsets the flat
// USED_TERA = -75 cost to the side: a useful tera (good defensive type + STAB
// available) recovers ~30 of the 75, leaving a net cost that the search can pay
// when the matchup justifies it.
const TERA_RESIST_BONUS: f32 = 2.0;     // per resisted common attack type
const TERA_DOUBLE_RESIST_BONUS: f32 = 3.0; // 0.25x (only possible vs same-type)
const TERA_IMMUNE_BONUS: f32 = 6.0;     // immune to a common attack type
const TERA_WEAK_PENALTY: f32 = -2.0;    // 2x weakness
const TERA_STAB_AVAILABLE: f32 = 12.0;  // active has a damaging tera_type move

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
    let base = baseline_eval();
    let sr = if base { STEALTH_ROCK_BASE } else { STEALTH_ROCK };
    let sp = if base { SPIKES_BASE } else { SPIKES };
    let pkmn_is_grounded = pokemon.is_grounded();
    if pokemon.item != Items::HEAVYDUTYBOOTS {
        if pokemon.ability != Abilities::MAGICGUARD {
            score += side.side_conditions.stealth_rock as f32 * sr;
            if pkmn_is_grounded {
                score += side.side_conditions.spikes as f32 * sp;
                score += side.side_conditions.toxic_spikes as f32 * TOXIC_SPIKES;
            }
        }
        if pkmn_is_grounded {
            score += side.side_conditions.sticky_web as f32 * STICKY_WEB;
        }
    }

    score
}

const HOPELESS_MATCHUP: f32 = -50.0;
const SPEED_TIER_BONUS: f32 = 20.0;

/// How well can this pokemon hit the defender? Returns (physical_threat, special_threat,
/// has_status_move). The third bool indicates whether the active has any non-attacking
/// option — used to gate HOPELESS_MATCHUP so defensive walls (Calm Mind / Roost / hazards
/// / cleric / phaze) aren't penalized as dead weight just because their attacks are 0x.
/// type_effectiveness_modifier already handles terastallization, Levitate, etc.
// Practical threat: estimated fraction of defender HP per hit, scaled so a 2HKO
// (~50% per hit) reads as 1.0. Captures stat bulk that pure type-effectiveness misses
// — e.g. Liquidation vs full-HP Cresselia is "1.0 type-eff" but ~28% per hit, so the
// real threat is ~0.56, not 1.0. Used to gate boost values, HOPELESS, and SPEED_TIER
// so attackers stuck against walls don't accumulate phantom value.
fn threat_vs(attacker: &Pokemon, defender: &Pokemon) -> (f32, f32, bool) {
    let mut best_phys: f32 = 0.0;
    let mut best_spec: f32 = 0.0;
    let mut has_status = false;
    let def_hp = defender.maxhp.max(1) as f32;
    let atk_stat = attacker.attack as f32;
    let spa_stat = attacker.special_attack as f32;
    let def_stat = defender.defense.max(1) as f32;
    let spd_stat = defender.special_defense.max(1) as f32;
    for mv in attacker.moves.into_iter() {
        if mv.id == Choices::NONE { continue; }
        let eff = type_effectiveness_modifier(&mv.choice.move_type, defender);
        match mv.choice.category {
            MoveCategory::Physical | MoveCategory::Special => {
                if eff == 0.0 { continue; }
                let bp = mv.choice.base_power;
                if bp == 0.0 { continue; }
                let stab = if mv.choice.move_type == attacker.types.0
                    || mv.choice.move_type == attacker.types.1 {
                    1.5
                } else {
                    1.0
                };
                let (off, def) = if mv.choice.category == MoveCategory::Physical {
                    (atk_stat, def_stat)
                } else {
                    (spa_stat, spd_stat)
                };
                // Lv100 simplified damage: 0.84 * BP * (off/def) * STAB * type_eff
                let dmg = 0.84 * bp * (off / def) * stab * eff;
                let frac = dmg / def_hp;
                // 2HKO (50% per hit) → 1.0 threat; OHKO+ clamps at 1.0
                let score = (frac / 0.5).min(1.0);
                if mv.choice.category == MoveCategory::Physical {
                    best_phys = best_phys.max(score);
                } else {
                    best_spec = best_spec.max(score);
                }
            }
            MoveCategory::Status => has_status = true,
            _ => {}
        }
    }
    (best_phys, best_spec, has_status)
}

// 18 standard offensive types — used as a fixed probe set for tera defensive value.
const COMMON_ATTACK_TYPES: [PokemonType; 18] = [
    PokemonType::NORMAL, PokemonType::FIRE, PokemonType::WATER, PokemonType::ELECTRIC,
    PokemonType::GRASS, PokemonType::ICE, PokemonType::FIGHTING, PokemonType::POISON,
    PokemonType::GROUND, PokemonType::FLYING, PokemonType::PSYCHIC, PokemonType::BUG,
    PokemonType::ROCK, PokemonType::GHOST, PokemonType::DRAGON, PokemonType::DARK,
    PokemonType::STEEL, PokemonType::FAIRY,
];

// Defensive value of the active mon's current typing (post-tera if terastallized,
// since type_effectiveness_modifier handles that). Sums signed bonuses across the
// fixed offensive-type probe set. Only invoked for terastallized actives so far —
// otherwise it would partially duplicate type-aware signals already implicit in
// the search's damage rollouts.
fn evaluate_tera_active(pokemon: &Pokemon) -> f32 {
    if !pokemon.terastallized {
        return 0.0;
    }
    let mut score = 0.0;
    for atk in COMMON_ATTACK_TYPES.iter() {
        let mult = type_effectiveness_modifier(atk, pokemon);
        if mult == 0.0 {
            score += TERA_IMMUNE_BONUS;
        } else if mult <= 0.25 {
            score += TERA_DOUBLE_RESIST_BONUS;
        } else if mult <= 0.5 {
            score += TERA_RESIST_BONUS;
        } else if mult >= 2.0 {
            score += TERA_WEAK_PENALTY;
        }
    }
    // STAB on tera_type: 2x instead of 1.5x. Credit if any damaging move matches.
    for mv in pokemon.moves.into_iter() {
        if mv.choice.move_type == pokemon.tera_type
            && mv.choice.category != MoveCategory::Status
        {
            score += TERA_STAB_AVAILABLE;
            break;
        }
    }
    score
}

fn evaluate_active_volatiles(pokemon: &Pokemon, side: &Side) -> f32 {
    let mut score = 0.0;
    if baseline_eval() {
        // upstream scores exactly three volatiles
        for vs in side.volatile_statuses.iter() {
            match vs {
                PokemonVolatileStatus::LEECHSEED => score += LEECH_SEED,
                PokemonVolatileStatus::SUBSTITUTE => score += SUBSTITUTE,
                PokemonVolatileStatus::CONFUSION => score += CONFUSION,
                _ => {}
            }
        }
        return score;
    }
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
            PokemonVolatileStatus::PROTOSYNTHESISATK
            | PokemonVolatileStatus::QUARKDRIVEATK => score += PARADOX_ATK,
            PokemonVolatileStatus::PROTOSYNTHESISDEF
            | PokemonVolatileStatus::QUARKDRIVEDEF => score += PARADOX_DEF,
            PokemonVolatileStatus::PROTOSYNTHESISSPA
            | PokemonVolatileStatus::QUARKDRIVESPA => score += PARADOX_SPA,
            PokemonVolatileStatus::PROTOSYNTHESISSPD
            | PokemonVolatileStatus::QUARKDRIVESPD => score += PARADOX_SPD,
            PokemonVolatileStatus::PROTOSYNTHESISSPE
            | PokemonVolatileStatus::QUARKDRIVESPE => score += PARADOX_SPE,
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

fn evaluate_weather_for_active(pokemon: &Pokemon, weather: Weather, trick_room: bool) -> f32 {
    // Speed-doubling weather abilities are bad under Trick Room (you wanted to be slower).
    let speed_bonus = if trick_room { 0.0 } else { WEATHER_SPEED_ABILITY };
    let mut score = 0.0;
    match weather {
        Weather::SUN | Weather::HARSHSUN => {
            if pokemon.has_type(&PokemonType::FIRE) { score += WEATHER_TYPE_BOOSTED; }
            if pokemon.has_type(&PokemonType::WATER) { score += WEATHER_TYPE_SUPPRESSED; }
            match pokemon.ability {
                Abilities::CHLOROPHYLL => score += speed_bonus,
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
                Abilities::SWIFTSWIM => score += speed_bonus,
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
                Abilities::SANDRUSH => score += speed_bonus,
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
                Abilities::SLUSHRUSH => score += speed_bonus,
                Abilities::ICEBODY => score += WEATHER_PASSIVE_HEAL,
                Abilities::SNOWCLOAK => score += WEATHER_ABILITY_MINOR,
                _ => {}
            }
        }
        Weather::NONE => {}
    }
    score
}

fn evaluate_terrain_for_active(pokemon: &Pokemon, terrain: Terrain, trick_room: bool) -> f32 {
    if !pokemon.is_grounded() { return 0.0; }
    let speed_bonus = if trick_room { 0.0 } else { WEATHER_SPEED_ABILITY };
    let mut score = 0.0;
    match terrain {
        Terrain::ELECTRICTERRAIN => {
            score += TERRAIN_ELECTRIC_SLEEP_BLOCK;
            if pokemon.has_type(&PokemonType::ELECTRIC) { score += TERRAIN_TYPE_BOOSTED; }
            if pokemon.ability == Abilities::SURGESURFER { score += speed_bonus; }
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

// Per-item value. Magnitudes kept small so team-composition asymmetries don't
// distort MCTS baselines (large per-item differences shifted the eval offset by
// 19+ for some matchups, biasing per-side play). Relative ordering preserved
// so Knock Off targets the right items.
fn evaluate_item(item: Items) -> f32 {
    match item {
        Items::NONE => 0.0,
        Items::CHOICEBAND
        | Items::CHOICESPECS
        | Items::CHOICESCARF => 13.0,
        Items::LIFEORB => 9.0,
        Items::HEAVYDUTYBOOTS
        | Items::LEFTOVERS
        | Items::BLACKSLUDGE
        | Items::ASSAULTVEST => 9.0,
        Items::EVIOLITE => 13.0,
        Items::ROCKYHELMET | Items::AIRBALLOON => 7.0,
        Items::FOCUSSASH => 8.0,
        Items::BOOSTERENERGY => 4.0,
        Items::SITRUSBERRY => 6.0,
        Items::TOXICORB | Items::FLAMEORB => 4.0,
        Items::EXPERTBELT
        | Items::MUSCLEBAND
        | Items::WISEGLASSES
        | Items::PUNCHINGGLOVE => 7.0,
        Items::LOADEDDICE => 12.0,
        Items::COVERTCLOAK
        | Items::CLEARAMULET
        | Items::POWERHERB
        | Items::WEAKNESSPOLICY
        | Items::SHELLBELL => 6.0,
        _ => 5.0,
    }
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

    // upstream scores "holding any item" as a flat +10; ours prices items individually
    if baseline_eval() {
        if pokemon.item != Items::NONE {
            score += 10.0;
        }
    } else {
        score += evaluate_item(pokemon.item);
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

    let base = baseline_eval();
    let s1_active = &state.side_one.pokemon[state.side_one.active_index];
    let s2_active = &state.side_two.pokemon[state.side_two.active_index];
    // Upstream scores boosts flat; we scale offensive boosts by how hard the
    // active can actually hit. In baseline mode the multipliers are 1.0 so the
    // boost terms reduce exactly to upstream's.
    let (s1_phys, s1_spec, s1_has_status) = if base {
        (1.0, 1.0, true)
    } else {
        threat_vs(s1_active, s2_active)
    };
    let (s2_phys, s2_spec, s2_has_status) = if base {
        (1.0, 1.0, true)
    } else {
        threat_vs(s2_active, s1_active)
    };

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn);
            score += evaluate_hazards(pkmn, &state.side_one);
            if iter.pokemon_index == state.side_one.active_index {
                score += evaluate_active_volatiles(pkmn, &state.side_one);
                if !base {
                    score += evaluate_tera_active(pkmn);
                }

                score += get_boost_multiplier(state.side_one.attack_boost)
                    * POKEMON_ATTACK_BOOST * s1_phys;
                score += get_boost_multiplier(state.side_one.defense_boost) * POKEMON_DEFENSE_BOOST;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s1_spec;
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
                if !base {
                    score -= evaluate_tera_active(pkmn);
                }

                score -= get_boost_multiplier(state.side_two.attack_boost)
                    * POKEMON_ATTACK_BOOST * s2_phys;
                score -= get_boost_multiplier(state.side_two.defense_boost) * POKEMON_DEFENSE_BOOST;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s2_spec;
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

    // everything below here is fork-added: upstream's evaluate() ends at the
    // side-condition block above
    if base {
        return score;
    }

    score += evaluate_pending_effects(&state.side_one);
    score -= evaluate_pending_effects(&state.side_two);

    let trick_room = state.trick_room.active;
    let weather = state.weather.weather_type;
    if weather != Weather::NONE {
        let s1_active = state.side_one.get_active_immutable();
        let s2_active = state.side_two.get_active_immutable();
        if s1_active.hp > 0 {
            score += evaluate_weather_for_active(s1_active, weather, trick_room);
        }
        if s2_active.hp > 0 {
            score -= evaluate_weather_for_active(s2_active, weather, trick_room);
        }
    }

    let terrain = state.terrain.terrain_type;
    if terrain != Terrain::NONE {
        let s1_active = state.side_one.get_active_immutable();
        let s2_active = state.side_two.get_active_immutable();
        if s1_active.hp > 0 {
            score += evaluate_terrain_for_active(s1_active, terrain, trick_room);
        }
        if s2_active.hp > 0 {
            score -= evaluate_terrain_for_active(s2_active, terrain, trick_room);
        }
    }

    // Hopeless matchup: an active that can't damage the opponent at all AND has no status
    // moves is dead weight. Defensive walls with Roost/setup/hazards/phaze aren't hopeless —
    // they still have work to do even when their attacks register 0x.
    if s1_active.hp > 0 && s1_phys == 0.0 && s1_spec == 0.0 && !s1_has_status {
        score += HOPELESS_MATCHUP;
    }
    if s2_active.hp > 0 && s2_phys == 0.0 && s2_spec == 0.0 && !s2_has_status {
        score -= HOPELESS_MATCHUP;
    }

    // Speed-tier: outspeeding only matters if you can land a hit. Trick Room reverses
    // the comparison.
    if s1_active.hp > 0 && s2_active.hp > 0 {
        let trick_room = state.trick_room.active;
        let s1_faster = if trick_room {
            s1_active.speed < s2_active.speed
        } else {
            s1_active.speed > s2_active.speed
        };
        let s2_faster = if trick_room {
            s2_active.speed < s1_active.speed
        } else {
            s2_active.speed > s1_active.speed
        };
        let s1_max = s1_phys.max(s1_spec);
        let s2_max = s2_phys.max(s2_spec);
        // Speed only matters when defender is weakened — outspeeding a 100% HP
        // tank you can't break is worthless (Barraskewda vs full-HP Cresselia).
        let s1_def_missing = 1.0 - (s1_active.hp as f32 / s1_active.maxhp as f32);
        let s2_def_missing = 1.0 - (s2_active.hp as f32 / s2_active.maxhp as f32);
        if s1_faster && s1_max > 0.0 {
            score += SPEED_TIER_BONUS * s1_max * s2_def_missing;
        } else if s2_faster && s2_max > 0.0 {
            score -= SPEED_TIER_BONUS * s2_max * s1_def_missing;
        }
    }

    score
}
