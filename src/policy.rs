// ============================================================
// POLICY NET: v2 move-aware ONNX inference for MCTS priors
// ============================================================
// 579-dim features: active_moves(120) + pokemon(444) + sides(10) + global(5)
//
// Usage:
//   let policy = PolicyNet::load("policy_net.onnx")?;
//   let (s1_priors, s2_priors) = policy.get_priors(&state, &s1_options, &s2_options);

use crate::choices::{Choices, Effect, MoveCategory, MOVES};
use crate::engine::items::Items;
use crate::engine::state::{MoveChoice, Weather};
use crate::state::{Move, Pokemon, PokemonSideCondition, PokemonStatus, PokemonType, Side, State};
use crate::engine::state::PokemonVolatileStatus;
use ort::session::Session;
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

// ==================== Constants ====================

const MOVE_FEATURES: usize = 31;       // per move slot
const POKEMON_FEATURES: usize = 38;    // per pokemon
const SIDE_EXTRAS: usize = 12;         // per side
const GLOBAL_FEATURES: usize = 5;      // weather + speed_cmp + priority

// total: 4*31 + 6*38*2 + 12*2 + 5 = 124 + 456 + 24 + 5 = 609
pub const STATE_FEATURES: usize =
    4 * MOVE_FEATURES + 6 * POKEMON_FEATURES * 2 + SIDE_EXTRAS * 2 + GLOBAL_FEATURES;
pub const N_ACTIONS: usize = 9;

// 17 gen2 types (matches Python TYPES_V2 order; Steel was added in gen2)
const TYPE_ORDER: [PokemonType; 17] = [
    PokemonType::NORMAL, PokemonType::FIRE, PokemonType::WATER, PokemonType::ELECTRIC,
    PokemonType::GRASS, PokemonType::ICE, PokemonType::FIGHTING, PokemonType::POISON,
    PokemonType::GROUND, PokemonType::FLYING, PokemonType::PSYCHIC, PokemonType::BUG,
    PokemonType::ROCK, PokemonType::GHOST, PokemonType::DRAGON, PokemonType::DARK,
    PokemonType::STEEL,
];

const N_TYPES: usize = 17;

// gen2 physical types (category determined by type, not move)
fn is_physical_type(t: &PokemonType) -> bool {
    matches!(t,
        PokemonType::NORMAL | PokemonType::FIGHTING | PokemonType::FLYING |
        PokemonType::POISON | PokemonType::GROUND | PokemonType::ROCK |
        PokemonType::BUG | PokemonType::GHOST | PokemonType::STEEL
    )
}

// gen2 type chart [attacker][defender] (17x17, no fairy/stellar)
#[rustfmt::skip]
const TYPE_CHART: [[f32; 17]; 17] = [
    //       NOR  FIR  WAT  ELE  GRA  ICE  FIG  POI  GRO  FLY  PSY  BUG  ROC  GHO  DRA  DAR  STE
    /*NOR*/[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5],
    /*FIR*/[1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0],
    /*WAT*/[1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0],
    /*ELE*/[1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0],
    /*GRA*/[1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5],
    /*ICE*/[1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],
    /*FIG*/[2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0],
    /*POI*/[1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0],
    /*GRO*/[1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0],
    /*FLY*/[1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5],
    /*PSY*/[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5],
    /*BUG*/[1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5],
    /*ROC*/[1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5],
    /*GHO*/[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    /*DRA*/[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5],
    /*DAR*/[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],
    /*STE*/[1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5],
];

fn type_index(t: &PokemonType) -> Option<usize> {
    TYPE_ORDER.iter().position(|x| x == t)
}

fn type_effectiveness(atk: &PokemonType, def1: &PokemonType, def2: &PokemonType) -> f32 {
    let ai = match type_index(atk) { Some(i) => i, None => return 1.0 };
    let mut mult = 1.0;
    if let Some(d1) = type_index(def1) { mult *= TYPE_CHART[ai][d1]; }
    if let Some(d2) = type_index(def2) {
        if d2 != type_index(def1).unwrap_or(99) { mult *= TYPE_CHART[ai][d2]; }
    }
    mult
}

// status ordering
const STATUS_ORDER: [PokemonStatus; 7] = [
    PokemonStatus::NONE, PokemonStatus::BURN, PokemonStatus::SLEEP,
    PokemonStatus::FREEZE, PokemonStatus::PARALYZE, PokemonStatus::POISON,
    PokemonStatus::TOXIC,
];

// item ordering (gen2-tuned for v2 feature extractor; only used by gen2 value net)
#[cfg(feature = "gen2")]
const ITEM_ORDER_V2: [Items; 5] = [
    Items::LEFTOVERS, Items::THICKCLUB, Items::LIGHTBALL,
    Items::MIRACLEBERRY, Items::MINTBERRY,
];

// placeholder slots for non-gen2 builds — v2 features aren't trained for modern gens,
// but the feature extractor still has to compile. A gen9 extractor would be a separate v3.
#[cfg(not(feature = "gen2"))]
const ITEM_ORDER_V2: [Items; 5] = [
    Items::LEFTOVERS, Items::CHOICEBAND, Items::CHOICESCARF,
    Items::CHOICESPECS, Items::LIFEORB,
];

// ==================== Boost Calculation ====================

fn apply_boost(stat: i16, stage: i8) -> f32 {
    let s = stat as f32;
    if stage >= 0 {
        s * (2.0 + stage as f32) / 2.0
    } else {
        s * 2.0 / (2.0 - stage as f32)
    }
}

// ==================== Feature Extraction ====================

fn move_features(
    m: &Move,
    user_types: &(PokemonType, PokemonType),
    opp_types: &(PokemonType, PokemonType),
    out: &mut [f32],
) {
    let mut i = 0;

    // look up move choice data from MOVES table
    let choice = MOVES.get(&m.id);

    let (move_type, base_power, accuracy, is_status) = if let Some(c) = choice {
        (
            c.move_type,
            c.base_power,
            c.accuracy,
            c.category == MoveCategory::Status,
        )
    } else {
        (PokemonType::NORMAL, 0.0, 100.0, true)
    };

    // type one-hot (16)
    for t in &TYPE_ORDER {
        out[i] = if move_type == *t { 1.0 } else { 0.0 };
        i += 1;
    }

    // power / 250
    out[i] = base_power / 250.0;
    i += 1;

    // accuracy / 100
    out[i] = accuracy / 100.0;
    i += 1;

    // physical flag
    out[i] = if !is_status && is_physical_type(&move_type) { 1.0 } else { 0.0 };
    i += 1;

    // status flag
    out[i] = if is_status { 1.0 } else { 0.0 };
    i += 1;

    // STAB
    out[i] = if move_type == user_types.0 || move_type == user_types.1 { 1.0 } else { 0.0 };
    i += 1;

    // PP fraction
    out[i] = if m.id != Choices::NONE { (m.pp as f32 / 32.0).min(1.0) } else { 0.0 };
    i += 1;

    // effectiveness vs opponent / 4
    out[i] = type_effectiveness(&move_type, &opp_types.0, &opp_types.1) / 4.0;
    i += 1;

    // effect flags (7): sleeps, paralyzes, poisons, burns, boosts_user, heals_user, phazes
    if let Some(c) = choice {
        // sleeps
        if let Some(ref status) = c.status {
            if status.status == PokemonStatus::SLEEP { out[i] = 1.0; }
            // paralyzes
            if status.status == PokemonStatus::PARALYZE { out[i + 1] = 1.0; }
            // poisons
            if status.status == PokemonStatus::POISON || status.status == PokemonStatus::TOXIC {
                out[i + 2] = 1.0;
            }
            // burns
            if status.status == PokemonStatus::BURN { out[i + 3] = 1.0; }
        }
        // check secondaries for status effects too
        if let Some(ref secs) = c.secondaries {
            for sec in secs {
                if let Effect::Status(ref status) = sec.effect {
                    match status {
                        PokemonStatus::PARALYZE => out[i + 1] = 1.0,
                        PokemonStatus::POISON | PokemonStatus::TOXIC => out[i + 2] = 1.0,
                        PokemonStatus::BURN => out[i + 3] = 1.0,
                        _ => {}
                    }
                }
            }
        }
        // boosts_user
        if c.boost.is_some() { out[i + 4] = 1.0; }
        // heals_user
        if c.heal.is_some() || c.drain.is_some() ||
           m.id == Choices::REST || m.id == Choices::RECOVER ||
           m.id == Choices::MORNINGSUN || m.id == Choices::SOFTBOILED ||
           m.id == Choices::MOONLIGHT {
            out[i + 5] = 1.0;
        }
        // phazes (roar/whirlwind)
        if m.id == Choices::ROAR || m.id == Choices::WHIRLWIND {
            out[i + 6] = 1.0;
        }
    }
}

fn pokemon_features_v2(
    pkmn: &Pokemon,
    boosts: Option<[i8; 5]>,
    out: &mut [f32],
) {
    let mut i = 0;

    // hp fraction
    out[i] = pkmn.hp as f32 / pkmn.maxhp.max(1) as f32;
    i += 1;

    // alive
    out[i] = if pkmn.hp > 0 { 1.0 } else { 0.0 };
    i += 1;

    // types (16)
    for t in &TYPE_ORDER {
        out[i] = if pkmn.types.0 == *t || pkmn.types.1 == *t { 1.0 } else { 0.0 };
        i += 1;
    }

    // stats / 500 (with boosts for active)
    let stats = [pkmn.attack, pkmn.defense, pkmn.special_attack,
                 pkmn.special_defense, pkmn.speed];
    if let Some(b) = boosts {
        for j in 0..5 {
            out[i] = apply_boost(stats[j], b[j]) / 500.0;
            i += 1;
        }
    } else {
        for j in 0..5 {
            out[i] = stats[j] as f32 / 500.0;
            i += 1;
        }
    }

    // status (7)
    for s in &STATUS_ORDER {
        out[i] = if pkmn.status == *s { 1.0 } else { 0.0 };
        i += 1;
    }

    // item (7): leftovers, thickclub, lightball, miracleberry, mintberry, other, none
    let mut matched = false;
    for item in &ITEM_ORDER_V2 {
        out[i] = if pkmn.item == *item { matched = true; 1.0 } else { 0.0 };
        i += 1;
    }
    // other
    out[i] = if !matched && pkmn.item != Items::NONE { 1.0 } else { 0.0 };
    i += 1;
    // none
    out[i] = if !matched && pkmn.item == Items::NONE { 1.0 } else { 0.0 };
}

/// Extract 579-dim v2 feature vector from a State.
pub fn extract_features(state: &State) -> Vec<f32> {
    let mut features = vec![0.0f32; STATE_FEATURES];
    let mut idx = 0;

    let s1 = &state.side_one;
    let s2 = &state.side_two;
    let s1_active = &s1.pokemon.pkmn[s1.active_index as usize];
    let s2_active = &s2.pokemon.pkmn[s2.active_index as usize];

    // ---- my active moves (4 x 30 = 120) ----
    let moves = [&s1_active.moves.m0, &s1_active.moves.m1,
                 &s1_active.moves.m2, &s1_active.moves.m3];
    for m in &moves {
        move_features(m, &s1_active.types, &s2_active.types,
                      &mut features[idx..idx + MOVE_FEATURES]);
        idx += MOVE_FEATURES;
    }

    // ---- my team (6 x 37 = 222) ----
    let s1_boosts = [s1.attack_boost, s1.defense_boost, s1.special_attack_boost,
                     s1.special_defense_boost, s1.speed_boost];
    for p in 0..6 {
        let is_active = p == s1.active_index as usize;
        let boosts = if is_active { Some(s1_boosts) } else { None };
        pokemon_features_v2(&s1.pokemon.pkmn[p], boosts,
                           &mut features[idx..idx + POKEMON_FEATURES]);
        idx += POKEMON_FEATURES;
    }

    // ---- opponent team (6 x 37 = 222) ----
    let s2_boosts = [s2.attack_boost, s2.defense_boost, s2.special_attack_boost,
                     s2.special_defense_boost, s2.speed_boost];
    for p in 0..6 {
        let is_active = p == s2.active_index as usize;
        let boosts = if is_active { Some(s2_boosts) } else { None };
        pokemon_features_v2(&s2.pokemon.pkmn[p], boosts,
                           &mut features[idx..idx + POKEMON_FEATURES]);
        idx += POKEMON_FEATURES;
    }

    fn write_side_extras(
        s_self: &Side, s_opp: &Side, self_active: &Pokemon,
        out: &mut [f32],
    ) {
        out[0] = s_self.side_conditions.spikes as f32 / 3.0;
        out[1] = if s_self.side_conditions.reflect > 0 { 1.0 } else { 0.0 };
        out[2] = if s_self.side_conditions.light_screen > 0 { 1.0 } else { 0.0 };
        // has sleeping target on opp side
        out[3] = if (0..6).any(|p| s_opp.pokemon.pkmn[p].status == PokemonStatus::SLEEP) {
            1.0
        } else { 0.0 };
        // num alive / 6
        out[4] = (0..6).filter(|&p| s_self.pokemon.pkmn[p].hp > 0).count() as f32 / 6.0;
        // active volatile statuses (5 binary bits)
        out[5] = if s_self.volatile_statuses.contains(&PokemonVolatileStatus::SUBSTITUTE) { 1.0 } else { 0.0 };
        out[6] = if s_self.volatile_statuses.contains(&PokemonVolatileStatus::ENCORE) { 1.0 } else { 0.0 };
        out[7] = if s_self.volatile_statuses.contains(&PokemonVolatileStatus::DISABLE) { 1.0 } else { 0.0 };
        out[8] = if s_self.volatile_statuses.contains(&PokemonVolatileStatus::MUSTRECHARGE) { 1.0 } else { 0.0 };
        out[9] = if s_self.volatile_statuses.contains(&PokemonVolatileStatus::PARTIALLYTRAPPED) { 1.0 } else { 0.0 };
        // active sleep_turns / 7 and rest_turns / 2
        out[10] = (self_active.sleep_turns as f32 / 7.0).clamp(0.0, 1.0);
        out[11] = (self_active.rest_turns as f32 / 2.0).clamp(0.0, 1.0);
    }

    // ---- side 1 extras (12) ----
    write_side_extras(s1, s2, s1_active, &mut features[idx..idx + SIDE_EXTRAS]);
    idx += SIDE_EXTRAS;

    // ---- side 2 extras (12) ----
    write_side_extras(s2, s1, s2_active, &mut features[idx..idx + SIDE_EXTRAS]);
    idx += SIDE_EXTRAS;

    // ---- global (5) ----
    // weather (3)
    match state.weather.weather_type {
        Weather::SUN | Weather::HARSHSUN => features[idx] = 1.0,
        Weather::RAIN | Weather::HEAVYRAIN => features[idx + 1] = 1.0,
        Weather::SAND => features[idx + 2] = 1.0,
        _ => {}
    }
    idx += 3;

    // speed comparison
    let my_speed = apply_boost(s1_active.speed, s1.speed_boost);
    let opp_speed = apply_boost(s2_active.speed, s2.speed_boost);
    features[idx] = if my_speed > opp_speed { 1.0 } else { 0.0 };
    idx += 1;

    // priority placeholder
    features[idx] = 0.0;

    features
}

// ==================== ONNX Inference ====================

pub struct PolicyNet {
    session: Mutex<Session>,
    /// Temperature for softmax. Higher = softer priors. 1.0 = no change.
    pub temperature: f32,
}

unsafe impl Sync for PolicyNet {}

impl PolicyNet {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(PolicyNet {
            session: Mutex::new(session),
            temperature: 1.0,
        })
    }

    pub fn with_temperature<P: AsRef<Path>>(path: P, temperature: f32) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(PolicyNet {
            session: Mutex::new(session),
            temperature,
        })
    }

    fn predict(&self, features: &[f32]) -> Vec<f32> {
        let input =
            ort::value::Tensor::from_array(([1usize, STATE_FEATURES], features.to_vec()))
                .expect("failed to create input tensor");

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![input]).expect("inference failed");
        let binding = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("failed to extract output tensor");

        let logits: Vec<f32> = binding.1.to_vec();
        softmax_with_temp(&logits, self.temperature)
    }

    pub fn get_priors(
        &self,
        state: &State,
        s1_options: &[MoveChoice],
        s2_options: &[MoveChoice],
    ) -> (Vec<f32>, Vec<f32>) {
        let features = extract_features(state);
        let probs = self.predict(&features);

        let s1_priors = map_priors_to_options(&probs, s1_options, &state.side_one);
        let s2_priors = map_priors_to_options(&probs, s2_options, &state.side_two);

        (s1_priors, s2_priors)
    }
}

// ==================== V3 Feature Extractor (gen9-aware, 2738 dim) ====================
// Mirrors showdown/features_v3.py field-for-field. Layout:
//   per-mon (223): hp_frac + alive + cur_types(20) + base_types(20) + tera_type(20)
//                  + terad + stats(5) + status(7) + sleep_turns/4 + abil_flags(8)
//                  + item_flags(11) + pp(4) + move_feats(4 * 31)
//   side extras (26): active_idx(6) + boosts(7) + hazards(9) + force_flags(4)
//   global (10): weather(5) + terrain(4) + trick_room
//   total per side: 6*223 + 26 = 1364
//   total: 2*1364 + 10 = 2738
const POKEMON_V3_FEATURES: usize = 223;
const SIDE_V3_EXTRAS: usize = 26;
const GLOBAL_V3_FEATURES: usize = 10;
pub const STATE_V3_FEATURES: usize = 6 * POKEMON_V3_FEATURES * 2 + SIDE_V3_EXTRAS * 2 + GLOBAL_V3_FEATURES;

// Type ordering matches showdown/features_v3.py TYPES_V3 (20 entries).
const TYPE_ORDER_V3: [PokemonType; 20] = [
    PokemonType::NORMAL, PokemonType::FIRE, PokemonType::WATER, PokemonType::ELECTRIC,
    PokemonType::GRASS, PokemonType::ICE, PokemonType::FIGHTING, PokemonType::POISON,
    PokemonType::GROUND, PokemonType::FLYING, PokemonType::PSYCHIC, PokemonType::BUG,
    PokemonType::ROCK, PokemonType::GHOST, PokemonType::DRAGON, PokemonType::DARK,
    PokemonType::STEEL, PokemonType::FAIRY, PokemonType::STELLAR, PokemonType::TYPELESS,
];
const N_TYPES_V3: usize = 20;

fn write_type_multihot(t1: &PokemonType, t2: &PokemonType, out: &mut [f32]) {
    for (i, t) in TYPE_ORDER_V3.iter().enumerate() {
        out[i] = if t1 == t || t2 == t { 1.0 } else { 0.0 };
    }
}

fn write_type_onehot(t: &PokemonType, out: &mut [f32]) {
    for (i, candidate) in TYPE_ORDER_V3.iter().enumerate() {
        out[i] = if candidate == t { 1.0 } else { 0.0 };
    }
}

#[allow(dead_code)]  // some abilities only matter post-gen3 build
fn ability_flags_v3(a: &crate::engine::abilities::Abilities, out: &mut [f32]) {
    use crate::engine::abilities::Abilities as A;
    out[0] = matches!(a, A::MULTISCALE | A::SHADOWSHIELD) as i32 as f32;
    out[1] = matches!(a, A::PROTOSYNTHESIS | A::QUARKDRIVE) as i32 as f32;
    out[2] = matches!(a, A::REGENERATOR) as i32 as f32;
    out[3] = matches!(a, A::MAGICGUARD | A::LEVITATE) as i32 as f32;
    out[4] = matches!(a, A::INTIMIDATE) as i32 as f32;
    out[5] = matches!(a, A::UNAWARE) as i32 as f32;
    out[6] = matches!(a, A::MAGICBOUNCE | A::GOODASGOLD | A::PURIFYINGSALT) as i32 as f32;
    out[7] = matches!(a, A::SUPREMEOVERLORD) as i32 as f32;
}

#[allow(dead_code)]
fn item_flags_v3(i: &Items, out: &mut [f32]) {
    out[0] = matches!(i, Items::HEAVYDUTYBOOTS) as i32 as f32;
    let is_choice = matches!(i, Items::CHOICEBAND | Items::CHOICESPECS | Items::CHOICESCARF);
    out[1] = is_choice as i32 as f32;
    out[2] = matches!(i, Items::CHOICESCARF) as i32 as f32;  // scarf-specific (speed)
    out[3] = matches!(i, Items::LIFEORB) as i32 as f32;
    out[4] = matches!(i, Items::FOCUSSASH) as i32 as f32;
    out[5] = matches!(i, Items::AIRBALLOON) as i32 as f32;
    out[6] = matches!(i, Items::LEFTOVERS | Items::BLACKSLUDGE | Items::SITRUSBERRY) as i32 as f32;
    out[7] = matches!(i, Items::ROCKYHELMET) as i32 as f32;
    out[8] = matches!(i, Items::BOOSTERENERGY) as i32 as f32;
    out[9] = 0.0;  // weather-rocks slot reserved (variants not in Items enum yet)
    out[10] = matches!(i, Items::TOXICORB | Items::FLAMEORB) as i32 as f32;
}

fn status_onehot_v3(s: &PokemonStatus, out: &mut [f32]) {
    // matches Python STATUSES order: NONE, BURN, SLEEP, FREEZE, PARALYZE, POISON, TOXIC
    out[0] = (*s == PokemonStatus::NONE) as i32 as f32;
    out[1] = (*s == PokemonStatus::BURN) as i32 as f32;
    out[2] = (*s == PokemonStatus::SLEEP) as i32 as f32;
    out[3] = (*s == PokemonStatus::FREEZE) as i32 as f32;
    out[4] = (*s == PokemonStatus::PARALYZE) as i32 as f32;
    out[5] = (*s == PokemonStatus::POISON) as i32 as f32;
    out[6] = (*s == PokemonStatus::TOXIC) as i32 as f32;
}

fn pokemon_features_v3(pkmn: &Pokemon, out: &mut [f32]) {
    // hp_frac, alive
    out[0] = pkmn.hp as f32 / pkmn.maxhp.max(1) as f32;
    out[1] = if pkmn.hp > 0 { 1.0 } else { 0.0 };
    let mut idx = 2;
    // current types (multi-hot 20)
    write_type_multihot(&pkmn.types.0, &pkmn.types.1, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    // base types (multi-hot 20)
    write_type_multihot(&pkmn.base_types.0, &pkmn.base_types.1, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    // tera_type (one-hot 20)
    write_type_onehot(&pkmn.tera_type, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    // terastallized
    out[idx] = if pkmn.terastallized { 1.0 } else { 0.0 };
    idx += 1;
    // stats (atk/def/spa/spd/spe), normalized by 500
    out[idx] = pkmn.attack as f32 / 500.0; idx += 1;
    out[idx] = pkmn.defense as f32 / 500.0; idx += 1;
    out[idx] = pkmn.special_attack as f32 / 500.0; idx += 1;
    out[idx] = pkmn.special_defense as f32 / 500.0; idx += 1;
    out[idx] = pkmn.speed as f32 / 500.0; idx += 1;
    // status one-hot (7)
    status_onehot_v3(&pkmn.status, &mut out[idx..idx + 7]);
    idx += 7;
    // sleep_turns / 4 (clamped)
    out[idx] = (pkmn.sleep_turns as f32 / 4.0).clamp(0.0, 1.0);
    idx += 1;
    // ability flags (8)
    ability_flags_v3(&pkmn.ability, &mut out[idx..idx + 8]);
    idx += 8;
    // item flags (11)
    item_flags_v3(&pkmn.item, &mut out[idx..idx + 11]);
    idx += 11;
    // PP fractions for 4 moves (pp / 32 clamped)
    let moves = [&pkmn.moves.m0, &pkmn.moves.m1, &pkmn.moves.m2, &pkmn.moves.m3];
    for m in &moves {
        out[idx] = (m.pp as f32 / 32.0).clamp(0.0, 1.0);
        idx += 1;
    }
    // Move-identity features (4 × 31 = 124).
    // Layout matches Python features_v3._parse_pokemon: PP fractions first for
    // all 4 slots, then move-feature blocks for all 4 slots in the same order.
    for m in &moves {
        write_move_features_v3(m.id, &mut out[idx..idx + N_MOVE_FEATURES]);
        idx += N_MOVE_FEATURES;
    }
    debug_assert_eq!(idx, POKEMON_V3_FEATURES);
}

fn write_side_extras_v3(side: &Side, out: &mut [f32]) {
    // active_idx one-hot (6)
    let ai = side.active_index as usize;
    if ai < 6 {
        out[ai] = 1.0;
    }
    // boosts (7): atk, def, spa, spd, spe, acc, eva — clamped to [-1, 1]
    let stages = [
        side.attack_boost, side.defense_boost,
        side.special_attack_boost, side.special_defense_boost,
        side.speed_boost, side.accuracy_boost, side.evasion_boost,
    ];
    for (i, s) in stages.iter().enumerate() {
        out[6 + i] = (*s as f32 / 6.0).clamp(-1.0, 1.0);
    }
    // hazards (9)
    let sc = &side.side_conditions;
    out[13] = (sc.spikes as f32 / 3.0).clamp(0.0, 1.0);
    out[14] = if sc.stealth_rock > 0 { 1.0 } else { 0.0 };
    out[15] = if sc.sticky_web > 0 { 1.0 } else { 0.0 };
    out[16] = (sc.toxic_spikes as f32 / 2.0).clamp(0.0, 1.0);
    out[17] = if sc.reflect > 0 { 1.0 } else { 0.0 };
    out[18] = if sc.light_screen > 0 { 1.0 } else { 0.0 };
    out[19] = if sc.aurora_veil > 0 { 1.0 } else { 0.0 };
    out[20] = if sc.tailwind > 0 { 1.0 } else { 0.0 };
    out[21] = if sc.safeguard > 0 { 1.0 } else { 0.0 };
    // force flags (4): force_switch, force_trapped, slow_uturn_move, baton_passing
    out[22] = if side.force_switch { 1.0 } else { 0.0 };
    out[23] = if side.force_trapped { 1.0 } else { 0.0 };
    out[24] = if side.slow_uturn_move { 1.0 } else { 0.0 };
    out[25] = if side.baton_passing { 1.0 } else { 0.0 };
}

// ==================== Move feature extraction (v3) ====================
//
// 31-dim per-move encoding mirrored 1:1 in Python (showdown/move_db_v3.py).
// Sourced directly from poke-engine's MOVES table — no manual table to maintain.
//
// Layout: type one-hot (18) + bp/200 + accuracy + is_status + is_special
//       + 9 function flags (pivot, setup, recovery, status_inflict,
//         priority_attack, hazard, hazard_remove, screen, phase)

pub const N_MOVE_FEATURES: usize = 31;

// Attacking types in the order Python expects (drop Stellar/Typeless — moves
// can't be those; tera state is encoded separately at the pokemon level).
const ATK_TYPES_V3: [PokemonType; 18] = [
    PokemonType::NORMAL, PokemonType::FIRE, PokemonType::WATER, PokemonType::ELECTRIC,
    PokemonType::GRASS, PokemonType::ICE, PokemonType::FIGHTING, PokemonType::POISON,
    PokemonType::GROUND, PokemonType::FLYING, PokemonType::PSYCHIC, PokemonType::BUG,
    PokemonType::ROCK, PokemonType::GHOST, PokemonType::DRAGON, PokemonType::DARK,
    PokemonType::STEEL, PokemonType::FAIRY,
];

fn atk_type_index(t: &PokemonType) -> Option<usize> {
    ATK_TYPES_V3.iter().position(|x| x == t)
}

/// Heuristic: is this move a hazard remover? No flag for it on Choice; small list.
fn is_hazard_remover(c: &Choices) -> bool {
    matches!(c,
        Choices::RAPIDSPIN
        | Choices::DEFOG
        | Choices::TIDYUP
        | Choices::COURTCHANGE
        | Choices::MORTALSPIN
    )
}

/// Write 31-dim move features into `out` for a known Choices enum value.
/// `out.len()` must be >= N_MOVE_FEATURES; first N_MOVE_FEATURES slots are
/// zeroed before writing. Unknown / NONE leaves all zeros.
fn write_move_features_v3(choice_id: Choices, out: &mut [f32]) {
    for slot in out.iter_mut().take(N_MOVE_FEATURES) {
        *slot = 0.0;
    }
    if choice_id == Choices::NONE {
        return;
    }
    let choice = match MOVES.get(&choice_id) {
        Some(c) => c,
        None => return,
    };

    // type one-hot (18)
    if let Some(idx) = atk_type_index(&choice.move_type) {
        out[idx] = 1.0;
    }
    let mut i = ATK_TYPES_V3.len();

    // bp / 200
    out[i] = (choice.base_power / 200.0).clamp(0.0, 1.0);
    i += 1;

    // accuracy: poke-engine stores accuracy as f32 in [0, 100]; 0 typically
    // means "not applicable" (status moves) — we want 1.0 for those by convention.
    let acc = if choice.accuracy <= 0.0 {
        1.0
    } else {
        (choice.accuracy / 100.0).clamp(0.0, 1.0)
    };
    out[i] = acc;
    i += 1;

    let is_status = matches!(choice.category, MoveCategory::Status);
    let is_special = matches!(choice.category, MoveCategory::Special);
    out[i] = if is_status { 1.0 } else { 0.0 };
    i += 1;
    out[i] = if is_special { 1.0 } else { 0.0 };
    i += 1;

    // flags region (9): pivot, setup, recovery, status_inflict, priority_attack,
    //                   hazard, hazard_remove, screen, phase
    out[i + 0] = if choice.flags.pivot { 1.0 } else { 0.0 };
    out[i + 1] = if is_status && choice.boost.is_some() { 1.0 } else { 0.0 };
    out[i + 2] = if choice.heal.is_some() { 1.0 } else { 0.0 };
    out[i + 3] = if is_status && choice.status.is_some() { 1.0 } else { 0.0 };
    out[i + 4] = if !is_status && choice.priority > 0 { 1.0 } else { 0.0 };

    let is_hazard = match &choice.side_condition {
        Some(sc) => matches!(sc.condition,
            PokemonSideCondition::Spikes
            | PokemonSideCondition::Stealthrock
            | PokemonSideCondition::StickyWeb
            | PokemonSideCondition::ToxicSpikes),
        None => false,
    };
    out[i + 5] = if is_hazard { 1.0 } else { 0.0 };
    out[i + 6] = if is_hazard_remover(&choice_id) { 1.0 } else { 0.0 };

    let is_screen = match &choice.side_condition {
        Some(sc) => matches!(sc.condition,
            PokemonSideCondition::Reflect
            | PokemonSideCondition::LightScreen
            | PokemonSideCondition::AuroraVeil),
        None => false,
    };
    out[i + 7] = if is_screen { 1.0 } else { 0.0 };
    out[i + 8] = if choice.flags.drag { 1.0 } else { 0.0 };
}

/// Returns 31-dim feature vector for a normalized move id (e.g. "earthquake").
/// Empty / "none" returns all zeros. Unknown moves use the default Choice
/// (NONE), which gives all-zero attributes — same effect as the Python "default
/// to neutral attacker" since the model just treats them as low-info.
///
/// Strips a trailing "-tera" suffix if present (Tera variants share base attrs).
pub fn extract_move_features_v3(move_id: &str) -> Vec<f32> {
    let mut out = vec![0.0f32; N_MOVE_FEATURES];
    let trimmed = move_id.strip_suffix("-tera").unwrap_or(move_id);
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("none") {
        return out;
    }
    let choice_id = match Choices::from_str(trimmed) {
        Ok(c) if c != Choices::NONE => c,
        _ => return out,
    };
    write_move_features_v3(choice_id, &mut out);
    out
}

/// Extract 1250-dim v3 feature vector from a State (gen9-aware).
/// Mirrors showdown/features_v3.py parse_state_v3 exactly:
///   [s1.mon0..mon5, s1.extras, s2.mon0..mon5, s2.extras, global]
pub fn extract_features_v3(state: &State) -> Vec<f32> {
    let mut features = vec![0.0f32; STATE_V3_FEATURES];
    let mut idx = 0;
    // side 1 mons + extras
    for p in 0..6 {
        pokemon_features_v3(&state.side_one.pokemon.pkmn[p],
                            &mut features[idx..idx + POKEMON_V3_FEATURES]);
        idx += POKEMON_V3_FEATURES;
    }
    write_side_extras_v3(&state.side_one, &mut features[idx..idx + SIDE_V3_EXTRAS]);
    idx += SIDE_V3_EXTRAS;
    // side 2 mons + extras
    for p in 0..6 {
        pokemon_features_v3(&state.side_two.pokemon.pkmn[p],
                            &mut features[idx..idx + POKEMON_V3_FEATURES]);
        idx += POKEMON_V3_FEATURES;
    }
    write_side_extras_v3(&state.side_two, &mut features[idx..idx + SIDE_V3_EXTRAS]);
    idx += SIDE_V3_EXTRAS;
    // global: weather (5), terrain (4), trick_room (1)
    // Python bit positions: SUN=0, RAIN=1, SAND=2, SNOW=3, HAIL=4,
    //                       ELECTRIC=5, GRASSY=6, MISTY=7, PSYCHIC=8, trick_room=9
    match state.weather.weather_type {
        Weather::SUN | Weather::HARSHSUN => features[idx + 0] = 1.0,
        Weather::RAIN | Weather::HEAVYRAIN => features[idx + 1] = 1.0,
        Weather::SAND => features[idx + 2] = 1.0,
        Weather::SNOW => features[idx + 3] = 1.0,
        Weather::HAIL => features[idx + 4] = 1.0,
        _ => {}
    }
    let terrain = state.get_terrain();
    use crate::engine::state::Terrain;
    match terrain {
        Terrain::ELECTRICTERRAIN => features[idx + 5] = 1.0,
        Terrain::GRASSYTERRAIN => features[idx + 6] = 1.0,
        Terrain::MISTYTERRAIN => features[idx + 7] = 1.0,
        Terrain::PSYCHICTERRAIN => features[idx + 8] = 1.0,
        _ => {}
    }
    if state.trick_room.active {
        features[idx + 9] = 1.0;
    }
    features
}

// ==================== Material-only fast path ====================
//
// The full v3 featurizer spends most of its time on the 124 move features per
// mon (24 hash lookups + flag extraction per call). The material model
// trained in showdown/material_train.py never reads those slots — its
// per-mon input is the first 95 dims of the v3 mon vector.
//
// MaterialNet:
//   - extracts only the 95 core dims per mon (no PP, no move features)
//   - runs a hand-rolled fp32 forward pass (no ONNX session overhead)
//   - reuses scratch buffers across calls so a 300ms MCTS doesn't churn
//     ~150k small Vec allocations
//
// Architecture mirrors MaterialValueNet in material_train.py exactly:
//   per-mon input  = [95 core, is_active_flag]  (96-dim)
//   mon_encoder    = Linear(96, mon_hidden) → ReLU
//                   → Linear(mon_hidden, mon_hidden) → ReLU
//                   → Linear(mon_hidden, d_emb)
//   side_pool      = [active_emb (d_emb), mean(bench_emb) (d_emb)]
//   head_input     = [s1_pool, s2_pool, s1_extras(26), s2_extras(26), glb(10)]
//   head           = Linear(head_in, hidden) → ReLU
//                   → Linear(hidden, hidden) → ReLU
//                   → Linear(hidden, 1)
//   v = sigmoid(head_out)

const N_MON_CORE: usize = 95;

/// Like `pokemon_features_v3`, but writes only the first 95 dims (skip PP +
/// per-move feature blocks). Identical bit-for-bit to the head of the v3
/// featurizer so the trained weights apply unchanged.
fn pokemon_features_v3_core(pkmn: &Pokemon, out: &mut [f32]) {
    out[0] = pkmn.hp as f32 / pkmn.maxhp.max(1) as f32;
    out[1] = if pkmn.hp > 0 { 1.0 } else { 0.0 };
    let mut idx = 2;
    write_type_multihot(&pkmn.types.0, &pkmn.types.1, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    write_type_multihot(&pkmn.base_types.0, &pkmn.base_types.1, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    write_type_onehot(&pkmn.tera_type, &mut out[idx..idx + N_TYPES_V3]);
    idx += N_TYPES_V3;
    out[idx] = if pkmn.terastallized { 1.0 } else { 0.0 };
    idx += 1;
    out[idx] = pkmn.attack as f32 / 500.0; idx += 1;
    out[idx] = pkmn.defense as f32 / 500.0; idx += 1;
    out[idx] = pkmn.special_attack as f32 / 500.0; idx += 1;
    out[idx] = pkmn.special_defense as f32 / 500.0; idx += 1;
    out[idx] = pkmn.speed as f32 / 500.0; idx += 1;
    status_onehot_v3(&pkmn.status, &mut out[idx..idx + 7]);
    idx += 7;
    out[idx] = (pkmn.sleep_turns as f32 / 4.0).clamp(0.0, 1.0);
    idx += 1;
    ability_flags_v3(&pkmn.ability, &mut out[idx..idx + 8]);
    idx += 8;
    item_flags_v3(&pkmn.item, &mut out[idx..idx + 11]);
    idx += 11;
    debug_assert_eq!(idx, N_MON_CORE);
}

/// Dot product with 8 independent accumulators. The 8-way unroll breaks the
/// fp-add dependency chain so the compiler can emit AVX2 vmulps/vfmadd ops
/// despite floating-point being non-associative — confirmed with target-cpu
/// native + opt-level 3 on x86_64.
#[inline(always)]
fn dot8(row: &[f32], input: &[f32]) -> f32 {
    let n = row.len();
    debug_assert_eq!(input.len(), n);
    let mut a = [0.0f32; 8];
    let mut j = 0;
    while j + 8 <= n {
        // Use raw indexing — the bounds-check elimination is reliable here
        // because the while loop guarantees j+7 < n.
        unsafe {
            a[0] += row.get_unchecked(j)     * input.get_unchecked(j);
            a[1] += row.get_unchecked(j + 1) * input.get_unchecked(j + 1);
            a[2] += row.get_unchecked(j + 2) * input.get_unchecked(j + 2);
            a[3] += row.get_unchecked(j + 3) * input.get_unchecked(j + 3);
            a[4] += row.get_unchecked(j + 4) * input.get_unchecked(j + 4);
            a[5] += row.get_unchecked(j + 5) * input.get_unchecked(j + 5);
            a[6] += row.get_unchecked(j + 6) * input.get_unchecked(j + 6);
            a[7] += row.get_unchecked(j + 7) * input.get_unchecked(j + 7);
        }
        j += 8;
    }
    let mut tail = 0.0f32;
    while j < n {
        tail += row[j] * input[j];
        j += 1;
    }
    ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7])) + tail
}

/// One Linear + ReLU pass. `weights` is row-major [out_dim, in_dim].
#[inline]
fn linear_relu(weights: &[f32], bias: &[f32], input: &[f32], out: &mut [f32]) {
    let out_dim = bias.len();
    let in_dim = input.len();
    debug_assert_eq!(weights.len(), out_dim * in_dim);
    debug_assert_eq!(out.len(), out_dim);
    for i in 0..out_dim {
        let row = &weights[i * in_dim..(i + 1) * in_dim];
        let acc = bias[i] + dot8(row, input);
        out[i] = if acc > 0.0 { acc } else { 0.0 };
    }
}

/// Same as `linear_relu` but without the activation (final layer).
#[inline]
fn linear(weights: &[f32], bias: &[f32], input: &[f32], out: &mut [f32]) {
    let out_dim = bias.len();
    let in_dim = input.len();
    for i in 0..out_dim {
        let row = &weights[i * in_dim..(i + 1) * in_dim];
        out[i] = bias[i] + dot8(row, input);
    }
}

struct MaterialScratch {
    mon_input: Vec<f32>,    // [96]
    enc_h1: Vec<f32>,       // [mon_hidden]
    enc_h2: Vec<f32>,       // [mon_hidden]
    mon_emb: Vec<f32>,      // [6 * d_emb] per side, reused
    head_in: Vec<f32>,      // [head_in_dim]
    head_h1: Vec<f32>,
    head_h2: Vec<f32>,
    head_out: Vec<f32>,     // [1]
}

pub struct MaterialNet {
    mon_hidden: usize,
    d_emb: usize,
    hidden: usize,
    /// Number of head outputs. 1 = V-net (scalar value, sigmoid → P(win)).
    /// >1 = Q-net (per-action values; we return max(sigmoid(Q[i])) at leaf,
    /// equivalent to "value of playing the best move from this state").
    n_outputs: usize,
    e0_w: Vec<f32>, e0_b: Vec<f32>,
    e2_w: Vec<f32>, e2_b: Vec<f32>,
    e4_w: Vec<f32>, e4_b: Vec<f32>,
    h0_w: Vec<f32>, h0_b: Vec<f32>,
    h2_w: Vec<f32>, h2_b: Vec<f32>,
    h4_w: Vec<f32>, h4_b: Vec<f32>,
    scratch: Mutex<MaterialScratch>,
}

impl MaterialNet {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        use std::io::{Read, Error, ErrorKind};
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"MAT1" {
            return Err(Error::new(ErrorKind::InvalidData,
                "material file magic mismatch (expected 'MAT1')"));
        }
        let read_u32 = |f: &mut std::fs::File| -> std::io::Result<u32> {
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf)?;
            Ok(u32::from_le_bytes(buf))
        };
        let mon_hidden = read_u32(&mut f)? as usize;
        let d_emb = read_u32(&mut f)? as usize;
        let hidden = read_u32(&mut f)? as usize;
        let n_mon_core = read_u32(&mut f)? as usize;
        let side_extras = read_u32(&mut f)? as usize;
        let n_global = read_u32(&mut f)? as usize;
        if n_mon_core != N_MON_CORE || side_extras != SIDE_V3_EXTRAS
            || n_global != GLOBAL_V3_FEATURES {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("material file dim mismatch: core={} extras={} glb={}",
                    n_mon_core, side_extras, n_global)));
        }
        let read_layer = |f: &mut std::fs::File| -> std::io::Result<(Vec<f32>, Vec<f32>)> {
            let out_dim = read_u32(f)? as usize;
            let in_dim = read_u32(f)? as usize;
            let mut wbuf = vec![0u8; out_dim * in_dim * 4];
            f.read_exact(&mut wbuf)?;
            let weights: Vec<f32> = wbuf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut bbuf = vec![0u8; out_dim * 4];
            f.read_exact(&mut bbuf)?;
            let bias: Vec<f32> = bbuf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok((weights, bias))
        };
        let (e0_w, e0_b) = read_layer(&mut f)?;
        let (e2_w, e2_b) = read_layer(&mut f)?;
        let (e4_w, e4_b) = read_layer(&mut f)?;
        let (h0_w, h0_b) = read_layer(&mut f)?;
        let (h2_w, h2_b) = read_layer(&mut f)?;
        let (h4_w, h4_b) = read_layer(&mut f)?;

        let head_in_dim = 4 * d_emb + 2 * SIDE_V3_EXTRAS + GLOBAL_V3_FEATURES;
        if h0_w.len() != hidden * head_in_dim {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("head input dim mismatch: got {} expected {}",
                    h0_w.len() / hidden, head_in_dim)));
        }

        // n_outputs is determined by the final layer's out_dim (1 for V,
        // 9 for Q-net via showdown/qnet_train.py).
        let n_outputs = h4_b.len();
        let scratch = MaterialScratch {
            mon_input: vec![0.0; N_MON_CORE + 1],
            enc_h1: vec![0.0; mon_hidden],
            enc_h2: vec![0.0; mon_hidden],
            mon_emb: vec![0.0; 6 * d_emb],
            head_in: vec![0.0; head_in_dim],
            head_h1: vec![0.0; hidden],
            head_h2: vec![0.0; hidden],
            head_out: vec![0.0; n_outputs],
        };
        Ok(MaterialNet {
            mon_hidden, d_emb, hidden, n_outputs,
            e0_w, e0_b, e2_w, e2_b, e4_w, e4_b,
            h0_w, h0_b, h2_w, h2_b, h4_w, h4_b,
            scratch: Mutex::new(scratch),
        })
    }

    /// Encode one 6-mon team into a 2*d_emb pool written to `pool_out`:
    /// `[active_emb (d_emb), mean(bench_emb) (d_emb)]`.
    /// `extras[0..6]` is the active one-hot. Takes the buffers as separate
    /// args so the caller can split-borrow `MaterialScratch` fields.
    fn encode_side(&self, side: &Side, extras: &[f32],
                   mon_input: &mut [f32], enc_h1: &mut [f32],
                   enc_h2: &mut [f32], mon_emb: &mut [f32],
                   pool_out: &mut [f32]) {
        debug_assert_eq!(pool_out.len(), 2 * self.d_emb);
        for p in 0..6 {
            pokemon_features_v3_core(&side.pokemon.pkmn[p], &mut mon_input[..N_MON_CORE]);
            mon_input[N_MON_CORE] = extras[p];
            linear_relu(&self.e0_w, &self.e0_b, mon_input, enc_h1);
            linear_relu(&self.e2_w, &self.e2_b, enc_h1, enc_h2);
            linear(&self.e4_w, &self.e4_b, enc_h2,
                   &mut mon_emb[p * self.d_emb..(p + 1) * self.d_emb]);
        }
        for slot in pool_out.iter_mut() { *slot = 0.0; }
        let mut bench_count: f32 = 0.0;
        for p in 0..6 {
            let is_active = extras[p];
            let off = p * self.d_emb;
            if is_active > 0.5 {
                pool_out[..self.d_emb].copy_from_slice(&mon_emb[off..off + self.d_emb]);
            } else {
                for k in 0..self.d_emb {
                    pool_out[self.d_emb + k] += mon_emb[off + k];
                }
                bench_count += 1.0;
            }
        }
        let denom = bench_count.max(1.0);
        for k in 0..self.d_emb {
            pool_out[self.d_emb + k] /= denom;
        }
    }

    /// Returns the raw post-sigmoid Q vector (length = n_outputs).
    /// For V-nets this is just [V] (length 1); for Q-nets it's the
    /// 9-dim per-action Q vector. Used when MCTS wants Q as priors.
    pub fn evaluate_vec(&self, state: &State) -> Vec<f32> {
        // Reuse evaluate() by replicating the forward pass but copying
        // out the full head_out before aggregation.
        let mut guard = self.scratch.lock().unwrap();
        let s: &mut MaterialScratch = &mut *guard;

        let mut s1_extras = [0.0f32; SIDE_V3_EXTRAS];
        let mut s2_extras = [0.0f32; SIDE_V3_EXTRAS];
        write_side_extras_v3(&state.side_one, &mut s1_extras);
        write_side_extras_v3(&state.side_two, &mut s2_extras);

        let MaterialScratch {
            mon_input, enc_h1, enc_h2, mon_emb, head_in, head_h1, head_h2, head_out, ..
        } = s;

        let d2 = 2 * self.d_emb;
        let (head_pools, rest) = head_in.split_at_mut(2 * d2);
        let (head_extras, head_glb) = rest.split_at_mut(2 * SIDE_V3_EXTRAS);
        let (s1_pool, s2_pool) = head_pools.split_at_mut(d2);
        self.encode_side(&state.side_one, &s1_extras,
                         mon_input, enc_h1, enc_h2, mon_emb, s1_pool);
        self.encode_side(&state.side_two, &s2_extras,
                         mon_input, enc_h1, enc_h2, mon_emb, s2_pool);
        head_extras[..SIDE_V3_EXTRAS].copy_from_slice(&s1_extras);
        head_extras[SIDE_V3_EXTRAS..].copy_from_slice(&s2_extras);
        for slot in head_glb.iter_mut() { *slot = 0.0; }
        match state.weather.weather_type {
            Weather::SUN | Weather::HARSHSUN => head_glb[0] = 1.0,
            Weather::RAIN | Weather::HEAVYRAIN => head_glb[1] = 1.0,
            Weather::SAND => head_glb[2] = 1.0,
            Weather::SNOW => head_glb[3] = 1.0,
            Weather::HAIL => head_glb[4] = 1.0,
            _ => {}
        }
        let terrain = state.get_terrain();
        use crate::engine::state::Terrain;
        match terrain {
            Terrain::ELECTRICTERRAIN => head_glb[5] = 1.0,
            Terrain::GRASSYTERRAIN => head_glb[6] = 1.0,
            Terrain::MISTYTERRAIN => head_glb[7] = 1.0,
            Terrain::PSYCHICTERRAIN => head_glb[8] = 1.0,
            _ => {}
        }
        if state.trick_room.active {
            head_glb[9] = 1.0;
        }
        linear_relu(&self.h0_w, &self.h0_b, head_in, head_h1);
        linear_relu(&self.h2_w, &self.h2_b, head_h1, head_h2);
        linear(&self.h4_w, &self.h4_b, head_h2, head_out);
        head_out.iter().map(|&z| 1.0 / (1.0 + (-z).exp())).collect()
    }

    /// Run the full forward pass for one state. Returns sigmoid(head_out).
    pub fn evaluate(&self, state: &State) -> f32 {
        let mut guard = self.scratch.lock().unwrap();
        // Deref MutexGuard once so field-level split borrows are visible to
        // the borrow checker (e.g. immutable &s.head_in alongside mutable
        // &mut s.head_h1 in the same call).
        let s: &mut MaterialScratch = &mut *guard;

        // Build the 26-dim extras vectors on the stack for both sides.
        let mut s1_extras = [0.0f32; SIDE_V3_EXTRAS];
        let mut s2_extras = [0.0f32; SIDE_V3_EXTRAS];
        write_side_extras_v3(&state.side_one, &mut s1_extras);
        write_side_extras_v3(&state.side_two, &mut s2_extras);

        // Destructure scratch so its fields can be borrowed independently
        // (avoids "borrow `*s` as mutable more than once" when we split
        // head_in below and still need mon_input/enc_h*/mon_emb for encoding).
        let MaterialScratch {
            mon_input, enc_h1, enc_h2, mon_emb, head_in, head_h1, head_h2, head_out, ..
        } = s;

        // Write each side's pool directly into head_in's prefix — saves
        // two Vec::clone allocations per call vs going through a scratch
        // side_pool buffer.
        let d2 = 2 * self.d_emb;
        let (head_pools, rest) = head_in.split_at_mut(2 * d2);
        let (head_extras, head_glb) = rest.split_at_mut(2 * SIDE_V3_EXTRAS);
        let (s1_pool, s2_pool) = head_pools.split_at_mut(d2);
        self.encode_side(&state.side_one, &s1_extras,
                         mon_input, enc_h1, enc_h2, mon_emb, s1_pool);
        self.encode_side(&state.side_two, &s2_extras,
                         mon_input, enc_h1, enc_h2, mon_emb, s2_pool);

        head_extras[..SIDE_V3_EXTRAS].copy_from_slice(&s1_extras);
        head_extras[SIDE_V3_EXTRAS..].copy_from_slice(&s2_extras);
        for slot in head_glb.iter_mut() { *slot = 0.0; }
        match state.weather.weather_type {
            Weather::SUN | Weather::HARSHSUN => head_glb[0] = 1.0,
            Weather::RAIN | Weather::HEAVYRAIN => head_glb[1] = 1.0,
            Weather::SAND => head_glb[2] = 1.0,
            Weather::SNOW => head_glb[3] = 1.0,
            Weather::HAIL => head_glb[4] = 1.0,
            _ => {}
        }
        let terrain = state.get_terrain();
        use crate::engine::state::Terrain;
        match terrain {
            Terrain::ELECTRICTERRAIN => head_glb[5] = 1.0,
            Terrain::GRASSYTERRAIN => head_glb[6] = 1.0,
            Terrain::MISTYTERRAIN => head_glb[7] = 1.0,
            Terrain::PSYCHICTERRAIN => head_glb[8] = 1.0,
            _ => {}
        }
        if state.trick_room.active {
            head_glb[9] = 1.0;
        }

        // head_in/h1/h2/out already destructured above.
        linear_relu(&self.h0_w, &self.h0_b, head_in, head_h1);
        linear_relu(&self.h2_w, &self.h2_b, head_h1, head_h2);
        linear(&self.h4_w, &self.h4_b, head_h2, head_out);
        if self.n_outputs == 1 {
            // V-net: scalar logit → sigmoid.
            1.0 / (1.0 + (-head_out[0]).exp())
        } else {
            // Q-net: per-action logits. Return MEAN sigmoid across all
            // N outputs — equivalent to a uniform-policy expected value.
            // Less optimistic than max(Q) (which is biased toward the
            // best-case continuation) and more robust to garbage Q for
            // rarely-played actions.
            let mut sum = 0.0f32;
            for &z in head_out.iter() {
                sum += 1.0 / (1.0 + (-z).exp());
            }
            sum / head_out.len() as f32
        }
    }
}

// ==================== BO-locked featurizer (184-dim) ====================
//
// Mirrors showdown/features_bo.py parse_state_bo. Used by value/policy nets
// trained exclusively against the Bulky Offense team (idx 1 in SAMPLE_TEAMS_GEN9).
// Auto-detects which side holds the BO mons (Garganacl/Darkrai/Great Tusk/
// Hatterene/Tornadus-Therian/Dragonite) and orients encoding BO-first.
//
// Layout: bo_side(30) + opp_side(126) + field(18) + matchup(10) = 184.

use crate::pokemon::PokemonName;

const BO_LIST: [PokemonName; 6] = [
    PokemonName::GARGANACL,
    PokemonName::DARKRAI,
    PokemonName::GREATTUSK,
    PokemonName::HATTERENE,
    PokemonName::TORNADUSTHERIAN,
    PokemonName::DRAGONITE,
];

fn bo_canonical_slot(name: PokemonName) -> Option<usize> {
    BO_LIST.iter().position(|&n| n == name)
}

fn is_bo_mon(name: PokemonName) -> bool {
    bo_canonical_slot(name).is_some()
}

// 10-category opp role table — must mirror OPP_ROLES order in features_bo.py.
const ROLE_FIRE_STEEL_WALL: usize = 0;
const ROLE_PRIORITY_BREAKER: usize = 1;
const ROLE_HAZARD_SETTER: usize = 4;
const ROLE_REMOVER: usize = 5;
const ROLE_PHYS_SWEEPER: usize = 6;
const ROLE_SPEC_SWEEPER: usize = 7;
const ROLE_WALLBREAKER: usize = 8;
const ROLE_WEATHER_SETTER: usize = 3;
const ROLE_OTHER: usize = 9;
const N_OPP_ROLES: usize = 10;

fn opp_role_of(name: PokemonName) -> usize {
    use PokemonName as N;
    match name {
        // fire-steel / generic walls
        N::HEATRAN | N::SKARMORY | N::CORVIKNIGHT | N::GHOLDENGO
        | N::CLODSIRE | N::TOXAPEX | N::BLISSEY | N::CLEFABLE | N::DONDOZO
            => ROLE_FIRE_STEEL_WALL,
        // hazard setters
        N::TINGLU | N::GLIMMORA | N::LANDORUSTHERIAN
            => ROLE_HAZARD_SETTER,
        // removers / spinners
        N::IRONTREADS
            => ROLE_REMOVER,
        // priority / phys priority breakers
        N::KINGAMBIT | N::MAMOSWINE | N::RILLABOOM | N::BISHARP
            => ROLE_PRIORITY_BREAKER,
        // weather setters
        N::PELIPPER | N::TORKOAL | N::TYRANITAR | N::NINETALES
            => ROLE_WEATHER_SETTER,
        // physical sweepers
        N::ROARINGMOON | N::DRAGONITE | N::VOLCARONA | N::BAXCALIBUR
        | N::BARRASKEWDA | N::EXCADRILL | N::GREATTUSK
            => ROLE_PHYS_SWEEPER,
        // special sweepers
        N::WALKINGWAKE | N::IRONVALIANT | N::IRONMOTH | N::RAGINGBOLT
        | N::MANAPHY | N::ARCHALUDON
            => ROLE_SPEC_SWEEPER,
        // wallbreakers
        N::DARKRAI | N::PECHARUNT | N::HOOPAUNBOUND | N::KYUREM | N::URSALUNA
            => ROLE_WALLBREAKER,
        _ => ROLE_OTHER,
    }
}

fn is_fire_steel_wall(name: PokemonName) -> bool {
    use PokemonName as N;
    matches!(name, N::HEATRAN | N::SKARMORY | N::CORVIKNIGHT | N::GHOLDENGO)
}

fn is_strong_priority_user(name: PokemonName) -> bool {
    use PokemonName as N;
    matches!(name,
        N::KINGAMBIT | N::MAMOSWINE | N::RILLABOOM | N::DRAGONITE
        | N::BISHARP | N::RAGINGBOLT
    )
}

fn is_trick_item_user(name: PokemonName) -> bool {
    use PokemonName as N;
    matches!(name,
        N::GHOLDENGO | N::IRONBOULDER | N::HOOPAUNBOUND
        | N::ALAKAZAM | N::GENGAR | N::LATIOS | N::LATIAS
    )
}

// 7-slot item table for opp active. Order mirrors features_bo.ITEM_FLAGS.
const N_ITEM_FLAGS_BO: usize = 7;
fn item_flag_idx(item: Items) -> usize {
    match item {
        Items::HEAVYDUTYBOOTS => 0,
        Items::CHOICESCARF    => 1,
        Items::CHOICESPECS    => 2,
        Items::CHOICEBAND     => 3,
        Items::LIFEORB        => 4,
        Items::BOOSTERENERGY  => 5,
        _                     => 6,
    }
}

// Block sizes (must match features_bo.py).
const N_BO_SIDE: usize = 6 * 3 + 6 + 5 + 1;                  // 30
const N_TYPES_BO: usize = 18;                                 // matches ATK_TYPES_V3
const N_OPP_SIDE: usize = 6 * 3 + 2 * N_TYPES_BO
    + 5 * N_OPP_ROLES + 6 + 5 + N_ITEM_FLAGS_BO + 4;          // 126
const N_FIELD_BO: usize = 5 + 4 + 1 + 6 + 2;                  // 18
const N_MATCHUP_BO: usize = 10;
pub const STATE_BO_FEATURES: usize =
    N_BO_SIDE + N_OPP_SIDE + N_FIELD_BO + N_MATCHUP_BO;       // 184

fn defensive_blocks_salt_cure(t1: &PokemonType, t2: &PokemonType) -> bool {
    matches!(t1, PokemonType::STEEL | PokemonType::GHOST)
        || matches!(t2, PokemonType::STEEL | PokemonType::GHOST)
}

fn bo_type_index_18(t: &PokemonType) -> Option<usize> {
    ATK_TYPES_V3.iter().position(|x| x == t)
}

fn count_bo_mons(side: &Side) -> usize {
    (0..6).filter(|&p| is_bo_mon(side.pokemon.pkmn[p].id)).count()
}

/// Returns 0 if side_one holds the BO team, 1 if side_two does. Ties (e.g.
/// mirror or no-match initial state) resolve to side_one.
fn detect_bo_side(state: &State) -> usize {
    let s1 = count_bo_mons(&state.side_one);
    let s2 = count_bo_mons(&state.side_two);
    if s2 > s1 { 1 } else { 0 }
}

fn encode_bo_side(side: &Side, out: &mut [f32]) {
    // 0..18 — per-mon (hp, status_any, alive) in canonical BO order.
    let mut tusk_alive = false;
    let mut tusk_item = Items::NONE;
    let mut slot_to_canon: [Option<usize>; 6] = [None; 6];
    for slot in 0..6 {
        let pk = &side.pokemon.pkmn[slot];
        if let Some(canon) = bo_canonical_slot(pk.id) {
            slot_to_canon[slot] = Some(canon);
            let base = canon * 3;
            out[base + 0] = pk.hp as f32 / pk.maxhp.max(1) as f32;
            out[base + 1] = if pk.status != PokemonStatus::NONE { 1.0 } else { 0.0 };
            out[base + 2] = if pk.hp > 0 { 1.0 } else { 0.0 };
            if pk.id == PokemonName::GREATTUSK {
                tusk_alive = pk.hp > 0;
                tusk_item = pk.item;
            }
        }
    }
    // 18..24 — active idx in canonical order.
    let active = side.active_index as usize;
    if active < 6 {
        if let Some(canon) = slot_to_canon[active] {
            out[18 + canon] = 1.0;
        }
    }
    // 24..29 — active boosts (atk/def/spa/spd/spe).
    let stages = [
        side.attack_boost, side.defense_boost,
        side.special_attack_boost, side.special_defense_boost,
        side.speed_boost,
    ];
    for (i, s) in stages.iter().enumerate() {
        out[24 + i] = (*s as f32 / 6.0).clamp(-1.0, 1.0);
    }
    // 29 — Tusk Proto online (alive + Booster consumed).
    out[29] = if tusk_alive && tusk_item != Items::BOOSTERENERGY { 1.0 } else { 0.0 };
}

fn encode_opp_side(side: &Side, out: &mut [f32]) {
    let active = side.active_index as usize;
    // 0..18 — per-mon hp/status/alive in slot order.
    for slot in 0..6 {
        let pk = &side.pokemon.pkmn[slot];
        let base = slot * 3;
        out[base + 0] = pk.hp as f32 / pk.maxhp.max(1) as f32;
        out[base + 1] = if pk.status != PokemonStatus::NONE { 1.0 } else { 0.0 };
        out[base + 2] = if pk.hp > 0 { 1.0 } else { 0.0 };
    }
    // 18..36 — opp active type1 one-hot (18).
    // 36..54 — opp active type2 one-hot (18).
    if active < 6 {
        let pk = &side.pokemon.pkmn[active];
        if let Some(i) = bo_type_index_18(&pk.types.0) {
            out[18 + i] = 1.0;
        }
        if let Some(i) = bo_type_index_18(&pk.types.1) {
            out[36 + i] = 1.0;
        }
    }
    // 54..104 — opp bench role one-hots (5 slots × 10 roles); active excluded.
    let mut bench_idx = 0;
    for slot in 0..6 {
        if slot == active { continue; }
        if bench_idx >= 5 { break; }
        let pk = &side.pokemon.pkmn[slot];
        let role = opp_role_of(pk.id);
        out[54 + bench_idx * N_OPP_ROLES + role] = 1.0;
        bench_idx += 1;
    }
    // 104..110 — opp active idx one-hot.
    if active < 6 {
        out[104 + active] = 1.0;
    }
    // 110..115 — opp active boosts.
    let stages = [
        side.attack_boost, side.defense_boost,
        side.special_attack_boost, side.special_defense_boost,
        side.speed_boost,
    ];
    for (i, s) in stages.iter().enumerate() {
        out[110 + i] = (*s as f32 / 6.0).clamp(-1.0, 1.0);
    }
    // 115..122 — opp active item flag one-hot.
    if active < 6 {
        let pk = &side.pokemon.pkmn[active];
        out[115 + item_flag_idx(pk.item)] = 1.0;
    }
    // 122..126 — opp active move-disabled flags (4 moves).
    if active < 6 {
        let pk = &side.pokemon.pkmn[active];
        out[122] = if pk.moves.m0.disabled { 1.0 } else { 0.0 };
        out[123] = if pk.moves.m1.disabled { 1.0 } else { 0.0 };
        out[124] = if pk.moves.m2.disabled { 1.0 } else { 0.0 };
        out[125] = if pk.moves.m3.disabled { 1.0 } else { 0.0 };
    }
}

fn encode_field_bo(state: &State, bo_side: &Side, opp_side: &Side, out: &mut [f32]) {
    // 0..5 — weather (drop NONE).
    match state.weather.weather_type {
        Weather::SUN | Weather::HARSHSUN => out[0] = 1.0,
        Weather::RAIN | Weather::HEAVYRAIN => out[1] = 1.0,
        Weather::SAND => out[2] = 1.0,
        Weather::SNOW => out[3] = 1.0,
        Weather::HAIL => out[4] = 1.0,
        _ => {}
    }
    // 5..9 — terrain (drop NONE).
    use crate::engine::state::Terrain;
    match state.get_terrain() {
        Terrain::ELECTRICTERRAIN => out[5] = 1.0,
        Terrain::GRASSYTERRAIN  => out[6] = 1.0,
        Terrain::MISTYTERRAIN   => out[7] = 1.0,
        Terrain::PSYCHICTERRAIN => out[8] = 1.0,
        _ => {}
    }
    // 9 — trick room.
    if state.trick_room.active { out[9] = 1.0; }
    // 10..16 — hazards: SR, Spikes, TSpikes for each side (BO first, then opp).
    let bo_sc = &bo_side.side_conditions;
    out[10] = if bo_sc.stealth_rock > 0 { 1.0 } else { 0.0 };
    out[11] = (bo_sc.spikes as f32 / 3.0).clamp(0.0, 1.0);
    out[12] = (bo_sc.toxic_spikes as f32 / 2.0).clamp(0.0, 1.0);
    let opp_sc = &opp_side.side_conditions;
    out[13] = if opp_sc.stealth_rock > 0 { 1.0 } else { 0.0 };
    out[14] = (opp_sc.spikes as f32 / 3.0).clamp(0.0, 1.0);
    out[15] = (opp_sc.toxic_spikes as f32 / 2.0).clamp(0.0, 1.0);
    // 16..18 — screens active per side.
    out[16] = if bo_sc.reflect > 0 || bo_sc.light_screen > 0 || bo_sc.aurora_veil > 0
              { 1.0 } else { 0.0 };
    out[17] = if opp_sc.reflect > 0 || opp_sc.light_screen > 0 || opp_sc.aurora_veil > 0
              { 1.0 } else { 0.0 };
}

fn encode_matchup_bo(bo_side: &Side, opp_side: &Side, out: &mut [f32]) {
    let bo_active = bo_side.active_index as usize;
    let opp_active = opp_side.active_index as usize;

    // Find canonical BO slots present on the side.
    let mut canon_slot: [Option<usize>; 6] = [None; 6];  // canon → real slot
    for slot in 0..6 {
        let pk = &bo_side.pokemon.pkmn[slot];
        if let Some(canon) = bo_canonical_slot(pk.id) {
            canon_slot[canon] = Some(slot);
        }
    }
    let bo_slot = |canon: usize| canon_slot[canon];
    let bo_pk = |canon: usize| -> Option<&Pokemon> {
        canon_slot[canon].map(|s| &bo_side.pokemon.pkmn[s])
    };

    // 0: Dnite DD level. Only when Dnite is active; use atk_boost as DD proxy.
    if let Some(slot) = bo_slot(5) {
        if bo_active == slot {
            out[0] = (bo_side.attack_boost as f32 / 6.0).clamp(0.0, 1.0);
        }
    }
    // 1: Hatt active.
    if let Some(slot) = bo_slot(3) {
        if bo_active == slot { out[1] = 1.0; }
    }
    // 2: Tusk active + Proto online (item consumed).
    if let Some(slot) = bo_slot(2) {
        if bo_active == slot {
            if let Some(pk) = bo_pk(2) {
                if pk.item != Items::BOOSTERENERGY {
                    out[2] = 1.0;
                }
            }
        }
    }
    // 3: Garg active + Salt Cure applicable (opp not Steel/Ghost).
    if let Some(slot) = bo_slot(0) {
        if bo_active == slot && opp_active < 6 {
            let opp = &opp_side.pokemon.pkmn[opp_active];
            if !defensive_blocks_salt_cure(&opp.types.0, &opp.types.1) {
                out[3] = 1.0;
            }
        }
    }
    // 4: Opp active outspeeds Dnite + 1 (proxy: opp_spe > dnite_spe * 1.5).
    if let (Some(dnite), true) = (bo_pk(5), opp_active < 6) {
        let opp = &opp_side.pokemon.pkmn[opp_active];
        if dnite.speed > 0 && (opp.speed as f32) > (dnite.speed as f32) * 1.5 {
            out[4] = 1.0;
        }
    }
    // 5: Opp has fire-steel wall alive (hand list).
    for slot in 0..6 {
        let pk = &opp_side.pokemon.pkmn[slot];
        if pk.hp > 0 && is_fire_steel_wall(pk.id) { out[5] = 1.0; break; }
    }
    // 6: Opp has strong priority user alive.
    for slot in 0..6 {
        let pk = &opp_side.pokemon.pkmn[slot];
        if pk.hp > 0 && is_strong_priority_user(pk.id) { out[6] = 1.0; break; }
    }
    // 7: Opp has Rocky Helmet user alive.
    for slot in 0..6 {
        let pk = &opp_side.pokemon.pkmn[slot];
        if pk.hp > 0 && pk.item == Items::ROCKYHELMET { out[7] = 1.0; break; }
    }
    // 8: Opp has Trick-item user alive (hand list + Choice item).
    for slot in 0..6 {
        let pk = &opp_side.pokemon.pkmn[slot];
        if pk.hp > 0 && is_trick_item_user(pk.id)
           && matches!(pk.item, Items::CHOICESCARF | Items::CHOICESPECS | Items::CHOICEBAND) {
            out[8] = 1.0;
            break;
        }
    }
    // 9: Sum opp active positive boosts (atk/def/spa/spd/spe), capped at 1.
    let stages = [
        opp_side.attack_boost, opp_side.defense_boost,
        opp_side.special_attack_boost, opp_side.special_defense_boost,
        opp_side.speed_boost,
    ];
    let pos_sum: i32 = stages.iter().filter(|&&s| s > 0).map(|&s| s as i32).sum();
    out[9] = (pos_sum as f32 / 6.0).clamp(0.0, 1.0);
}

/// Extract 184-dim BO-locked feature vector. Auto-orients BO-first regardless
/// of p1/p2 assignment so the same value net works for either side.
pub fn extract_features_bo(state: &State) -> Vec<f32> {
    let mut features = vec![0.0f32; STATE_BO_FEATURES];
    let (bo_side, opp_side) = if detect_bo_side(state) == 0 {
        (&state.side_one, &state.side_two)
    } else {
        (&state.side_two, &state.side_one)
    };
    let mut idx = 0;
    encode_bo_side(bo_side, &mut features[idx..idx + N_BO_SIDE]);
    idx += N_BO_SIDE;
    encode_opp_side(opp_side, &mut features[idx..idx + N_OPP_SIDE]);
    idx += N_OPP_SIDE;
    encode_field_bo(state, bo_side, opp_side, &mut features[idx..idx + N_FIELD_BO]);
    idx += N_FIELD_BO;
    encode_matchup_bo(bo_side, opp_side, &mut features[idx..idx + N_MATCHUP_BO]);
    debug_assert_eq!(idx + N_MATCHUP_BO, STATE_BO_FEATURES);
    features
}

// ==================== Value Network (dispatch wrapper) ====================

enum ValueNetBackend {
    Onnx {
        session: Mutex<Session>,
        input_dim: usize,
    },
    Material(MaterialNet),
}

pub struct ValueNet {
    backend: ValueNetBackend,
}

unsafe impl Sync for ValueNet {}

/// Detect whether a model file is the raw-Rust .material format (starts with
/// b"MAT1") or anything else (assumed ONNX).
fn is_material_file<P: AsRef<Path>>(path: P) -> bool {
    use std::io::Read;
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).is_ok() && &magic == b"MAT1"
}

impl ValueNet {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let path_ref = path.as_ref();
        if is_material_file(path_ref) {
            let net = MaterialNet::load(path_ref)?;
            return Ok(ValueNet { backend: ValueNetBackend::Material(net) });
        }
        let session = Session::builder()?.commit_from_file(path_ref)?;
        let input_dim = session.inputs().first()
            .and_then(|outlet| match outlet.dtype() {
                ort::value::ValueType::Tensor { shape, .. } => {
                    shape.get(1).and_then(|d| if *d > 0 { Some(*d as usize) } else { None })
                }
                _ => None,
            })
            .unwrap_or(STATE_FEATURES);
        Ok(ValueNet {
            backend: ValueNetBackend::Onnx {
                session: Mutex::new(session),
                input_dim,
            },
        })
    }

    /// Evaluate a state: returns win probability for side_one [0.0, 1.0].
    pub fn evaluate(&self, state: &State) -> f32 {
        match &self.backend {
            ValueNetBackend::Material(net) => net.evaluate(state),
            ValueNetBackend::Onnx { session, input_dim } => {
                let features = if *input_dim == STATE_BO_FEATURES {
                    extract_features_bo(state)
                } else if *input_dim == STATE_V3_FEATURES {
                    extract_features_v3(state)
                } else {
                    extract_features(state)
                };
                let input =
                    ort::value::Tensor::from_array(([1usize, *input_dim], features))
                        .expect("failed to create input tensor");
                let mut session = session.lock().unwrap();
                let outputs = session.run(ort::inputs![input]).expect("inference failed");
                let binding = outputs[0]
                    .try_extract_tensor::<f32>()
                    .expect("failed to extract output tensor");
                let logit = binding.1[0];
                1.0 / (1.0 + (-logit).exp())
            }
        }
    }

    /// Returns the full per-output vector. For V-nets this is [V]; for
    /// Q-nets it's the per-action Q vector. Used to feed Q values into
    /// MCTS as action priors (softmax(Q)) rather than aggregating to a
    /// scalar leaf value.
    pub fn evaluate_vec(&self, state: &State) -> Vec<f32> {
        match &self.backend {
            ValueNetBackend::Material(net) => net.evaluate_vec(state),
            ValueNetBackend::Onnx { .. } => {
                // ONNX-backed nets always have 1-d output for our use cases;
                // fall back to a single-element vec.
                vec![self.evaluate(state)]
            }
        }
    }

    /// Batched evaluation: amortizes per-call overhead across K states. For
    /// the ONNX backend this is the big win — one session.run call instead
    /// of K. For Material the overhead is already negligible so we just
    /// loop (could be batched later).
    pub fn evaluate_batch(&self, states: &[&State]) -> Vec<f32> {
        if states.is_empty() {
            return Vec::new();
        }
        match &self.backend {
            ValueNetBackend::Material(net) => {
                states.iter().map(|s| net.evaluate(s)).collect()
            }
            ValueNetBackend::Onnx { session, input_dim } => {
                let k = states.len();
                let dim = *input_dim;
                // Flatten all K feature vectors into a single [K, dim] buffer.
                let mut buf = Vec::with_capacity(k * dim);
                for &state in states {
                    if dim == STATE_BO_FEATURES {
                        buf.extend(extract_features_bo(state));
                    } else if dim == STATE_V3_FEATURES {
                        buf.extend(extract_features_v3(state));
                    } else {
                        buf.extend(extract_features(state));
                    }
                }
                let input = ort::value::Tensor::from_array(([k, dim], buf))
                    .expect("failed to create batched input tensor");
                let mut session = session.lock().unwrap();
                let outputs = session.run(ort::inputs![input])
                    .expect("batched inference failed");
                let binding = outputs[0]
                    .try_extract_tensor::<f32>()
                    .expect("failed to extract output tensor");
                // Output is shape [K, 1] or [K] — sigmoid each logit.
                binding.1.iter()
                    .map(|&logit| 1.0 / (1.0 + (-logit).exp()))
                    .collect()
            }
        }
    }
}

fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let t = temperature.max(0.01); // prevent division by zero
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| ((x - max) / t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

fn map_priors_to_options(probs: &[f32], options: &[MoveChoice], side: &Side) -> Vec<f32> {
    let mut priors = Vec::with_capacity(options.len());

    for opt in options {
        let idx = match opt {
            MoveChoice::Move(move_idx) => *move_idx as usize,
            #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
            MoveChoice::MoveTera(move_idx) => *move_idx as usize,
            #[cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]
            MoveChoice::MoveMega(move_idx) => *move_idx as usize,
            MoveChoice::Switch(pkmn_idx) => {
                let target = *pkmn_idx as usize;
                let active = side.active_index as usize;
                if target < active { 4 + target } else { 3 + target }
            }
            MoveChoice::None => 0,
        };
        let p = if idx < probs.len() { probs[idx] } else { 0.01 };
        priors.push(p);
    }

    let sum: f32 = priors.iter().sum();
    if sum > 0.0 {
        for p in &mut priors { *p /= sum; }
    }

    priors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_dimensions() {
        let state = State::default();
        let features = extract_features(&state);
        assert_eq!(features.len(), STATE_FEATURES);
        assert_eq!(STATE_FEATURES, 609);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_type_effectiveness_basic() {
        // fire vs grass = 2x
        assert_eq!(type_effectiveness(&PokemonType::FIRE, &PokemonType::GRASS, &PokemonType::TYPELESS), 2.0);
        // electric vs ground = 0x
        assert_eq!(type_effectiveness(&PokemonType::ELECTRIC, &PokemonType::GROUND, &PokemonType::TYPELESS), 0.0);
        // ice vs grass/flying = 4x
        assert_eq!(type_effectiveness(&PokemonType::ICE, &PokemonType::GRASS, &PokemonType::FLYING), 4.0);
    }

    #[test]
    fn test_load_and_predict() {
        let model_path = std::env::var("POLICY_NET_PATH").unwrap_or_default();
        if model_path.is_empty() {
            eprintln!("skipping: set POLICY_NET_PATH to run");
            return;
        }
        let policy = PolicyNet::load(&model_path).expect("failed to load model");
        let state = State::default();
        let features = extract_features(&state);
        let probs = policy.predict(&features);
        assert_eq!(probs.len(), N_ACTIONS);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probs sum to {}", sum);
        eprintln!("probs: {:?}", probs);
    }
}
