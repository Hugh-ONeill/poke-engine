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
use crate::state::{Move, Pokemon, PokemonStatus, PokemonType, Side, State};
use crate::engine::state::PokemonVolatileStatus;
use ort::session::Session;
use std::path::Path;
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

// item ordering (trimmed for gen2)
const ITEM_ORDER_V2: [Items; 5] = [
    Items::LEFTOVERS, Items::THICKCLUB, Items::LIGHTBALL,
    Items::MIRACLEBERRY, Items::MINTBERRY,
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

// ==================== Value Network ====================

pub struct ValueNet {
    session: Mutex<Session>,
}

unsafe impl Sync for ValueNet {}

impl ValueNet {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, ort::Error> {
        let session = Session::builder()?.commit_from_file(path)?;
        Ok(ValueNet {
            session: Mutex::new(session),
        })
    }

    /// Evaluate a state: returns win probability for side_one [0.0, 1.0].
    pub fn evaluate(&self, state: &State) -> f32 {
        let features = extract_features(state);
        let input =
            ort::value::Tensor::from_array(([1usize, STATE_FEATURES], features))
                .expect("failed to create input tensor");

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![input]).expect("inference failed");
        let binding = outputs[0]
            .try_extract_tensor::<f32>()
            .expect("failed to extract output tensor");

        // output is a logit, apply sigmoid
        let logit = binding.1[0];
        1.0 / (1.0 + (-logit).exp())
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
