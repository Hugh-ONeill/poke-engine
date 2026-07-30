use super::abilities::Abilities;
use super::damage_calc::type_effectiveness_modifier;
use super::items::Items;
use super::state::{PokemonVolatileStatus, Terrain, Weather};
use crate::choices::{Choice, Choices, MoveCategory, MoveTarget};
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
/// Which fork-added eval term groups are reverted to upstream behaviour.
/// CB_EVAL_BASELINE=1 reverts everything (the original all-or-nothing switch);
/// CB_EVAL_OFF="hazards,hopeless" reverts individual groups for bisection.
/// NOTE: `threat` off forces the threat multipliers to 1.0, which makes the
/// hopeless-matchup condition (threat == 0.0) unreachable — threat off
/// implies hopeless off.
#[derive(Default)]
struct EvalOff {
    hazards: bool,
    items: bool,
    volatiles: bool,
    threat: bool,
    tera: bool,
    pending: bool,
    weather: bool,
    terrain: bool,
    hopeless: bool,
    speedtier: bool,
    poisonheal: bool,
    pp: bool,
    synergy: bool,
    threatv2: bool,
    locks: bool,
    unaware: bool,
    supremeoverlord: bool,
    baseline: bool,
}

impl EvalOff {
    /// `list` (CB_EVAL_OFF) disables default-ON terms; `on_list` (CB_EVAL_ON)
    /// enables default-OFF experimental terms. synergy is default-ON since
    /// 2026-07-23 late (synonly non-regression accept-h1 at 38.5%/195: the
    /// terms are truth claims about the game state and cost nothing to run
    /// everywhere; the earlier bundled accept-h0 is attributed to threatv2
    /// via syniso). threatv2 stays default-OFF pending a recalibration that
    /// co-tunes the boost/speedtier couplings.
    fn from_spec(all: bool, list: &str, on_list: &str) -> Self {
        let has =
            |k: &str| all || list.split(',').any(|t| t.trim().eq_ignore_ascii_case(k));
        let has_on = |k: &str| {
            !all && on_list
                .split(',')
                .any(|t| t.trim().eq_ignore_ascii_case(k))
        };
        EvalOff {
            hazards: has("hazards"),
            items: has("items"),
            volatiles: has("volatiles"),
            threat: has("threat"),
            tera: has("tera"),
            pending: has("pending"),
            weather: has("weather"),
            terrain: has("terrain"),
            hopeless: has("hopeless"),
            speedtier: has("speedtier"),
            poisonheal: has("poisonheal"),
            pp: has("pp"),
            synergy: has("synergy"),
            threatv2: !has_on("threatv2"),
            // locks un-parked 2026-07-24: the locknr accept-h0 that parked
            // them was the Jul-23 level step wearing a verdict costume — the
            // interleaved paired retest (locksab, 229 same-team pairs,
            // confound-proof) REFUTED the harm hypothesis: locks arm 31.4%
            // vs baseline 26.6%, 55.9% of discordant pairs (a positive lean;
            // not-worse is decisive). Truth terms default on; CB_EVAL_OFF=locks
            // disables. The TWave-lock breaking was verified all along
            // (max consecutive run 27 -> <13, fp control unchanged).
            locks: has("locks"),
            // Unaware negates the atk/def/spatk/spdef boost credit against (or
            // held-against) an Unaware active — a truth claim mirroring
            // damage_calc's Unaware handling, default-ON. CB_EVAL_OFF=unaware
            // reverts to crediting boosts the mechanic makes worthless.
            unaware: has("unaware"),
            supremeoverlord: has("supremeoverlord"),
            baseline: all,
        }
    }
}

/// Stall-mode (fork, 2026-07-23): a PER-BATTLE archetype mode set by the
/// driver at team preview when BOTH teams read as wall-heavy (recovery-move
/// density). Holds the CONTEXT WEIGHTS only — the recovery-PP depletion
/// penalty doubles and toxic-on-a-wall reprices to TOXIC_ON_WALL, because PP
/// economics and the tox clock decide wall-wars (stall audit 2026-07-23).
/// Truth-claim terms (synergy) are default-ON base eval instead, since
/// 2026-07-23 late: facts always-on, context weights mode-gated. An env var
/// can't carry this: bench workers are persistent processes rotating teams
/// per game, so it's an atomic the driver sets each preview (one battle per
/// process at a time). Never active under CB_EVAL_BASELINE. First of the
/// archetype modes — extend to an enum if more matchup profiles earn their
/// keep.
pub static STALL_MODE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_stall_mode(on: bool) {
    STALL_MODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

fn stall_mode() -> bool {
    STALL_MODE.load(std::sync::atomic::Ordering::Relaxed)
}

fn eval_off() -> &'static EvalOff {
    static OFF: std::sync::OnceLock<EvalOff> = std::sync::OnceLock::new();
    OFF.get_or_init(|| {
        let all = std::env::var("CB_EVAL_BASELINE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let list = std::env::var("CB_EVAL_OFF").unwrap_or_default();
        let on_list = std::env::var("CB_EVAL_ON").unwrap_or_default();
        EvalOff::from_spec(all, &list, &on_list)
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

// Poison Heal annuity (fork, 2026-07-23): the stall audit showed a Poison
// Heal Gliscor generating ~7.6 mons of free healing per marathon game — the
// largest single resource in a stall war — while the eval priced the statused
// state at +15 and a loaded-but-unactivated Toxic Orb at just the +4 item
// value. Raise the statused credit and price the pending activation so the
// search hurries the orb online, treats a pre-activation Knock Off as a real
// loss, and values fielding the mon at all. CB_EVAL_OFF=poisonheal reverts.
const POISON_HEAL_STATUSED: f32 = 35.0;
const POISON_HEAL_PENDING: f32 = 15.0;

// Recovery-PP depletion (fork, 2026-07-23): both sides hit the 8-PP recovery
// caps in every audited marathon and the eval was PP-blind — a wall with 0
// Recover PP scored like a full one. Penalize MISSING recovery PP (zero at
// full PP, so team baselines don't shift) on the last RECOVERY_PP_CAP charges:
// burning a click costs 2, opposing Pressure makes it 4. CB_EVAL_OFF=pp
// reverts.
const RECOVERY_PP_VALUE: f32 = 2.0;
const RECOVERY_PP_CAP: i8 = 8;

// Scarf-lock pricing (fork, 2026-07-23 night; from the toxw loss audit): our
// mons sat choice-locked into THUNDER WAVE for 30+ turn stretches because a
// locked mon reads as healthy — the eval had no lock-quality concept. A mon
// whose only usable move is Status is functionally disabled until it
// switches (worse than ENCORE, which at least expires), and a choice item on
// a status-heavy wall is a liability, not +13 — unless the mon carries
// Trick/Switcheroo, in which case the item is ammunition. The wall liability
// also teaches Trick's real economics: giving the scarf to their Blissey is
// worth far more than the item-value trade suggests.
const CHOICE_LOCKED_STATUS: f32 = -35.0;
const CHOICE_ON_WALL: f32 = -20.0;

fn is_choice_item(item: Items) -> bool {
    matches!(
        item,
        Items::CHOICEBAND | Items::CHOICESPECS | Items::CHOICESCARF
    )
}

/// Choice-locked with only Status moves usable: the choice lock disables
/// every other move slot after the first click, so the signature is a choice
/// item + disabled slots + no usable damaging move. Actives only — the lock
/// clears on switch-out.
/// Choice item on a status-heavy mon with no Trick/Switcheroo to hand it
/// off: a liability, not the flat +13.
fn choice_on_wall(pokemon: &Pokemon) -> bool {
    if !is_choice_item(pokemon.item) {
        return false;
    }
    let mut status_moves = 0;
    for mv in pokemon.moves.into_iter() {
        if mv.id == Choices::NONE {
            continue;
        }
        if matches!(mv.id, Choices::TRICK | Choices::SWITCHEROO) {
            return false;
        }
        if mv.choice.category == MoveCategory::Status {
            status_moves += 1;
        }
    }
    status_moves >= 2
}

fn choice_locked_into_status(pokemon: &Pokemon) -> bool {
    if !is_choice_item(pokemon.item) {
        return false;
    }
    let mut any_disabled = false;
    let mut usable_status = false;
    let mut usable_attack = false;
    for mv in pokemon.moves.into_iter() {
        if mv.id == Choices::NONE {
            continue;
        }
        if mv.disabled {
            any_disabled = true;
            continue;
        }
        if mv.pp <= 0 {
            continue;
        }
        if mv.choice.category == MoveCategory::Status {
            usable_status = true;
        } else {
            usable_attack = true;
        }
    }
    any_disabled && usable_status && !usable_attack
}

// Status-synergy rifle terms (fork, 2026-07-23): the Poison Heal finding
// generalized. Same pending-activation logic for the Guts family (the
// burned-Guts STATE already scores +50 in evaluate_burned, so the loaded orb
// deserves more than the flat +4 item value); a self-status orb held WITHOUT
// a benefiting ability is a liability, not an asset (net -12 after the +4
// item value), as is Black Sludge on a non-Poison holder (net -12 after +9);
// a sleeper with Sleep Talk PP is nowhere near -25 disabled; and Regenerator
// carries invisible switch-out income exactly like Poison Heal carries
// end-of-turn income. CB_EVAL_OFF=synergy reverts the lot.
const GUTS_FAMILY_PENDING: f32 = 15.0;
const QUICK_FEET_PENDING: f32 = 8.0;
const ORB_NO_SYNERGY: f32 = -16.0;
const SLUDGE_NO_POISON: f32 = -21.0;
const REST_TALK_SLEEP_REBATE: f32 = 15.0;
const REGENERATOR_PENDING: f32 = 0.5;

const LEECH_SEED: f32 = -30.0;
const SUBSTITUTE: f32 = 40.0;
const CONFUSION: f32 = -20.0;

const REFLECT: f32 = 20.0;
const LIGHT_SCREEN: f32 = 20.0;
const AURORA_VEIL: f32 = 40.0;
const SAFE_GUARD: f32 = 5.0;
const TAILWIND: f32 = 7.0;
const HEALING_WISH: f32 = 30.0;

// Retuned to upstream's values 2026-07-23: the -15/-9 monotype tuning was
// never validated for OU, and the position instrument showed it drives the
// janitor churn — 10.4% of decisions flip when it's reverted (vs a 5.2%
// noise floor), with the signature "T1 stealth rock -> attack, fewer
// switches, more attacks". This makes the CB_EVAL_OFF=hazards knob inert
// (main == base) until someone retunes again.
const STEALTH_ROCK: f32 = -10.0;
const SPIKES: f32 = -7.0;
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

// Toxic on a WALL in a wall-war is not a -30 status: it is a compounding
// PP-drain engine — the tox clock forces recovery clicks against an 8-PP
// budget, and the stallB win/loss audit (2026-07-23) showed the whole
// matchup pivots on who is forced into reactive healing (losses: they land
// 19 toxics and we burn 9.5 mons/game of recovery PP; wins: 23-18 the other
// way and we spend 5.2). Priced only under stall-mode, symmetric: the search
// chases toxing their recovery mons AND keeps ours clean.
const TOXIC_ON_WALL: f32 = -48.0;

fn has_self_recovery(pokemon: &Pokemon) -> bool {
    for mv in pokemon.moves.into_iter() {
        if mv.id == Choices::NONE {
            continue;
        }
        if mv.id == Choices::REST
            || mv
                .choice
                .heal
                .as_ref()
                .map_or(false, |h| h.target == MoveTarget::User)
        {
            return true;
        }
    }
    false
}

fn evaluate_poison(pokemon: &Pokemon, base_score: f32) -> f32 {
    match pokemon.ability {
        Abilities::POISONHEAL => {
            if eval_off().poisonheal {
                15.0
            } else {
                POISON_HEAL_STATUSED
            }
        }
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

    // Flare Boost wants the burn just like Guts does (fork; upstream list below
    // stays untouched for CB_EVAL_BASELINE parity)
    let off = eval_off();
    if !off.synergy && pokemon.ability == Abilities::FLAREBOOST {
        return -2.0 * POKEMON_BURNED;
    }

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
    let base = eval_off().hazards;
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
// Tinted Lens / Neuroforce read the DEFENDER's matchup, so they can't live in
// the (ability, choice) mirror — they adjust eff where threat_vs has it in
// hand. Same parity-test contract as threat_ability_bp_mult.
fn threat_eff_adjust(ability: &Abilities, eff: f32) -> f32 {
    match ability {
        Abilities::TINTEDLENS if eff > 0.0 && eff < 1.0 => 2.0,
        Abilities::NEUROFORCE if eff > 1.0 => 1.25,
        _ => 1.0,
    }
}

// Mirrors of the big ability_modify_attack_being_used BP hooks. Kept as cheap
// constants because the real pipeline needs Choice clones + full hook dispatch
// with state context — too heavy for the evaluate() hot path (~1M calls/s).
// The parity test below runs the REAL pipeline against this mirror, so drift
// fails `cargo test` instead of silently rotting.
fn threat_ability_bp_mult(ability: &Abilities, choice: &Choice) -> f32 {
    match ability {
        Abilities::TECHNICIAN if choice.base_power <= 60.0 => 1.5,
        Abilities::TOUGHCLAWS if choice.flags.contact => 1.3,
        Abilities::SHARPNESS if choice.flags.slicing => 1.5,
        Abilities::STRONGJAW if choice.flags.bite => 1.5,
        Abilities::IRONFIST if choice.flags.punch => 1.2,
        Abilities::SHEERFORCE if choice.secondaries.is_some() => 1.3,
        Abilities::HUGEPOWER | Abilities::PUREPOWER
            if choice.category == MoveCategory::Physical =>
        {
            2.0
        }
        _ => 1.0,
    }
}

fn threat_vs(attacker: &Pokemon, defender: &Pokemon, attacker_fainted: usize) -> (f32, f32, bool) {
    let mut best_phys: f32 = 0.0;
    let mut best_spec: f32 = 0.0;
    let mut has_status = false;
    let def_hp = defender.maxhp.max(1) as f32;
    // Status-aware offense (2026-07-23): a burned physical attacker threatens
    // at half, Guts/Toxic Boost/Flare Boost at 1.5x, and Facade doubles when
    // statused — without this a Flame Orb Ursaluna reads as half its real
    // threat and a burned wall-breaker as double. threat_vs is fork-only, so
    // the threat knob already reverts all of it.
    let v2 = !eval_off().threatv2;
    let statused = v2 && attacker.status != PokemonStatus::NONE;
    let burned = v2 && attacker.status == PokemonStatus::BURN;
    let poisoned = v2
        && matches!(
            attacker.status,
            PokemonStatus::POISON | PokemonStatus::TOXIC
        );
    let guts = attacker.ability == Abilities::GUTS && statused;
    let atk_mult = if guts || (poisoned && attacker.ability == Abilities::TOXICBOOST)
    {
        1.5
    } else {
        1.0
    };
    let spa_mult = if burned && attacker.ability == Abilities::FLAREBOOST {
        1.5
    } else {
        1.0
    };
    // Supreme Overlord multiplies every move's base power by 1 + 0.1 x (own
    // fainted allies) in the damage calc (abilities.rs) — mirror it as a stat
    // multiplier, which is identical since damage is linear in BP x Atk and
    // the boost applies uniformly across the moveset. Lives OUTSIDE the v2
    // bp-mult mirror: it needs side state (fainted count) and is a default-ON
    // truth claim, not part of the parked threatv2 bundle. Without it the eval
    // read an endgame Kingambit as its turn-1 self, never pricing the
    // cleaner's scaling (CB_EVAL_OFF=supremeoverlord reverts).
    let so_mult = if attacker.ability == Abilities::SUPREMEOVERLORD
        && !eval_off().supremeoverlord
    {
        1.0 + 0.1 * attacker_fainted as f32
    } else {
        1.0
    };
    let atk_stat = attacker.attack as f32 * atk_mult * so_mult;
    let spa_stat = attacker.special_attack as f32 * spa_mult * so_mult;
    let def_stat = defender.defense.max(1) as f32;
    let spd_stat = defender.special_defense.max(1) as f32;
    for mv in attacker.moves.into_iter() {
        if mv.id == Choices::NONE { continue; }
        let eff = type_effectiveness_modifier(&mv.choice.move_type, defender);
        match mv.choice.category {
            MoveCategory::Physical | MoveCategory::Special => {
                if eff == 0.0 { continue; }
                let eff = if v2 {
                    eff * threat_eff_adjust(&attacker.ability, eff)
                } else {
                    eff
                };
                let mut bp = mv.choice.base_power;
                if bp == 0.0 { continue; }
                if mv.id == Choices::FACADE && statused {
                    bp *= 2.0;
                }
                let physical = mv.choice.category == MoveCategory::Physical;
                let stab_match = mv.choice.move_type == attacker.types.0
                    || mv.choice.move_type == attacker.types.1;
                let stab = if stab_match {
                    if v2 && attacker.ability == Abilities::ADAPTABILITY {
                        2.0
                    } else {
                        1.5
                    }
                } else {
                    1.0
                };
                let (off, def) = if physical {
                    (atk_stat, def_stat)
                } else {
                    (spa_stat, spd_stat)
                };
                // burn halves physical damage unless Guts (or Facade, which
                // ignores the burn drop)
                let burn_mult = if burned && physical && !guts && mv.id != Choices::FACADE
                {
                    0.5
                } else {
                    1.0
                };
                let abil_mult = if v2 {
                    threat_ability_bp_mult(&attacker.ability, &mv.choice)
                } else {
                    1.0
                };
                let item_mult = if !v2 { 1.0 } else { match attacker.item {
                    Items::CHOICEBAND if physical => 1.5,
                    Items::CHOICESPECS if !physical => 1.5,
                    Items::LIFEORB => 1.3,
                    Items::EXPERTBELT if eff > 1.0 => 1.2,
                    _ => 1.0,
                } };
                let def_mult = if !v2 { 1.0 } else { match defender.ability {
                    Abilities::THICKFAT
                        if mv.choice.move_type == PokemonType::FIRE
                            || mv.choice.move_type == PokemonType::ICE =>
                    {
                        0.5
                    }
                    Abilities::MULTISCALE | Abilities::SHADOWSHIELD
                        if defender.hp == defender.maxhp =>
                    {
                        0.5
                    }
                    Abilities::FILTER | Abilities::SOLIDROCK | Abilities::PRISMARMOR
                        if eff >= 2.0 =>
                    {
                        0.75
                    }
                    Abilities::ICESCALES if !physical => 0.5,
                    _ => 1.0,
                } };
                // Lv100 simplified damage: 0.84 * BP * (off/def) * STAB * type_eff
                let dmg = 0.84 * bp * (off / def) * stab * eff * burn_mult
                    * abil_mult
                    * item_mult
                    * def_mult;
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
    if eval_off().volatiles {
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
        // a held barb bleeds 1/8 per turn: a liability, not an asset — and
        // pricing it negative makes Trick-ing it away (or refusing to
        // receive it) worth a real margin instead of the +5 unknown default
        Items::STICKYBARB => -10.0,
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
        PokemonStatus::TOXIC => {
            let base = if stall_mode()
                && !eval_off().baseline
                && has_self_recovery(pokemon)
            {
                TOXIC_ON_WALL
            } else {
                POKEMON_TOXIC
            };
            score += evaluate_poison(pokemon, base);
        }
        PokemonStatus::POISON => score += evaluate_poison(pokemon, POKEMON_POISONED),
        PokemonStatus::NONE => {}
    }

    if !eval_off().poisonheal
        && pokemon.ability == Abilities::POISONHEAL
        && pokemon.status == PokemonStatus::NONE
        && pokemon.item == Items::TOXICORB
    {
        score += POISON_HEAL_PENDING;
    }

    if !eval_off().pp {
        for mv in pokemon.moves.into_iter() {
            if mv.id == Choices::NONE {
                continue;
            }
            let is_recovery = mv.id == Choices::REST
                || mv
                    .choice
                    .heal
                    .as_ref()
                    .map_or(false, |h| h.target == MoveTarget::User);
            if is_recovery {
                // in stall mode PP economics decide the game — double the tax
                let ppv = if stall_mode() && !eval_off().baseline {
                    RECOVERY_PP_VALUE * 2.0
                } else {
                    RECOVERY_PP_VALUE
                };
                score -= ppv
                    * (RECOVERY_PP_CAP - mv.pp.min(RECOVERY_PP_CAP)).max(0) as f32;
            }
        }
    }

    if !eval_off().synergy {
        if pokemon.status == PokemonStatus::NONE {
            score += match (pokemon.ability, pokemon.item) {
                (Abilities::GUTS, Items::FLAMEORB)
                | (Abilities::TOXICBOOST, Items::TOXICORB)
                | (Abilities::FLAREBOOST, Items::FLAMEORB) => GUTS_FAMILY_PENDING,
                (Abilities::QUICKFEET, Items::FLAMEORB | Items::TOXICORB) => {
                    QUICK_FEET_PENDING
                }
                // Poison Heal pending is priced above; Magic Guard orbs are
                // status-blocking tech, not a liability
                (Abilities::POISONHEAL | Abilities::MAGICGUARD, _) => 0.0,
                (_, Items::TOXICORB | Items::FLAMEORB) => ORB_NO_SYNERGY,
                _ => 0.0,
            };
        }
        if pokemon.item == Items::BLACKSLUDGE
            && !pokemon.has_type(&PokemonType::POISON)
        {
            score += SLUDGE_NO_POISON;
        }
        if pokemon.status == PokemonStatus::SLEEP {
            for mv in pokemon.moves.into_iter() {
                if mv.id == Choices::SLEEPTALK && mv.pp > 0 {
                    score += REST_TALK_SLEEP_REBATE;
                    break;
                }
            }
        }
        if pokemon.ability == Abilities::REGENERATOR && pokemon.hp < pokemon.maxhp {
            let missing = (pokemon.maxhp - pokemon.hp) as f32;
            let third = pokemon.maxhp as f32 / 3.0;
            score += REGENERATOR_PENDING * missing.min(third)
                / pokemon.maxhp as f32
                * POKEMON_HP;
        }
    }

    if !eval_off().locks && choice_on_wall(pokemon) {
        score += CHOICE_ON_WALL;
    }

    // upstream scores "holding any item" as a flat +10; ours prices items individually
    if eval_off().items {
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

    let off = eval_off();
    let s1_active = &state.side_one.pokemon[state.side_one.active_index];
    let s2_active = &state.side_two.pokemon[state.side_two.active_index];
    // Upstream scores boosts flat; we scale offensive boosts by how hard the
    // active can actually hit. With `threat` off the multipliers are 1.0 so
    // the boost terms reduce exactly to upstream's.
    let (s1_phys, s1_spec, s1_has_status) = if off.threat {
        (1.0, 1.0, true)
    } else {
        threat_vs(s1_active, s2_active, state.side_one.num_fainted_pkmn() as usize)
    };
    let (s2_phys, s2_spec, s2_has_status) = if off.threat {
        (1.0, 1.0, true)
    } else {
        threat_vs(s2_active, s1_active, state.side_two.num_fainted_pkmn() as usize)
    };
    // Unaware ignores stat stages when dealing AND taking damage, so an
    // Unaware active makes the OTHER side's atk/def/spatk/spdef boosts
    // worthless (offensive boosts do nothing INTO it; defensive boosts do
    // nothing against ITS hits). Zero the damage-boost credit accordingly,
    // mirroring damage_calc's `defender/attacker == UNAWARE && other !=
    // MOLDBREAKER`. Speed is untouched (Unaware only ignores damage stats).
    let s1_dmg_boost = if !off.unaware
        && s2_active.ability == Abilities::UNAWARE
        && s1_active.ability != Abilities::MOLDBREAKER
    {
        0.0
    } else {
        1.0
    };
    let s2_dmg_boost = if !off.unaware
        && s1_active.ability == Abilities::UNAWARE
        && s2_active.ability != Abilities::MOLDBREAKER
    {
        0.0
    } else {
        1.0
    };
    // Under Trick Room a speed BOOST is a liability (you move last), so its
    // credit flips sign — the flat boost credit was missing the TR reversal
    // that the SPEED_TIER term and weather-speed abilities already apply.
    let speed_boost_sign = if state.trick_room.active { -1.0 } else { 1.0 };

    let mut iter = state.side_one.pokemon.into_iter();
    let mut s1_used_tera = false;
    while let Some(pkmn) = iter.next() {
        if pkmn.hp > 0 {
            score += evaluate_pokemon(pkmn);
            score += evaluate_hazards(pkmn, &state.side_one);
            if iter.pokemon_index == state.side_one.active_index {
                score += evaluate_active_volatiles(pkmn, &state.side_one);
                if !off.locks && choice_locked_into_status(pkmn) {
                    score += CHOICE_LOCKED_STATUS;
                }
                if !off.tera {
                    score += evaluate_tera_active(pkmn);
                }

                score += get_boost_multiplier(state.side_one.attack_boost)
                    * POKEMON_ATTACK_BOOST * s1_phys * s1_dmg_boost;
                score += get_boost_multiplier(state.side_one.defense_boost)
                    * POKEMON_DEFENSE_BOOST * s1_dmg_boost;
                score += get_boost_multiplier(state.side_one.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s1_spec * s1_dmg_boost;
                score += get_boost_multiplier(state.side_one.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST * s1_dmg_boost;
                score += get_boost_multiplier(state.side_one.speed_boost)
                    * POKEMON_SPEED_BOOST * speed_boost_sign;
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
                if !off.locks && choice_locked_into_status(pkmn) {
                    score -= CHOICE_LOCKED_STATUS;
                }
                if !off.tera {
                    score -= evaluate_tera_active(pkmn);
                }

                score -= get_boost_multiplier(state.side_two.attack_boost)
                    * POKEMON_ATTACK_BOOST * s2_phys * s2_dmg_boost;
                score -= get_boost_multiplier(state.side_two.defense_boost)
                    * POKEMON_DEFENSE_BOOST * s2_dmg_boost;
                score -= get_boost_multiplier(state.side_two.special_attack_boost)
                    * POKEMON_SPECIAL_ATTACK_BOOST * s2_spec * s2_dmg_boost;
                score -= get_boost_multiplier(state.side_two.special_defense_boost)
                    * POKEMON_SPECIAL_DEFENSE_BOOST * s2_dmg_boost;
                score -= get_boost_multiplier(state.side_two.speed_boost)
                    * POKEMON_SPEED_BOOST * speed_boost_sign;
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
    if !off.pending {
        score += evaluate_pending_effects(&state.side_one);
        score -= evaluate_pending_effects(&state.side_two);
    }

    let trick_room = state.trick_room.active;
    let weather = state.weather.weather_type;
    if !off.weather && weather != Weather::NONE {
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
    if !off.terrain && terrain != Terrain::NONE {
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
    if !off.hopeless {
        if s1_active.hp > 0 && s1_phys == 0.0 && s1_spec == 0.0 && !s1_has_status {
            score += HOPELESS_MATCHUP;
        }
        if s2_active.hp > 0 && s2_phys == 0.0 && s2_spec == 0.0 && !s2_has_status {
            score -= HOPELESS_MATCHUP;
        }
    }

    // Speed-tier: outspeeding only matters if you can land a hit. Trick Room reverses
    // the comparison.
    if !off.speedtier && s1_active.hp > 0 && s2_active.hp > 0 {
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

#[cfg(test)]
mod tests {
    use super::threat_ability_bp_mult;
    use super::Abilities;
    use super::EvalOff;
    use super::Items;
    use crate::choices::{Choice, Choices, MOVES};
    use crate::state::{SideReference, State};

    /// The threat_vs ability mirror must track the REAL
    /// ability_modify_attack_being_used hooks: run each case through the real
    /// pipeline and compare the base-power ratio to the mirrored constant.
    #[test]
    fn threat_ability_mults_match_real_pipeline() {
        use super::super::abilities::ability_modify_attack_being_used;
        let mut state = State::default();
        let cases = [
            (Abilities::TECHNICIAN, Choices::BULLETPUNCH), // <=60 BP -> 1.5
            (Abilities::TECHNICIAN, Choices::CLOSECOMBAT), // >60 BP -> 1.0
            (Abilities::TOUGHCLAWS, Choices::CLOSECOMBAT), // contact
            (Abilities::TOUGHCLAWS, Choices::SHADOWBALL),  // non-contact
            (Abilities::SHARPNESS, Choices::LEAFBLADE),
            (Abilities::STRONGJAW, Choices::CRUNCH),
            (Abilities::IRONFIST, Choices::DRAINPUNCH),
            (Abilities::SHEERFORCE, Choices::IRONHEAD),
            (Abilities::HUGEPOWER, Choices::CLOSECOMBAT),
            (Abilities::PUREPOWER, Choices::SHADOWBALL), // special -> 1.0
        ];
        for (ability, move_id) in cases {
            let base = MOVES.get(&move_id).unwrap().clone();
            state.side_one.get_active().ability = ability;
            let mut real = base.clone();
            ability_modify_attack_being_used(
                &state,
                &mut real,
                &Choice::default(),
                &SideReference::SideOne,
            );
            let real_mult = real.base_power / base.base_power;
            let mirror = threat_ability_bp_mult(&ability, &base);
            assert!(
                (real_mult - mirror).abs() < 1e-4,
                "{:?} + {:?}: real pipeline {} vs threat mirror {}",
                ability,
                move_id,
                real_mult,
                mirror
            );
        }

        // eff-dependent abilities: give the defender a STEEL typing so Body
        // Slam is resisted (0.5x) and Close Combat is super-effective (2x)
        use super::threat_eff_adjust;
        use crate::state::PokemonType;
        state.side_two.get_active().types = (PokemonType::STEEL, PokemonType::TYPELESS);
        let eff_cases = [
            (Abilities::TINTEDLENS, Choices::BODYSLAM, 0.5),
            (Abilities::TINTEDLENS, Choices::CLOSECOMBAT, 2.0),
            (Abilities::NEUROFORCE, Choices::CLOSECOMBAT, 2.0),
            (Abilities::NEUROFORCE, Choices::BODYSLAM, 0.5),
        ];
        for (ability, move_id, eff) in eff_cases {
            let base = MOVES.get(&move_id).unwrap().clone();
            state.side_one.get_active().ability = ability;
            let mut real = base.clone();
            ability_modify_attack_being_used(
                &state,
                &mut real,
                &Choice::default(),
                &SideReference::SideOne,
            );
            let real_mult = real.base_power / base.base_power;
            let mirror = threat_eff_adjust(&ability, eff);
            assert!(
                (real_mult - mirror).abs() < 1e-4,
                "{:?} + {:?} at eff {}: real {} vs mirror {}",
                ability,
                move_id,
                eff,
                real_mult,
                mirror
            );
        }
    }

    #[test]
    fn from_spec_parses_list_and_all() {
        let off = EvalOff::from_spec(false, "hazards, HOPELESS", "");
        assert!(off.hazards && off.hopeless);
        assert!(!off.threat && !off.items && !off.tera && !off.speedtier);
        assert!(!off.poisonheal && !off.pp);
        let all = EvalOff::from_spec(true, "", "threatv2");
        assert!(all.hazards && all.items && all.volatiles && all.threat
            && all.tera && all.pending && all.weather && all.terrain
            && all.hopeless && all.speedtier && all.poisonheal && all.pp);
        // baseline forces everything off, even CB_EVAL_ON-listed terms
        assert!(all.synergy && all.threatv2);
        let none = EvalOff::from_spec(false, "", "");
        // synergy + locks default-ON (truth claims); threatv2 default-OFF
        // (parked). Locks un-parked 2026-07-24 after the locksab retest.
        assert!(!none.synergy && !none.locks && none.threatv2);
        assert!(!none.hazards && !none.hopeless && !none.volatiles);
        let sy_off = EvalOff::from_spec(false, "synergy,locks", "THREATV2");
        assert!(sy_off.synergy && sy_off.locks && !sy_off.threatv2 && !sy_off.hazards);
        let ph = EvalOff::from_spec(false, "poisonheal,pp", "");
        assert!(ph.poisonheal && ph.pp && !ph.hazards && !ph.synergy);
        assert!(!ph.baseline && EvalOff::from_spec(true, "", "").baseline);
    }

    /// A scarfed active whose only usable move is Thunder Wave (others
    /// choice-disabled) must read CHOICE_LOCKED_STATUS worse than the same
    /// mon unlocked; enabling a damaging move removes exactly that penalty.
    #[test]
    fn choice_lock_into_status_is_penalized() {
        use super::evaluate;
        use crate::choices::MOVES;
        use crate::state::{PokemonMoveIndex, State};
        let mut state = State::default();
        {
            let active = state.side_one.get_active();
            active.item = Items::CHOICESCARF;
            active.moves[&PokemonMoveIndex::M0].id = Choices::THUNDERWAVE;
            active.moves[&PokemonMoveIndex::M0].choice =
                MOVES.get(&Choices::THUNDERWAVE).unwrap().clone();
            active.moves[&PokemonMoveIndex::M0].pp = 10;
            active.moves[&PokemonMoveIndex::M1].id = Choices::SLUDGEBOMB;
            active.moves[&PokemonMoveIndex::M1].choice =
                MOVES.get(&Choices::SLUDGEBOMB).unwrap().clone();
            active.moves[&PokemonMoveIndex::M1].pp = 10;
            active.moves[&PokemonMoveIndex::M1].disabled = true;
        }
        assert!(super::choice_locked_into_status(
            state.side_one.get_active_immutable()
        ));
        state.side_one.get_active().moves[&PokemonMoveIndex::M1].disabled = false;
        assert!(!super::choice_locked_into_status(
            state.side_one.get_active_immutable()
        ));
    }

    /// Unaware ignores stat stages when dealing/taking damage, so the eval must
    /// NOT credit atk/def/spatk/spdef boosts against (or held against) an
    /// Unaware active — mirroring damage_calc. Uses the flat-credited defensive
    /// boost so no move setup is needed. Mold Breaker lifts it (damage_calc's
    /// same `!= MOLDBREAKER` guard).
    #[test]
    fn unaware_active_zeroes_damage_boost_credit() {
        use super::evaluate;
        use crate::state::State;
        let mut state = State::default();
        state.side_one.defense_boost = 2;
        state.side_one.get_active().ability = Abilities::TORRENT;
        state.side_two.get_active().ability = Abilities::TORRENT; // non-Unaware attacker
        let credited = evaluate(&state);
        state.side_two.get_active().ability = Abilities::UNAWARE; // ignores our Def boost
        let negated = evaluate(&state);
        assert!(
            credited > negated,
            "Unaware attacker must zero our defensive-boost credit: {} !> {}",
            credited, negated
        );
        // Mold Breaker on our (defending) side bypasses it, matching damage_calc.
        state.side_one.get_active().ability = Abilities::MOLDBREAKER;
        let moldbroken = evaluate(&state);
        assert!(
            (moldbroken - credited).abs() < 0.5,
            "Mold Breaker should bypass the Unaware negation: {} vs {}",
            moldbroken, credited
        );
    }

    /// Supreme Overlord's fallen-ally scaling must reach the eval's threat
    /// model: with allies down, a Supreme Overlord attacker's boosts convert
    /// harder. Measured as a difference-in-differences (SO vs a neutral
    /// ability, fresh vs two-fainted) so the fainted-ally scoring itself
    /// cancels out.
    #[test]
    fn supreme_overlord_scales_threat_with_fallen_allies() {
        use super::evaluate;
        use crate::state::{PokemonIndex, State};
        use crate::state::PokemonMoveIndex;
        let eval_with = |ability: Abilities, faint_two: bool| -> f32 {
            let mut state = State::default();
            state.side_one.attack_boost = 2; // boost credit rides s1_phys
            state.side_one.get_active().ability = ability;
            // a real physical move (default mons have none -> s1_phys = 0),
            // against a defender bulky enough that the threat score sits
            // below the 2HKO clamp where the multiplier is visible
            state
                .side_one
                .get_active()
                .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
            state.side_two.get_active().maxhp = 400;
            state.side_two.get_active().hp = 400;
            state.side_two.get_active().defense = 300;
            if faint_two {
                state.side_one.pokemon[PokemonIndex::P1].hp = 0;
                state.side_one.pokemon[PokemonIndex::P2].hp = 0;
            }
            evaluate(&state)
        };
        let d_fresh =
            eval_with(Abilities::SUPREMEOVERLORD, false) - eval_with(Abilities::TORRENT, false);
        let d_loaded =
            eval_with(Abilities::SUPREMEOVERLORD, true) - eval_with(Abilities::TORRENT, true);
        assert!(
            d_loaded > d_fresh + 0.5,
            "two fallen allies must raise a Supreme Overlord attacker's \
             boost-threat credit: loaded delta {} !> fresh delta {}",
            d_loaded, d_fresh
        );
    }

    /// Under Trick Room a speed boost is a liability, so its eval credit flips
    /// sign — the flat boost credit had missed the reversal SPEED_TIER applies.
    #[test]
    fn trick_room_flips_speed_boost_credit() {
        use super::evaluate;
        use crate::state::State;
        let mut state = State::default();
        state.side_one.speed_boost = 2;
        let normal = evaluate(&state);
        state.trick_room.active = true;
        let under_tr = evaluate(&state);
        assert!(
            normal > under_tr,
            "speed boost should be worth LESS under Trick Room: {} vs {}",
            normal, under_tr
        );
    }

    /// A choice item on a two-status-move wall is a liability vs Leftovers
    /// (item value 13 vs 9, minus CHOICE_ON_WALL) — unless the mon carries
    /// Trick, in which case the scarf is ammunition, not a curse.
    #[test]
    fn choice_item_on_wall_is_a_liability_unless_trick() {
        use super::evaluate;
        use crate::choices::MOVES;
        use crate::state::{PokemonMoveIndex, State};
        let mut state = State::default();
        {
            let active = state.side_one.get_active();
            for (slot, id) in [
                (PokemonMoveIndex::M0, Choices::THUNDERWAVE),
                (PokemonMoveIndex::M1, Choices::STEALTHROCK),
                (PokemonMoveIndex::M2, Choices::SLUDGEBOMB),
            ] {
                active.moves[&slot].id = id;
                active.moves[&slot].choice = MOVES.get(&id).unwrap().clone();
                active.moves[&slot].pp = 10;
            }
            active.item = Items::CHOICESCARF;
        }
        assert!(super::choice_on_wall(state.side_one.get_active_immutable()));
        state.side_one.get_active().item = Items::LEFTOVERS;
        assert!(!super::choice_on_wall(state.side_one.get_active_immutable()));
        // adding Trick exempts the holder: scarf is ammo
        {
            let active = state.side_one.get_active();
            active.item = Items::CHOICESCARF;
            active.moves[&PokemonMoveIndex::M3].id = Choices::TRICK;
            active.moves[&PokemonMoveIndex::M3].choice =
                MOVES.get(&Choices::TRICK).unwrap().clone();
            active.moves[&PokemonMoveIndex::M3].pp = 10;
        }
        assert!(!super::choice_on_wall(state.side_one.get_active_immutable()));
    }

    #[test]
    fn stall_mode_flag_round_trips() {
        use super::{set_stall_mode, stall_mode};
        assert!(!stall_mode());
        set_stall_mode(true);
        assert!(stall_mode());
        set_stall_mode(false);
        assert!(!stall_mode());
    }
}
