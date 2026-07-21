#![cfg(not(any(feature = "gen1", feature = "gen2", feature = "gen3")))]

use poke_engine::choices::Choices;
use poke_engine::engine::generate_instructions::generate_instructions_from_move_pair;
use poke_engine::engine::state::MoveChoice;
use poke_engine::mcts::{state_match_score, PersistentMcts};
use poke_engine::state::{PokemonMoveIndex, State};
use std::time::Duration;

fn basic_state() -> State {
    let mut state = State::default();
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    state
        .side_one
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::WATERGUN);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M0, Choices::TACKLE);
    state
        .side_two
        .get_active()
        .replace_move(PokemonMoveIndex::M1, Choices::THUNDERBOLT);
    state
}

/// Play out (s1_move, s2_move) from `state` the same way root expansion does
/// (branch_on_damage = true) and return the state implied by one of the
/// engine's chance hypotheses — i.e. "what actually happened" in a world
/// consistent with the engine's own model.
fn observed_after(state: &State, s1: &MoveChoice, s2: &MoveChoice) -> State {
    let mut scratch = state.clone();
    let instructions = generate_instructions_from_move_pair(&mut scratch, s1, s2, true);
    let mut observed = state.clone();
    observed.apply_instructions(&instructions[0].instruction_list);
    observed
}

#[test]
fn test_search_accumulates_across_calls() {
    let mut handle = PersistentMcts::new(basic_state());
    let r1 = handle.search(Duration::from_millis(30));
    let r2 = handle.search(Duration::from_millis(30));
    assert!(r1.iteration_count > 0);
    assert!(r2.iteration_count > r1.iteration_count);
}

#[test]
fn test_advance_exact_transition_reuses_subtree() {
    let state = basic_state();
    let mut handle = PersistentMcts::new(state.clone());
    handle.search(Duration::from_millis(100));
    let visits_before = handle.root_visits();

    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    let observed = observed_after(&state, &s1, &s2);

    let report = handle.advance(&s1, &s2, observed, 0.8);
    assert!(report.reused, "expected reuse, got: {}", report.reason);
    assert!(report.retained_visits > 0);
    assert!(
        report.match_score > 0.95,
        "exact transition should match near-perfectly, got {}",
        report.match_score
    );
    // the promoted subtree is strictly smaller than the whole tree
    assert!(handle.root_visits() < visits_before);
    assert_eq!(handle.root_visits(), report.retained_visits);

    // searching after an advance accumulates on top of the retained visits
    let r = handle.search(Duration::from_millis(30));
    assert!(r.iteration_count > report.retained_visits);
}

#[test]
fn test_advance_twice_walks_consecutive_decision_points() {
    let mut state = basic_state();
    let mut handle = PersistentMcts::new(state.clone());
    handle.search(Duration::from_millis(100));

    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    state = observed_after(&state, &s1, &s2);
    let report1 = handle.advance(&s1, &s2, state.clone(), 0.8);
    assert!(report1.reused, "first advance: {}", report1.reason);

    handle.search(Duration::from_millis(100));
    state = observed_after(&state, &s1, &s2);
    let report2 = handle.advance(&s1, &s2, state.clone(), 0.8);
    assert!(report2.reused, "second advance: {}", report2.reason);
    assert!(report2.retained_visits > 0);
}

#[test]
fn test_advance_unknown_move_resets_cleanly() {
    let state = basic_state();
    let mut handle = PersistentMcts::new(state.clone());
    handle.search(Duration::from_millis(30));

    // None is not a root option in a healthy no-forced-switch state
    let report = handle.advance(&MoveChoice::None, &MoveChoice::None, state, 0.8);
    assert!(!report.reused);
    assert_eq!(report.retained_visits, 0);

    // the handle must still be searchable after a failed advance
    let r = handle.search(Duration::from_millis(30));
    assert!(r.iteration_count > 0);
}

#[test]
fn test_advance_option_drift_resets() {
    let state = basic_state();
    let mut handle = PersistentMcts::new(state.clone());
    handle.search(Duration::from_millis(100));

    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    let mut observed = observed_after(&state, &s1, &s2);
    // authoritative state reveals our M1 is disabled (e.g. a choice lock the
    // tree did not model) -> cached option lists no longer describe the root
    observed.side_one.get_active().moves[&PokemonMoveIndex::M1].disabled = true;

    let report = handle.advance(&s1, &s2, observed, 0.8);
    assert!(!report.reused);
    assert_eq!(report.reason, "cached options != authoritative options");
}

#[test]
fn test_advance_mismatched_outcome_resets() {
    let state = basic_state();
    let mut handle = PersistentMcts::new(state.clone());
    handle.search(Duration::from_millis(100));

    let s1 = MoveChoice::Move(PokemonMoveIndex::M0);
    let s2 = MoveChoice::Move(PokemonMoveIndex::M0);
    // "reality" wildly disagrees with every engine hypothesis for this pair
    let mut observed = observed_after(&state, &s1, &s2);
    observed.side_two.get_active().hp = 1;
    for idx in poke_engine::state::pokemon_index_iter() {
        observed.side_one.pokemon[idx].hp = 1;
    }

    let report = handle.advance(&s1, &s2, observed, 0.8);
    assert!(!report.reused);
    assert_eq!(report.reason, "no chance outcome matched");
}

#[test]
fn test_state_match_score_discriminates() {
    let state = basic_state();
    assert!(state_match_score(&state, &state) > 0.999);

    let mut hurt = state.clone();
    hurt.side_two.get_active().hp = 1;
    assert!(state_match_score(&state, &hurt) < 0.8);

    // small HP differences (translator rounding / adjacent damage rolls)
    // stay comfortably above the default threshold
    let mut close = state.clone();
    close.side_two.get_active().hp = 97;
    assert!(state_match_score(&state, &close) > 0.9);
}
