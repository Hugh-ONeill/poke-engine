use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::mcts::perform_mcts;
#[cfg(feature = "policy")]
use crate::mcts::perform_mcts_with_priors;
#[cfg(feature = "policy")]
use crate::policy::PolicyNet;
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Duration;

/// Result of a completed game.
pub struct GameResult {
    /// 1.0 = side_one won, -1.0 = side_two won, 0.0 = draw/timeout
    pub winner: f32,
    /// Number of turns played
    pub turns: u32,
}

/// Play a complete game: MCTS (side_one) vs MCTS (side_two).
///
/// side_one uses `s1_search_ms` per turn, side_two uses `s2_search_ms`.
/// Set s2_search_ms to a small value (e.g. 50) for a heuristic-level opponent.
pub fn play_game(
    state: &mut State,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> GameResult {
    let mut rng = rng();

    for turn in 0..max_turns {
        // check if battle is over
        let result = state.battle_is_over();
        if result != 0.0 {
            return GameResult {
                winner: result,
                turns: turn,
            };
        }

        // get available moves for both sides
        let (s1_options, s2_options) = state.get_all_options();

        // if either side has no options, game is stuck
        if s1_options.is_empty() || s2_options.is_empty() {
            return GameResult {
                winner: state.battle_is_over(),
                turns: turn,
            };
        }

        // P1: MCTS search
        let s1_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s1_search_ms),
        );
        let s1_move = s1_result
            .s1
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s1_options[0].clone());

        // P2: MCTS search (short time = heuristic)
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_search_ms),
        );
        let s2_move = s2_result
            .s2
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s2_options[0].clone());

        // generate all possible outcomes for this move pair
        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);

        if instructions.is_empty() {
            return GameResult {
                winner: 0.0,
                turns: turn,
            };
        }

        // sample an outcome weighted by probability
        let total_pct: f32 = instructions.iter().map(|i| i.percentage).sum();
        let roll = rng.random::<f32>() * total_pct;
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;
        for (i, inst) in instructions.iter().enumerate() {
            cumulative += inst.percentage;
            if roll <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        // apply the chosen outcome
        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    // hit max turns = draw
    GameResult {
        winner: 0.0,
        turns: max_turns,
    }
}

/// Record of a single turn for training data.
pub struct TurnRecord {
    /// Serialized state string before the move
    pub state_string: String,
    /// Move chosen by side_one
    pub s1_move: String,
    /// Visit counts for each s1 option (for soft targets)
    pub s1_visits: Vec<(String, u32)>,
    /// Move chosen by side_two
    pub s2_move: String,
    /// Visit counts for each s2 option (for soft targets)
    pub s2_visits: Vec<(String, u32)>,
}

/// Result of a recorded game.
pub struct RecordedGameResult {
    pub winner: f32,
    pub turns: Vec<TurnRecord>,
}

/// Play a game and record all turns for training data.
pub fn play_game_recorded(
    state: &mut State,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> RecordedGameResult {
    let mut rng = rng();
    let mut turns = Vec::new();

    for turn in 0..max_turns {
        let result = state.battle_is_over();
        if result != 0.0 {
            return RecordedGameResult { winner: result, turns };
        }

        let (s1_options, s2_options) = state.get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            return RecordedGameResult {
                winner: state.battle_is_over(),
                turns,
            };
        }

        // record state before moves
        let state_string = state.serialize();

        // P1: MCTS search
        let s1_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s1_search_ms),
        );
        let s1_move_node = s1_result.s1.iter().max_by_key(|m| m.visits).unwrap();
        let s1_move = s1_move_node.move_choice.clone();

        // record visit distribution for soft targets
        let s1_visits: Vec<(String, u32)> = s1_result
            .s1
            .iter()
            .map(|m| (m.move_choice.to_string(&state.side_one), m.visits))
            .collect();

        // P2: MCTS search
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_search_ms),
        );
        let s2_move_node = s2_result.s2.iter().max_by_key(|m| m.visits).unwrap();
        let s2_move = s2_move_node.move_choice.clone();

        // record visit distribution for s2 soft targets
        let s2_visits: Vec<(String, u32)> = s2_result
            .s2
            .iter()
            .map(|m| (m.move_choice.to_string(&state.side_two), m.visits))
            .collect();

        turns.push(TurnRecord {
            state_string,
            s1_move: s1_move.to_string(&state.side_one),
            s1_visits,
            s2_move: s2_move.to_string(&state.side_two),
            s2_visits,
        });

        // resolve turn
        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);
        if instructions.is_empty() {
            return RecordedGameResult { winner: 0.0, turns };
        }

        let total_pct: f32 = instructions.iter().map(|i| i.percentage).sum();
        let roll = rng.random::<f32>() * total_pct;
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;
        for (i, inst) in instructions.iter().enumerate() {
            cumulative += inst.percentage;
            if roll <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    RecordedGameResult { winner: 0.0, turns }
}

/// Play multiple recorded games in parallel.
pub fn play_games_recorded(
    state_template: &State,
    n_games: u32,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> Vec<RecordedGameResult> {
    (0..n_games)
        .into_par_iter()
        .map(|_| {
            let mut state = state_template.clone();
            play_game_recorded(&mut state, s1_search_ms, s2_search_ms, max_turns)
        })
        .collect()
}

/// Play multiple games in parallel and return (s1_wins, s2_wins, draws).
///
/// Uses rayon to run games across all available CPU cores.
pub fn play_games(
    state_template: &State,
    n_games: u32,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> (u32, u32, u32) {
    let results: Vec<f32> = (0..n_games)
        .into_par_iter()
        .map(|_| {
            let mut state = state_template.clone();
            let result = play_game(&mut state, s1_search_ms, s2_search_ms, max_turns);
            result.winner
        })
        .collect();

    let mut s1_wins = 0u32;
    let mut s2_wins = 0u32;
    let mut draws = 0u32;
    for w in results {
        match w {
            w if w > 0.0 => s1_wins += 1,
            w if w < 0.0 => s2_wins += 1,
            _ => draws += 1,
        }
    }

    (s1_wins, s2_wins, draws)
}

// ==================== Policy-Guided Games ====================

#[cfg(feature = "policy")]
/// Play a game with policy net guiding side_one's MCTS via PUCT priors.
/// Side two uses standard MCTS (no priors).
pub fn play_game_with_policy(
    state: &mut State,
    policy: &PolicyNet,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> GameResult {
    let mut rng = rng();

    for turn in 0..max_turns {
        let result = state.battle_is_over();
        if result != 0.0 {
            return GameResult {
                winner: result,
                turns: turn,
            };
        }

        let (s1_options, s2_options) = state.get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            return GameResult {
                winner: state.battle_is_over(),
                turns: turn,
            };
        }

        // P1: policy-guided MCTS
        let (s1_priors, s2_priors) = policy.get_priors(state, &s1_options, &s2_options);
        let s1_result = perform_mcts_with_priors(
            state,
            s1_options.clone(),
            s2_options.clone(),
            &s1_priors,
            &s2_priors,
            Duration::from_millis(s1_search_ms),
        );
        let s1_move = s1_result
            .s1
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s1_options[0].clone());

        // P2: standard MCTS
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_search_ms),
        );
        let s2_move = s2_result
            .s2
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s2_options[0].clone());

        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);
        if instructions.is_empty() {
            return GameResult {
                winner: 0.0,
                turns: turn,
            };
        }

        let total_pct: f32 = instructions.iter().map(|i| i.percentage).sum();
        let roll = rng.random::<f32>() * total_pct;
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;
        for (i, inst) in instructions.iter().enumerate() {
            cumulative += inst.percentage;
            if roll <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    GameResult {
        winner: 0.0,
        turns: max_turns,
    }
}

#[cfg(feature = "policy")]
/// Play multiple policy-guided games in parallel.
///
/// The PolicyNet is shared across threads via Arc (ort Sessions are thread-safe).
pub fn play_games_with_policy(
    state_template: &State,
    policy: Arc<PolicyNet>,
    n_games: u32,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> (u32, u32, u32) {
    let results: Vec<f32> = (0..n_games)
        .into_par_iter()
        .map(|_| {
            let mut state = state_template.clone();
            let result =
                play_game_with_policy(&mut state, &policy, s1_search_ms, s2_search_ms, max_turns);
            result.winner
        })
        .collect();

    let mut s1_wins = 0u32;
    let mut s2_wins = 0u32;
    let mut draws = 0u32;
    for w in results {
        match w {
            w if w > 0.0 => s1_wins += 1,
            w if w < 0.0 => s2_wins += 1,
            _ => draws += 1,
        }
    }

    (s1_wins, s2_wins, draws)
}

// ==================== Policy + Value Games ====================

#[cfg(feature = "policy")]
use crate::mcts::perform_mcts_with_value;
#[cfg(feature = "policy")]
use crate::policy::ValueNet;

#[cfg(feature = "policy")]
/// Play a game: side_one uses policy priors + value net eval, side_two uses standard MCTS.
pub fn play_game_with_policy_and_value(
    state: &mut State,
    policy: &PolicyNet,
    value_net: &ValueNet,
    s1_search_ms: u64,
    s2_search_ms: u64,
    max_turns: u32,
) -> GameResult {
    let mut rng = rng();

    for turn in 0..max_turns {
        let result = state.battle_is_over();
        if result != 0.0 {
            return GameResult {
                winner: result,
                turns: turn,
            };
        }

        let (s1_options, s2_options) = state.get_all_options();
        if s1_options.is_empty() || s2_options.is_empty() {
            return GameResult {
                winner: state.battle_is_over(),
                turns: turn,
            };
        }

        // P1: policy priors + value net eval
        let (s1_priors, s2_priors) = policy.get_priors(state, &s1_options, &s2_options);
        let s1_result = perform_mcts_with_value(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Some(&s1_priors),
            Some(&s2_priors),
            value_net,
            1.0,
            false,  // residual mode off in legacy game runner
            1,      // sequential mode (no batching)
            Duration::from_millis(s1_search_ms),
        );
        let s1_move = s1_result
            .s1
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s1_options[0].clone());

        // P2: standard MCTS
        let s2_result = perform_mcts(
            state,
            s1_options.clone(),
            s2_options.clone(),
            Duration::from_millis(s2_search_ms),
        );
        let s2_move = s2_result
            .s2
            .iter()
            .max_by_key(|m| m.visits)
            .map(|m| m.move_choice.clone())
            .unwrap_or(s2_options[0].clone());

        let instructions = generate_instructions_from_move_pair(state, &s1_move, &s2_move, true);
        if instructions.is_empty() {
            return GameResult {
                winner: 0.0,
                turns: turn,
            };
        }

        let total_pct: f32 = instructions.iter().map(|i| i.percentage).sum();
        let roll = rng.random::<f32>() * total_pct;
        let mut cumulative = 0.0;
        let mut chosen_idx = 0;
        for (i, inst) in instructions.iter().enumerate() {
            cumulative += inst.percentage;
            if roll <= cumulative {
                chosen_idx = i;
                break;
            }
        }

        state.apply_instructions(&instructions[chosen_idx].instruction_list);
    }

    GameResult {
        winner: 0.0,
        turns: max_turns,
    }
}
