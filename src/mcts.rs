use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::state::State;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::time::Duration;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub children: HashMap<(usize, usize), Vec<Node>>,
    pub times_visited: u32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,
    pub use_priors: bool,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            children: HashMap::new(),
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            use_priors: false,
        }
    }
    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        let n_s1 = s1_options.len() as f32;
        let n_s2 = s2_options.len() as f32;
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: 1.0 / n_s1,  // uniform prior
                pending_visits: 0,
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .map(|x| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: 1.0 / n_s2,  // uniform prior
                pending_visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    unsafe fn populate_with_priors(
        &mut self,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_priors: &[f32],
        s2_priors: &[f32],
    ) {
        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: if i < s1_priors.len() { s1_priors[i] } else { 1.0 / s1_options.len() as f32 },
                pending_visits: 0,
            })
            .collect();
        let n_s2 = s2_options.len() as f32;
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: if i < s2_priors.len() { s2_priors[i] } else { 1.0 / n_s2 },
                pending_visits: 0,
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode], use_priors: bool) -> usize {
        let mut choice = 0;
        let mut best_score = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let score = if use_priors {
                node.puct(self.times_visited)
            } else {
                node.ucb1(self.times_visited)
            };
            if score > best_score {
                best_score = score;
                choice = index;
            }
        }
        choice
    }

    pub unsafe fn selection(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let use_p = self.use_priors;
        let s1_mc_index = self.maximize_ucb_for_side(&self.s1_options.as_ref().unwrap(), use_p);
        let s2_mc_index = self.maximize_ucb_for_side(&self.s2_options.as_ref().unwrap(), use_p);
        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Vec<Node>) -> *mut Node {
        let mut rng = rng();
        let weights: Vec<f64> = (*move_vector)
            .iter()
            .map(|x| x.instructions.percentage as f64)
            .collect();
        let dist = WeightedIndex::new(weights).unwrap();
        let chosen_node = &mut (&mut *move_vector)[dist.sample(&mut rng)];
        let chosen_node_ptr = chosen_node as *mut Node;
        chosen_node_ptr
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;

            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on
        let new_node_ptr = self.sample_node(&mut this_pair_vec);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);
        self.children
            .insert((s1_move_index, s2_move_index), this_pair_vec);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    /// Like `selection` but increments `pending_visits` on each MoveNode it
    /// selects along the descent. Used in batched MCTS so K concurrent
    /// selections within one batch spread across different moves.
    pub unsafe fn selection_with_vloss(&mut self, state: &mut State) -> (*mut Node, usize, usize) {
        let return_node = self as *mut Node;
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let use_p = self.use_priors;
        let s1_mc_index = self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap(), use_p);
        let s2_mc_index = self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap(), use_p);

        // Apply virtual loss to the selected moves at this node.
        self.s1_options.as_mut().unwrap()[s1_mc_index].pending_visits += 1;
        self.s2_options.as_mut().unwrap()[s2_mc_index].pending_visits += 1;

        let child_vector = self.children.get_mut(&(s1_mc_index, s2_mc_index));
        match child_vector {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Vec<Node>;
                let chosen_child = self.sample_node(child_vec_ptr);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection_with_vloss(state)
            }
            None => (return_node, s1_mc_index, s2_mc_index),
        }
    }

    /// Like `backpropagate` but for batched MCTS:
    ///  * does not reverse instructions on state (caller uses cloned states
    ///    and discards them after backprop, so no reverse needed),
    ///  * decrements `pending_visits` on each MoveNode it touches, undoing
    ///    the virtual loss added by `selection_with_vloss`.
    pub unsafe fn backpropagate_no_reverse(&mut self, score: f32) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;
        parent_s1_movenode.pending_visits =
            parent_s1_movenode.pending_visits.saturating_sub(1);

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;
        parent_s2_movenode.pending_visits =
            parent_s2_movenode.pending_visits.saturating_sub(1);

        (*self.parent).backpropagate_no_reverse(score);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = evaluate(state);
            sigmoid(eval - root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    pub prior: f32,
    /// Virtual-loss counter: in-flight selections that picked this move but
    /// haven't backpropped yet. Treated as 0-score visits in UCB/PUCT so
    /// concurrent selections within one batch spread across moves.
    pub pending_visits: u32,
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        let eff_visits = self.visits + self.pending_visits;
        if eff_visits == 0 {
            return f32::INFINITY;
        }
        // pending visits count as 0-score: total_score / (visits + pending)
        let score = (self.total_score / eff_visits as f32)
            + (2.0 * (parent_visits as f32).ln() / eff_visits as f32).sqrt();
        score
    }

    /// PUCT selection: uses prior probability to guide exploration.
    /// With uniform priors (all equal), this reduces to standard UCB1.
    pub fn puct(&self, parent_visits: u32) -> f32 {
        let c = 2.0;
        let eff_visits = self.visits + self.pending_visits;
        let q = if eff_visits == 0 {
            0.5  // optimistic init
        } else {
            // pending acts as 0-score visits
            self.total_score / eff_visits as f32
        };
        q + c * self.prior * (parent_visits as f32).sqrt() / (1.0 + eff_visits as f32)
    }

    pub fn average_score(&self) -> f32 {
        let score = self.total_score / self.visits as f32;
        score
    }
}

#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

fn do_mcts(root_node: &mut Node, state: &mut State, root_eval: &f32) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

/// SCALE for converting raw engine eval to a "winning probability" baseline.
/// Calibrated so sigmoid(eval / SCALE) over typical mid-game states has
/// roughly the same distribution as MCTS visit-weighted V (~N(0.5, 0.17)).
#[cfg(feature = "policy")]
const RESIDUAL_EVAL_SCALE: f32 = 150.0;

#[cfg(feature = "policy")]
fn do_mcts_with_value_net(
    root_node: &mut Node,
    state: &mut State,
    value_net: &crate::policy::ValueNet,
    root_eval: &f32,
    alpha: f32,
    residual: bool,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move) };
    // Two leaf-eval modes:
    //  blend:    rollout = alpha * v + (1 - alpha) * sigmoid(eval - root_eval)
    //  residual: rollout = clamp(sigmoid(eval/SCALE) + alpha * (2v - 1), 0, 1)
    //            (model output v ∈ [0,1] is interpreted as a centered residual)
    let battle_is_over = state.battle_is_over();
    let rollout_result = if battle_is_over == 0.0 {
        if residual {
            let v = value_net.evaluate(state);
            let h_abs = sigmoid(crate::engine::evaluate::evaluate(state) / RESIDUAL_EVAL_SCALE);
            (h_abs + alpha * (2.0 * v - 1.0)).clamp(0.0, 1.0)
        } else if alpha >= 1.0 {
            value_net.evaluate(state)
        } else if alpha <= 0.0 {
            sigmoid(crate::engine::evaluate::evaluate(state) - root_eval)
        } else {
            let v = value_net.evaluate(state);
            let h = sigmoid(crate::engine::evaluate::evaluate(state) - root_eval);
            alpha * v + (1.0 - alpha) * h
        }
    } else if battle_is_over == -1.0 {
        0.0
    } else {
        battle_is_over
    };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;

    let root_eval = evaluate(state);
    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..1000 {
            do_mcts(&mut root_node, state, &root_eval);
        }

        /*
        Cut off after 10 million iterations

        Under normal circumstances the bot will only run for 2.5-3.5 million iterations
        however towards the end of a battle the bot may perform tens of millions of iterations

        Beyond about 30 million iterations some floating point nonsense happens where
        MoveNode.total_score stops updating because f32 does not have enough precision

        I can push the problem farther out by using f64 but if the bot is running for 10 million iterations
        then it almost certainly sees a forced win
        */
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}

/// MCTS with policy net priors (PUCT selection).
pub fn perform_mcts_with_priors(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    s1_priors: &[f32],
    s2_priors: &[f32],
    max_time: Duration,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate_with_priors(
            side_one_options, side_two_options, s1_priors, s2_priors,
        );
    }
    root_node.root = true;
    root_node.use_priors = true;

    let root_eval = evaluate(state);
    let start_time = std::time::Instant::now();
    while start_time.elapsed() < max_time {
        for _ in 0..1000 {
            do_mcts(&mut root_node, state, &root_eval);
        }
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}

/// MCTS over multiple sampled opponent states.
///
/// Shares a single search tree but randomly picks which state to simulate
/// each batch of iterations. This naturally averages over uncertainty about
/// the opponent's team without splitting the search budget.
///
/// All states must have the same side_one (our team) and same available moves.
/// They differ only in side_two (the opponent's unrevealed Pokemon).
pub fn perform_mcts_multi(
    states: &mut Vec<State>,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
) -> MctsResult {
    if states.is_empty() {
        panic!("perform_mcts_multi called with empty states");
    }
    if states.len() == 1 {
        return perform_mcts(&mut states[0], side_one_options, side_two_options, max_time);
    }

    let mut root_node = Node::new();
    unsafe {
        root_node.populate(side_one_options, side_two_options);
    }
    root_node.root = true;

    // average root eval across all states
    let root_eval: f32 = states.iter().map(|s| evaluate(s)).sum::<f32>() / states.len() as f32;

    let n_states = states.len();
    let start_time = std::time::Instant::now();
    let mut state_idx = 0;

    while start_time.elapsed() < max_time {
        // round-robin: each batch of 1000 iterations uses the next state
        // guarantees equal coverage across all sampled opponent teams
        let state = &mut states[state_idx];
        for _ in 0..1000 {
            do_mcts(&mut root_node, state, &root_eval);
        }
        state_idx = (state_idx + 1) % n_states;

        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}

/// Batched value-net MCTS: runs K selections in lockstep (each on a cloned
/// state, with virtual loss to push concurrent paths apart), then evaluates
/// all K leaves in a single value-net call. Amortizes per-call NN overhead
/// across K — biggest win on ONNX backends where session.run has ~10-20μs
/// fixed cost per call.
#[cfg(feature = "policy")]
fn do_mcts_with_value_batched(
    root_node: &mut Node,
    root_state: &State,
    value_net: &crate::policy::ValueNet,
    root_eval: &f32,
    alpha: f32,
    residual: bool,
    batch_size: usize,
) {
    struct Pending {
        state: State,
        leaf_ptr: *mut Node,
        // Set only when expand() returned the parent (no child created):
        // we must manually undo virtual loss at (parent, s1_idx, s2_idx).
        no_child: Option<(*mut Node, usize, usize)>,
        battle_outcome: f32,        // 0.0 if not over
        value_idx: Option<usize>,   // index into the batch-eval result
    }

    let mut pending: Vec<Pending> = Vec::with_capacity(batch_size);
    let mut eval_idxs: Vec<usize> = Vec::with_capacity(batch_size);

    // Selection + expand for each rollout in the batch.
    for _ in 0..batch_size {
        let mut state_copy = root_state.clone();
        let (parent_ptr, s1_idx, s2_idx) = unsafe {
            root_node.selection_with_vloss(&mut state_copy)
        };
        let leaf_ptr = unsafe { (*parent_ptr).expand(&mut state_copy, s1_idx, s2_idx) };
        let no_child = std::ptr::eq(leaf_ptr, parent_ptr);
        let battle_outcome = state_copy.battle_is_over();
        // Only call the value net when battle isn't over AND its output
        // actually contributes (alpha > 0 or residual mode).
        let need_eval = battle_outcome == 0.0 && (alpha > 0.0 || residual);
        let value_idx = if need_eval {
            eval_idxs.push(pending.len());
            Some(eval_idxs.len() - 1)
        } else {
            None
        };
        pending.push(Pending {
            state: state_copy,
            leaf_ptr,
            no_child: if no_child { Some((parent_ptr, s1_idx, s2_idx)) } else { None },
            battle_outcome,
            value_idx,
        });
    }

    // Single batched value-net call covering every leaf that needs one.
    let values: Vec<f32> = if !eval_idxs.is_empty() {
        let states: Vec<&State> = eval_idxs.iter().map(|&i| &pending[i].state).collect();
        value_net.evaluate_batch(&states)
    } else {
        Vec::new()
    };

    // Compute leaf scores + backprop each rollout.
    for pr in pending.iter() {
        let leaf_score: f32 = if pr.battle_outcome != 0.0 {
            if pr.battle_outcome == -1.0 { 0.0 } else { pr.battle_outcome }
        } else if let Some(vi) = pr.value_idx {
            let v = values[vi];
            if residual {
                let h_abs = sigmoid(crate::engine::evaluate::evaluate(&pr.state) / RESIDUAL_EVAL_SCALE);
                (h_abs + alpha * (2.0 * v - 1.0)).clamp(0.0, 1.0)
            } else if alpha >= 1.0 {
                v
            } else {
                let h = sigmoid(crate::engine::evaluate::evaluate(&pr.state) - root_eval);
                alpha * v + (1.0 - alpha) * h
            }
        } else {
            // alpha == 0 (and not residual): pure engine heuristic at leaf.
            sigmoid(crate::engine::evaluate::evaluate(&pr.state) - root_eval)
        };

        // If expand made no new child (battle over at a non-root leaf), the
        // virtual loss applied at (parent.s1_options[s1_idx], ...[s2_idx])
        // is "stranded" — backprop_no_reverse only touches grandparent's
        // options. Manually decrement here before backprop.
        if let Some((p, s1i, s2i)) = pr.no_child {
            unsafe {
                let s1mn = &mut (*p).s1_options.as_mut().unwrap()[s1i];
                s1mn.pending_visits = s1mn.pending_visits.saturating_sub(1);
                let s2mn = &mut (*p).s2_options.as_mut().unwrap()[s2i];
                s2mn.pending_visits = s2mn.pending_visits.saturating_sub(1);
            }
        }
        unsafe { (*pr.leaf_ptr).backpropagate_no_reverse(leaf_score) };
    }
}

/// MCTS with value network for leaf evaluation.
/// Optionally also takes policy priors for PUCT selection.
/// `alpha` mixes value net with the engineered heuristic at leaves.
/// In *blend* mode (default):
///   rollout = alpha * value_net + (1 - alpha) * sigmoid(eval - root_eval)
/// In *residual* mode (set `residual=true`):
///   rollout = clamp(sigmoid(eval/SCALE) + alpha * (2*v - 1), 0, 1)
/// where the model's output `v ∈ [0,1]` is interpreted as a centered delta.
///
/// `batch_size`: when > 1, runs batched MCTS with virtual loss — K rollouts
/// per batch, one value-net call per batch. Best speedup with heavy value
/// nets (transformers, large MLPs); marginal for tiny material nets.
/// When 0 or 1, falls back to sequential (one rollout per leaf eval).
#[cfg(feature = "policy")]
pub fn perform_mcts_with_value(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    s1_priors: Option<&[f32]>,
    s2_priors: Option<&[f32]>,
    value_net: &crate::policy::ValueNet,
    alpha: f32,
    residual: bool,
    batch_size: usize,
    max_time: Duration,
) -> MctsResult {
    let mut root_node = Node::new();

    if let (Some(s1p), Some(s2p)) = (s1_priors, s2_priors) {
        unsafe {
            root_node.populate_with_priors(
                side_one_options, side_two_options, s1p, s2p,
            );
        }
        root_node.use_priors = true;
    } else {
        unsafe {
            root_node.populate(side_one_options, side_two_options);
        }
    }
    root_node.root = true;

    let root_eval = crate::engine::evaluate::evaluate(state);
    let start_time = std::time::Instant::now();
    let use_batched = batch_size > 1;
    let inner_loop_count = if use_batched { 1000 / batch_size.max(1) } else { 1000 };
    while start_time.elapsed() < max_time {
        if use_batched {
            // root_state stays immutable across the batched loop; each
            // rollout inside clones it before mutating.
            let root_state_ref: &State = state;
            for _ in 0..inner_loop_count.max(1) {
                do_mcts_with_value_batched(
                    &mut root_node, root_state_ref, value_net,
                    &root_eval, alpha, residual, batch_size,
                );
            }
        } else {
            for _ in 0..1000 {
                do_mcts_with_value_net(&mut root_node, state, value_net, &root_eval, alpha, residual);
            }
        }
        if root_node.times_visited == 10_000_000 {
            break;
        }
    }

    MctsResult {
        s1: root_node.s1_options.as_ref().unwrap().iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            }).collect(),
        s2: root_node.s2_options.as_ref().unwrap().iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            }).collect(),
        iteration_count: root_node.times_visited,
    }
}
