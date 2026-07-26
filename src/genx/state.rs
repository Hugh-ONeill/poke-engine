use super::abilities::Abilities;
use super::choice_effects::charge_volatile_to_choice;
use super::generate_instructions::immune_to_status;
use super::items::Items;
use crate::choices::{Choice, Choices, MoveCategory, MoveTarget};
use crate::define_enum_with_from_str;
use crate::instruction::BoostInstruction;
use crate::instruction::{
    ChangeSideConditionInstruction, ChangeStatInstruction, ChangeType,
    ChangeVolatileStatusDurationInstruction, Instruction, RemoveVolatileStatusInstruction,
    StateInstructions,
};
use crate::pokemon::PokemonName;
use crate::state::{
    LastUsedMove, Pokemon, PokemonBoostableStat, PokemonIndex, PokemonMoveIndex,
    PokemonSideCondition, PokemonStatus, PokemonType, Side, SideReference, State,
};
use core::panic;
use std::collections::HashSet;

fn common_pkmn_stat_calc(stat: u16, ev: u16, level: u16) -> u16 {
    // 31 IV always used
    ((2 * stat + 31 + (ev / 4)) * level) / 100
}

fn multiply_boost(boost_num: i8, stat_value: i16) -> i16 {
    match boost_num {
        -6 => stat_value * 2 / 8,
        -5 => stat_value * 2 / 7,
        -4 => stat_value * 2 / 6,
        -3 => stat_value * 2 / 5,
        -2 => stat_value * 2 / 4,
        -1 => stat_value * 2 / 3,
        0 => stat_value,
        1 => stat_value * 3 / 2,
        2 => stat_value * 4 / 2,
        3 => stat_value * 5 / 2,
        4 => stat_value * 6 / 2,
        5 => stat_value * 7 / 2,
        6 => stat_value * 8 / 2,
        _ => panic!("Invalid boost number: {}", boost_num),
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, Hash)]
pub enum MoveChoice {
    MoveTera(PokemonMoveIndex),
    MoveMega(PokemonMoveIndex),
    Move(PokemonMoveIndex),
    Switch(PokemonIndex),
    None,
}

impl MoveChoice {
    pub fn to_string(&self, side: &Side) -> String {
        match self {
            MoveChoice::MoveTera(index) => {
                format!("{}-tera", side.get_active_immutable().moves[&index].id).to_lowercase()
            }
            MoveChoice::MoveMega(index) => {
                format!("{}-mega", side.get_active_immutable().moves[&index].id).to_lowercase()
            }
            MoveChoice::Move(index) => {
                format!("{}", side.get_active_immutable().moves[&index].id).to_lowercase()
            }
            MoveChoice::Switch(index) => format!("{}", side.pokemon[*index].id).to_lowercase(),
            MoveChoice::None => "No Move".to_string(),
        }
    }
    pub fn from_string(s: &str, side: &Side) -> Option<MoveChoice> {
        let s = s.to_lowercase();
        if s == "none" {
            return Some(MoveChoice::None);
        }

        let mut pkmn_iter = side.pokemon.into_iter();
        while let Some(pkmn) = pkmn_iter.next() {
            if pkmn.id.to_string().to_lowercase() == s
                && pkmn_iter.pokemon_index != side.active_index
            {
                return Some(MoveChoice::Switch(pkmn_iter.pokemon_index));
            }
        }

        // check if s endswith `-tera`
        // if it does, find the move with the name and return MoveChoice::MoveTera
        // if it doesn't, find the move with the name and return MoveChoice::Move
        let mut move_iter = side.get_active_immutable().moves.into_iter();
        let mut move_name = s;
        if move_name.ends_with("-tera") {
            move_name = move_name[..move_name.len() - 5].to_string();
            while let Some(mv) = move_iter.next() {
                if format!("{:?}", mv.id).to_lowercase() == move_name {
                    return Some(MoveChoice::MoveTera(move_iter.pokemon_move_index));
                }
            }
        } else if move_name.ends_with("-mega") {
            move_name = move_name[..move_name.len() - 5].to_string();
            while let Some(mv) = move_iter.next() {
                if format!("{:?}", mv.id).to_lowercase() == move_name {
                    return Some(MoveChoice::MoveMega(move_iter.pokemon_move_index));
                }
            }
        } else {
            while let Some(mv) = move_iter.next() {
                if format!("{:?}", mv.id).to_lowercase() == move_name {
                    return Some(MoveChoice::Move(move_iter.pokemon_move_index));
                }
            }
        }

        None
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(PartialEq, Eq, Hash, Debug, Copy, Clone)]
    PokemonVolatileStatus {
        NONE,
        AQUARING,
        ATTRACT,
        AUTOTOMIZE,
        BANEFULBUNKER,
        BIDE,
        BOUNCE,
        BURNINGBULWARK,
        CHARGE,
        CONFUSION,
        CURSE,
        DEFENSECURL,
        DESTINYBOND,
        DIG,
        DISABLE,
        DIVE,
        ELECTRIFY,
        ELECTROSHOT,
        EMBARGO,
        ENCORE,
        ENDURE,
        FLASHFIRE,
        FLINCH,
        FLY,
        FOCUSENERGY,
        FOLLOWME,
        FORESIGHT,
        FREEZESHOCK,
        GASTROACID,
        GEOMANCY,
        GLAIVERUSH,
        GRUDGE,
        HEALBLOCK,
        HELPINGHAND,
        ICEBURN,
        IMPRISON,
        INGRAIN,
        KINGSSHIELD,
        LASERFOCUS,
        LEECHSEED,
        LIGHTSCREEN,
        LOCKEDMOVE,
        MAGICCOAT,
        MAGNETRISE,
        MAXGUARD,
        METEORBEAM,
        MINIMIZE,
        MIRACLEEYE,
        MUSTRECHARGE,
        NIGHTMARE,
        NORETREAT,
        OCTOLOCK,
        PARTIALLYTRAPPED,
        PERISH4,
        PERISH3,
        PERISH2,
        PERISH1,
        PHANTOMFORCE,
        POWDER,
        POWERSHIFT,
        POWERTRICK,
        PROTECT,
        PROTOSYNTHESISATK,
        PROTOSYNTHESISDEF,
        PROTOSYNTHESISSPA,
        PROTOSYNTHESISSPD,
        PROTOSYNTHESISSPE,
        QUARKDRIVEATK,
        QUARKDRIVEDEF,
        QUARKDRIVESPA,
        QUARKDRIVESPD,
        QUARKDRIVESPE,
        RAGE,
        RAGEPOWDER,
        RAZORWIND,
        REFLECT,
        ROOST,
        SALTCURE,
        SHADOWFORCE,
        SKULLBASH,
        SKYATTACK,
        SKYDROP,
        SILKTRAP,
        SLOWSTART,
        SMACKDOWN,
        SNATCH,
        SOLARBEAM,
        SOLARBLADE,
        SPARKLINGARIA,
        SPIKYSHIELD,
        SPOTLIGHT,
        STOCKPILE,
        SUBSTITUTE,
        SYRUPBOMB,
        TARSHOT,
        TAUNT,
        TELEKINESIS,
        THROATCHOP,
        TRUANT,
        TORMENT,
        TYPECHANGE,
        UNBURDEN,
        UPROAR,
        YAWN,
    },
    default = NONE
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone)]
    Weather {
        NONE,
        SUN,
        RAIN,
        SAND,
        HAIL,
        SNOW,
        HARSHSUN,
        HEAVYRAIN,
    }
}

define_enum_with_from_str! {
    #[repr(u8)]
    #[derive(Debug, PartialEq, Copy, Clone)]
    Terrain {
        NONE,
        ELECTRICTERRAIN,
        PSYCHICTERRAIN,
        MISTYTERRAIN,
        GRASSYTERRAIN,
    }
}

impl Pokemon {
    pub fn can_mega_evolve(&self) -> bool {
        // this assumes that if you have the correct mega stone, you can always mega evolve
        // even if another pkmn on the team already mega evolved
        // it is incorrect but practically most teams aren't going to have multiple mega stones
        if let Some(_mega_evolve_data) = self.id.mega_evolve_target(self.item) {
            true
        } else {
            false
        }
    }

    pub fn recalculate_stats(
        &mut self,
        side_ref: &SideReference,
        instructions: &mut StateInstructions,
    ) {
        // recalculate stats from base-stats and push any changes made to the StateInstructions
        let stats = self.calculate_stats_from_base_stats();
        if stats.1 != self.attack {
            let ins = Instruction::ChangeAttack(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.1 - self.attack,
            });
            self.attack = stats.1;
            instructions.instruction_list.push(ins);
        }
        if stats.2 != self.defense {
            let ins = Instruction::ChangeDefense(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.2 - self.defense,
            });
            self.defense = stats.2;
            instructions.instruction_list.push(ins);
        }
        if stats.3 != self.special_attack {
            let ins = Instruction::ChangeSpecialAttack(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.3 - self.special_attack,
            });
            self.special_attack = stats.3;
            instructions.instruction_list.push(ins);
        }
        if stats.4 != self.special_defense {
            let ins = Instruction::ChangeSpecialDefense(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.4 - self.special_defense,
            });
            self.special_defense = stats.4;
            instructions.instruction_list.push(ins);
        }
        if stats.5 != self.speed {
            let ins = Instruction::ChangeSpeed(ChangeStatInstruction {
                side_ref: *side_ref,
                amount: stats.5 - self.speed,
            });
            self.speed = stats.5;
            instructions.instruction_list.push(ins);
        }
    }
    pub fn calculate_stats_from_base_stats(&self) -> (i16, i16, i16, i16, i16, i16) {
        let base_stats = self.id.base_stats();
        (
            (common_pkmn_stat_calc(base_stats.0 as u16, self.evs.0 as u16, self.level as u16)
                + self.level as u16
                + 10) as i16,
            (common_pkmn_stat_calc(base_stats.1 as u16, self.evs.1 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.2 as u16, self.evs.2 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.3 as u16, self.evs.3 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.4 as u16, self.evs.4 as u16, self.level as u16) + 5)
                as i16,
            (common_pkmn_stat_calc(base_stats.5 as u16, self.evs.5 as u16, self.level as u16) + 5)
                as i16,
        )
    }
    pub fn add_available_moves(
        &self,
        vec: &mut Vec<MoveChoice>,
        last_used_move: &LastUsedMove,
        encored: bool,
        taunted: bool,
        can_tera: bool,
    ) {
        let mut iter = self.moves.into_iter();
        while let Some(p) = iter.next() {
            if !p.disabled && p.pp > 0 {
                match last_used_move {
                    LastUsedMove::Move(last_used_move) => {
                        if encored && last_used_move != &iter.pokemon_move_index {
                            continue;
                        } else if (self.moves[last_used_move].id == Choices::BLOODMOON
                            || self.moves[last_used_move].id == Choices::GIGATONHAMMER)
                            && &iter.pokemon_move_index == last_used_move
                        {
                            continue;
                        }
                    }
                    _ => {
                        // there are some situations where you switched out and got encored into
                        // a move from a different pokemon because you also have that move.
                        // just assume nothing is locked in this case
                    }
                }
                if (self.item == Items::ASSAULTVEST || taunted)
                    && self.moves[&iter.pokemon_move_index].choice.category == MoveCategory::Status
                {
                    continue;
                }
                vec.push(MoveChoice::Move(iter.pokemon_move_index));
                if can_tera {
                    vec.push(MoveChoice::MoveTera(iter.pokemon_move_index));
                }
                if self.can_mega_evolve() {
                    vec.push(MoveChoice::MoveMega(iter.pokemon_move_index));
                }
            }
        }
    }

    pub fn add_move_from_choice(&self, vec: &mut Vec<MoveChoice>, choice: Choices) {
        let mut iter = self.moves.into_iter();
        while let Some(p) = iter.next() {
            if p.id == choice {
                vec.push(MoveChoice::Move(iter.pokemon_move_index));
            }
        }
    }

    #[cfg(feature = "terastallization")]
    pub fn has_type(&self, pkmn_type: &PokemonType) -> bool {
        if self.terastallized {
            pkmn_type == &self.tera_type
        } else {
            pkmn_type == &self.types.0 || pkmn_type == &self.types.1
        }
    }

    #[cfg(not(feature = "terastallization"))]
    pub fn has_type(&self, pkmn_type: &PokemonType) -> bool {
        pkmn_type == &self.types.0 || pkmn_type == &self.types.1
    }

    pub fn item_is_permanent(&self) -> bool {
        match self.item {
            Items::LUSTROUSGLOBE => self.id == PokemonName::PALKIAORIGIN,
            Items::GRISEOUSCORE => self.id == PokemonName::GIRATINAORIGIN,
            Items::ADAMANTCRYSTAL => self.id == PokemonName::DIALGAORIGIN,
            Items::RUSTEDSWORD => {
                self.id == PokemonName::ZACIANCROWNED || self.id == PokemonName::ZACIAN
            }
            Items::RUSTEDSHIELD => {
                self.id == PokemonName::ZAMAZENTACROWNED || self.id == PokemonName::ZAMAZENTA
            }
            Items::SPLASHPLATE => self.id == PokemonName::ARCEUSWATER,
            Items::TOXICPLATE => self.id == PokemonName::ARCEUSPOISON,
            Items::EARTHPLATE => self.id == PokemonName::ARCEUSGROUND,
            Items::STONEPLATE => self.id == PokemonName::ARCEUSROCK,
            Items::INSECTPLATE => self.id == PokemonName::ARCEUSBUG,
            Items::SPOOKYPLATE => self.id == PokemonName::ARCEUSGHOST,
            Items::IRONPLATE => self.id == PokemonName::ARCEUSSTEEL,
            Items::FLAMEPLATE => self.id == PokemonName::ARCEUSFIRE,
            Items::MEADOWPLATE => self.id == PokemonName::ARCEUSGRASS,
            Items::ZAPPLATE => self.id == PokemonName::ARCEUSELECTRIC,
            Items::MINDPLATE => self.id == PokemonName::ARCEUSPSYCHIC,
            Items::ICICLEPLATE => self.id == PokemonName::ARCEUSICE,
            Items::DRACOPLATE => self.id == PokemonName::ARCEUSDRAGON,
            Items::DREADPLATE => self.id == PokemonName::ARCEUSDARK,
            Items::FISTPLATE => self.id == PokemonName::ARCEUSFIGHTING,
            Items::BLANKPLATE => self.id == PokemonName::ARCEUS,
            Items::SKYPLATE => self.id == PokemonName::ARCEUSFLYING,
            Items::PIXIEPLATE => self.id == PokemonName::ARCEUSFAIRY,
            Items::BUGMEMORY => self.id == PokemonName::SILVALLYBUG,
            Items::FIGHTINGMEMORY => self.id == PokemonName::SILVALLYFIGHTING,
            Items::GHOSTMEMORY => self.id == PokemonName::SILVALLYGHOST,
            Items::PSYCHICMEMORY => self.id == PokemonName::SILVALLYPSYCHIC,
            Items::FLYINGMEMORY => self.id == PokemonName::SILVALLYFLYING,
            Items::STEELMEMORY => self.id == PokemonName::SILVALLYSTEEL,
            Items::ICEMEMORY => self.id == PokemonName::SILVALLYICE,
            Items::POISONMEMORY => self.id == PokemonName::SILVALLYPOISON,
            Items::FIREMEMORY => self.id == PokemonName::SILVALLYFIRE,
            Items::DRAGONMEMORY => self.id == PokemonName::SILVALLYDRAGON,
            Items::GROUNDMEMORY => self.id == PokemonName::SILVALLYGROUND,
            Items::WATERMEMORY => self.id == PokemonName::SILVALLYWATER,
            Items::DARKMEMORY => self.id == PokemonName::SILVALLYDARK,
            Items::ROCKMEMORY => self.id == PokemonName::SILVALLYROCK,
            Items::GRASSMEMORY => self.id == PokemonName::SILVALLYGRASS,
            Items::FAIRYMEMORY => self.id == PokemonName::SILVALLYFAIRY,
            Items::ELECTRICMEMORY => self.id == PokemonName::SILVALLYELECTRIC,
            Items::CORNERSTONEMASK => {
                self.id == PokemonName::OGERPONCORNERSTONE
                    || self.id == PokemonName::OGERPONCORNERSTONETERA
            }
            Items::HEARTHFLAMEMASK => {
                self.id == PokemonName::OGERPONHEARTHFLAME
                    || self.id == PokemonName::OGERPONHEARTHFLAMETERA
            }
            Items::WELLSPRINGMASK => {
                self.id == PokemonName::OGERPONWELLSPRING
                    || self.id == PokemonName::OGERPONWELLSPRINGTERA
            }
            _ => false,
        }
    }

    pub fn item_can_be_removed(&self) -> bool {
        if self.ability == Abilities::STICKYHOLD {
            return false;
        }
        !self.item_is_permanent()
    }

    pub fn is_grounded(&self) -> bool {
        if self.item == Items::IRONBALL {
            return true;
        }
        if self.has_type(&PokemonType::FLYING)
            || self.ability == Abilities::LEVITATE
            || self.item == Items::AIRBALLOON
        {
            return false;
        }
        true
    }

    pub fn volatile_status_can_be_applied(
        &self,
        volatile_status: &PokemonVolatileStatus,
        active_volatiles: &HashSet<PokemonVolatileStatus>,
        first_move: bool,
    ) -> bool {
        if active_volatiles.contains(volatile_status) || self.hp == 0 {
            return false;
        }
        match volatile_status {
            PokemonVolatileStatus::LEECHSEED => {
                if self.has_type(&PokemonType::GRASS)
                    || active_volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE)
                {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::CONFUSION => {
                if active_volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE) {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::SUBSTITUTE => self.hp > self.maxhp / 4,
            PokemonVolatileStatus::FLINCH => {
                if !first_move || [Abilities::INNERFOCUS].contains(&self.ability) {
                    return false;
                }
                true
            }
            PokemonVolatileStatus::PROTECT => first_move,
            PokemonVolatileStatus::TAUNT
            | PokemonVolatileStatus::TORMENT
            | PokemonVolatileStatus::ENCORE
            | PokemonVolatileStatus::DISABLE
            | PokemonVolatileStatus::HEALBLOCK
            | PokemonVolatileStatus::ATTRACT => self.ability != Abilities::AROMAVEIL,
            _ => true,
        }
    }

    pub fn immune_to_stats_lowered_by_opponent(
        &self,
        stat: &PokemonBoostableStat,
        volatiles: &HashSet<PokemonVolatileStatus>,
    ) -> bool {
        if [
            Abilities::CLEARBODY,
            Abilities::WHITESMOKE,
            Abilities::FULLMETALBODY,
        ]
        .contains(&self.ability)
            || ([Items::CLEARAMULET].contains(&self.item))
        {
            return true;
        }

        if volatiles.contains(&PokemonVolatileStatus::SUBSTITUTE) {
            return true;
        }

        if stat == &PokemonBoostableStat::Attack && self.ability == Abilities::HYPERCUTTER {
            return true;
        } else if stat == &PokemonBoostableStat::Accuracy && self.ability == Abilities::KEENEYE {
            return true;
        }

        false
    }
}

impl Side {
    pub fn reset_negative_boosts(
        &mut self,
        side_ref: SideReference,
        instructions: &mut StateInstructions,
    ) -> bool {
        let mut changed = false;
        if self.attack_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::Attack,
                    amount: -self.attack_boost,
                }));
            self.attack_boost = 0;
            changed = true;
        }
        if self.defense_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::Defense,
                    amount: -self.defense_boost,
                }));
            self.defense_boost = 0;
            changed = true;
        }
        if self.special_attack_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::SpecialAttack,
                    amount: -self.special_attack_boost,
                }));
            self.special_attack_boost = 0;
            changed = true;
        }
        if self.special_defense_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::SpecialDefense,
                    amount: -self.special_defense_boost,
                }));
            self.special_defense_boost = 0;
            changed = true;
        }
        if self.speed_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::Speed,
                    amount: -self.speed_boost,
                }));
            self.speed_boost = 0;
            changed = true;
        }
        if self.accuracy_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::Accuracy,
                    amount: -self.accuracy_boost,
                }));
            self.accuracy_boost = 0;
            changed = true;
        }
        if self.evasion_boost < 0 {
            instructions
                .instruction_list
                .push(Instruction::Boost(BoostInstruction {
                    side_ref,
                    stat: PokemonBoostableStat::Evasion,
                    amount: -self.evasion_boost,
                }));
            self.evasion_boost = 0;
            changed = true;
        }
        changed
    }
    pub fn active_is_charging_move(&self) -> Option<PokemonMoveIndex> {
        for volatile in self.volatile_statuses.iter() {
            if let Some(choice) = charge_volatile_to_choice(volatile) {
                let mut iter = self.get_active_immutable().moves.into_iter();
                while let Some(mv) = iter.next() {
                    if mv.id == choice {
                        return Some(iter.pokemon_move_index);
                    }
                }
            }
        }
        None
    }

    pub fn calculate_highest_stat(&self) -> PokemonBoostableStat {
        let mut highest_stat = PokemonBoostableStat::Attack;
        let mut highest_stat_value = self.calculate_boosted_stat(PokemonBoostableStat::Attack);
        for stat in [
            PokemonBoostableStat::Defense,
            PokemonBoostableStat::SpecialAttack,
            PokemonBoostableStat::SpecialDefense,
            PokemonBoostableStat::Speed,
        ] {
            let stat_value = self.calculate_boosted_stat(stat);
            if stat_value > highest_stat_value {
                highest_stat = stat;
                highest_stat_value = stat_value;
            }
        }
        highest_stat
    }
    pub fn get_boost_from_boost_enum(&self, boost_enum: &PokemonBoostableStat) -> i8 {
        match boost_enum {
            PokemonBoostableStat::Attack => self.attack_boost,
            PokemonBoostableStat::Defense => self.defense_boost,
            PokemonBoostableStat::SpecialAttack => self.special_attack_boost,
            PokemonBoostableStat::SpecialDefense => self.special_defense_boost,
            PokemonBoostableStat::Speed => self.speed_boost,
            PokemonBoostableStat::Evasion => self.evasion_boost,
            PokemonBoostableStat::Accuracy => self.accuracy_boost,
        }
    }

    pub fn calculate_boosted_stat(&self, stat: PokemonBoostableStat) -> i16 {
        /*
        In Gen4, simple doubles the effective boost, without it visually being doubled
        It will not boost beyond an effective value of 6 though.
        */
        let active = self.get_active_immutable();
        match stat {
            PokemonBoostableStat::Attack => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.attack_boost * 2).min(6).max(-6)
                } else {
                    self.attack_boost
                };

                #[cfg(not(feature = "gen4"))]
                let boost = self.attack_boost;

                multiply_boost(boost, active.attack)
            }
            PokemonBoostableStat::Defense => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.defense_boost * 2).min(6).max(-6)
                } else {
                    self.defense_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.defense_boost;

                multiply_boost(boost, active.defense)
            }
            PokemonBoostableStat::SpecialAttack => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.special_attack_boost * 2).min(6).max(-6)
                } else {
                    self.special_attack_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.special_attack_boost;

                multiply_boost(boost, active.special_attack)
            }
            PokemonBoostableStat::SpecialDefense => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.special_defense_boost * 2).min(6).max(-6)
                } else {
                    self.special_defense_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.special_defense_boost;

                multiply_boost(boost, active.special_defense)
            }
            PokemonBoostableStat::Speed => {
                #[cfg(feature = "gen4")]
                let boost = if active.ability == Abilities::SIMPLE {
                    (self.speed_boost * 2).min(6).max(-6)
                } else {
                    self.speed_boost
                };
                #[cfg(not(feature = "gen4"))]
                let boost = self.speed_boost;

                multiply_boost(boost, active.speed)
            }
            _ => {
                panic!("Not implemented")
            }
        }
    }

    pub fn has_alive_non_rested_sleeping_pkmn(&self) -> bool {
        for p in self.pokemon.into_iter() {
            if p.status == PokemonStatus::SLEEP && p.hp > 0 && p.rest_turns == 0 {
                return true;
            }
        }
        false
    }

    #[cfg(not(feature = "terastallization"))]
    pub fn can_use_tera(&self) -> bool {
        false
    }

    #[cfg(feature = "terastallization")]
    pub fn can_use_tera(&self) -> bool {
        for p in self.pokemon.into_iter() {
            if p.terastallized {
                return false;
            }
        }
        true
    }

    pub fn add_switches(&self, vec: &mut Vec<MoveChoice>) {
        let mut iter = self.pokemon.into_iter();
        while let Some(p) = iter.next() {
            if p.hp > 0 && iter.pokemon_index != self.active_index {
                vec.push(MoveChoice::Switch(iter.pokemon_index));
            }
        }
        if vec.len() == 0 {
            vec.push(MoveChoice::None);
        }
    }

    pub fn trapped(&self, opponent_active: &Pokemon) -> bool {
        let active_pkmn = self.get_active_immutable();
        if self
            .volatile_statuses
            .contains(&PokemonVolatileStatus::LOCKEDMOVE)
            || self
                .volatile_statuses
                .contains(&PokemonVolatileStatus::NORETREAT)
        {
            return true;
        }
        if active_pkmn.item == Items::SHEDSHELL || active_pkmn.has_type(&PokemonType::GHOST) {
            return false;
        } else if self
            .volatile_statuses
            .contains(&PokemonVolatileStatus::PARTIALLYTRAPPED)
        {
            return true;
        } else if opponent_active.ability == Abilities::SHADOWTAG {
            return true;
        } else if opponent_active.ability == Abilities::ARENATRAP && active_pkmn.is_grounded() {
            return true;
        } else if opponent_active.ability == Abilities::MAGNETPULL
            && active_pkmn.has_type(&PokemonType::STEEL)
        {
            return true;
        }
        false
    }

    pub fn num_fainted_pkmn(&self) -> i8 {
        let mut count = 0;
        for p in self.pokemon.into_iter() {
            if p.hp == 0 {
                count += 1;
            }
        }
        count
    }
}

/// CB_BEST_USEFUL=1 enables the root no-op option filter (side-one only).
fn best_useful_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("CB_BEST_USEFUL").map_or(false, |v| v == "1" || v == "true")
    })
}

impl State {
    pub fn root_get_all_options(&self) -> (Vec<MoveChoice>, Vec<MoveChoice>) {
        if self.team_preview {
            let mut s1_options = Vec::with_capacity(6);
            let mut s2_options = Vec::with_capacity(6);

            let mut pkmn_iter = self.side_one.pokemon.into_iter();
            while let Some(_) = pkmn_iter.next() {
                if self.side_one.pokemon[pkmn_iter.pokemon_index].hp > 0 {
                    s1_options.push(MoveChoice::Switch(pkmn_iter.pokemon_index));
                }
            }
            let mut pkmn_iter = self.side_two.pokemon.into_iter();
            while let Some(_) = pkmn_iter.next() {
                if self.side_two.pokemon[pkmn_iter.pokemon_index].hp > 0 {
                    s2_options.push(MoveChoice::Switch(pkmn_iter.pokemon_index));
                }
            }
            return (s1_options, s2_options);
        }

        let (mut s1_options, mut s2_options) = self.get_all_options();

        if self.side_one.force_trapped {
            s1_options.retain(|x| match x {
                MoveChoice::Move(_) | MoveChoice::MoveTera(_) | MoveChoice::MoveMega(_) => true,
                MoveChoice::Switch(_) => false,
                MoveChoice::None => true,
            });
        }
        if self.side_one.slow_uturn_move {
            s1_options.clear();
            let encored = self
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            let taunted = self
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::TAUNT);
            self.side_one.get_active_immutable().add_available_moves(
                &mut s1_options,
                &self.side_one.last_used_move,
                encored,
                taunted,
                self.side_one.can_use_tera(),
            );
        }

        if self.side_two.force_trapped {
            s2_options.retain(|x| match x {
                MoveChoice::Move(_) | MoveChoice::MoveTera(_) | MoveChoice::MoveMega(_) => true,
                MoveChoice::Switch(_) => false,
                MoveChoice::None => true,
            });
        }
        if self.side_two.slow_uturn_move {
            s2_options.clear();
            let encored = self
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            let taunted = self
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::TAUNT);
            self.side_two.get_active_immutable().add_available_moves(
                &mut s2_options,
                &self.side_two.last_used_move,
                encored,
                taunted,
                self.side_two.can_use_tera(),
            );
        }

        if best_useful_enabled() {
            self.filter_side_one_noop_moves(&mut s1_options);
        }

        if s1_options.len() == 0 {
            s1_options.push(MoveChoice::None);
        }
        if s2_options.len() == 0 {
            s2_options.push(MoveChoice::None);
        }

        (s1_options, s2_options)
    }

    /// Drop side-one move options that are strict no-ops: moves the engine
    /// itself would resolve to "nothing happens" against every alive opponent
    /// and every response. In a flat position such a move's subtree is the
    /// best line delayed one tempo, indistinguishable to the eval, so MCTS
    /// soaks root visits on it. Three provable classes only:
    ///   - re-setting a hazard already at max layers
    ///   - a pure self-boost with every touched stat already clamped
    ///   - a pure status move that every alive opponent blocks (an immune
    ///     ACTIVE alone does not qualify — that is a legitimate
    ///     switch-prediction play; every reserve must be blocked too)
    /// Tera variants are never dropped (tera itself changes state), and if
    /// filtering would empty the list (trapped and locked into a no-op) the
    /// original options are kept.
    fn filter_side_one_noop_moves(&self, options: &mut Vec<MoveChoice>) {
        let active = self.side_one.get_active_immutable();
        let filtered: Vec<MoveChoice> = options
            .iter()
            .filter(|mc| match mc {
                MoveChoice::Move(idx) => !self.move_is_strict_noop(&active.moves[idx].choice),
                _ => true,
            })
            .copied()
            .collect();
        if !filtered.is_empty() && filtered.len() < options.len() {
            *options = filtered;
        }
    }

    fn move_is_strict_noop(&self, choice: &Choice) -> bool {
        if choice.category != MoveCategory::Status || choice.secondaries.is_some() {
            return false;
        }
        // exactly one primary effect: a second effect (or none encoded at
        // all — special-cased moves like Trick live outside these fields)
        // means "nothing happens" cannot be proven from here
        let effect_count = choice.boost.is_some() as u8
            + choice.status.is_some() as u8
            + choice.heal.is_some() as u8
            + choice.volatile_status.is_some() as u8
            + choice.side_condition.is_some() as u8;
        if effect_count != 1 {
            return false;
        }
        if let Some(sc) = &choice.side_condition {
            if sc.target != MoveTarget::Opponent {
                return false;
            }
            let theirs = &self.side_two.side_conditions;
            return match sc.condition {
                PokemonSideCondition::Stealthrock => theirs.stealth_rock >= 1,
                PokemonSideCondition::Spikes => theirs.spikes >= 3,
                PokemonSideCondition::ToxicSpikes => theirs.toxic_spikes >= 2,
                PokemonSideCondition::StickyWeb => theirs.sticky_web >= 1,
                _ => false,
            };
        }
        if let Some(boost) = &choice.boost {
            let user = self.side_one.get_active_immutable();
            if boost.target != MoveTarget::User
                || user.ability == Abilities::CONTRARY
                || (choice.move_id == Choices::CURSE && user.has_type(&PokemonType::GHOST))
            {
                return false;
            }
            let clamped = |amount: i8, current: i8| {
                amount == 0 || (amount > 0 && current >= 6) || (amount < 0 && current <= -6)
            };
            let b = &boost.boosts;
            let s = &self.side_one;
            return clamped(b.attack, s.attack_boost)
                && clamped(b.defense, s.defense_boost)
                && clamped(b.special_attack, s.special_attack_boost)
                && clamped(b.special_defense, s.special_defense_boost)
                && clamped(b.speed, s.speed_boost)
                && clamped(b.accuracy, s.accuracy_boost);
        }
        if let Some(status) = &choice.status {
            if status.target != MoveTarget::Opponent {
                return false;
            }
            let mut iter = self.side_two.pokemon.into_iter();
            while iter.next().is_some() {
                let idx = iter.pokemon_index;
                if self.side_two.pokemon[idx].hp > 0
                    && !self.s2_mon_blocks_status(idx, choice, &status.status)
                {
                    return false;
                }
            }
            return true;
        }
        false
    }

    /// Would `status` from side-one's active provably fail against this
    /// side-two mon, whether it is (or switches in as) the active? Mirrors
    /// the engine's own gates (move_has_no_effect + immune_to_status) so
    /// no-op-ness is engine-relative, not Showdown-relative.
    fn s2_mon_blocks_status(
        &self,
        idx: PokemonIndex,
        choice: &Choice,
        status: &PokemonStatus,
    ) -> bool {
        let pkmn = &self.side_two.pokemon[idx];
        if choice.move_type == PokemonType::ELECTRIC && pkmn.has_type(&PokemonType::GROUND) {
            return true;
        }
        #[cfg(any(feature = "gen6", feature = "gen7", feature = "gen8", feature = "gen9"))]
        if choice.flags.powder
            && (pkmn.has_type(&PokemonType::GRASS) || pkmn.item == Items::SAFETYGOGGLES)
        {
            return true;
        }
        if idx == self.side_two.active_index {
            return immune_to_status(self, &MoveTarget::Opponent, &SideReference::SideTwo, status);
        }
        // hypothetical switch-in: side-level volatiles (sub included) are gone
        let mut hypo = self.clone();
        hypo.side_two.active_index = idx;
        hypo.side_two.volatile_statuses.clear();
        hypo.side_two.substitute_health = 0;
        immune_to_status(&hypo, &MoveTarget::Opponent, &SideReference::SideTwo, status)
    }

    /// Every options list handed to MCTS must be non-empty: Node::expand
    /// indexes `[move_index]` and populate builds one child per entry, so an
    /// empty side panics (mcts.rs:240, "index 0 into len 0"). The final
    /// return of get_all_options guards this, but the force_switch early
    /// returns did NOT — a forced switch with no alive reserves (a faint
    /// with an empty bench, or an Eject/Red Card fire in a search branch
    /// that already lost its reserves) returned an empty vec and crashed the
    /// search. This restores the "no legal action" -> MoveChoice::None
    /// invariant on those paths.
    fn guarded_options(
        mut s1: Vec<MoveChoice>,
        mut s2: Vec<MoveChoice>,
    ) -> (Vec<MoveChoice>, Vec<MoveChoice>) {
        if s1.is_empty() {
            s1.push(MoveChoice::None);
        }
        if s2.is_empty() {
            s2.push(MoveChoice::None);
        }
        (s1, s2)
    }

    pub fn get_all_options(&self) -> (Vec<MoveChoice>, Vec<MoveChoice>) {
        let mut side_one_options: Vec<MoveChoice> = Vec::with_capacity(9);
        let mut side_two_options: Vec<MoveChoice> = Vec::with_capacity(9);

        let side_one_active = self.side_one.get_active_immutable();
        let side_two_active = self.side_two.get_active_immutable();

        if self.side_one.force_switch {
            self.side_one.add_switches(&mut side_one_options);
            if self.side_two.switch_out_move_second_saved_move == Choices::NONE {
                side_two_options.push(MoveChoice::None);
            } else {
                self.side_two.get_active_immutable().add_move_from_choice(
                    &mut side_two_options,
                    self.side_two.switch_out_move_second_saved_move,
                );
            }
            return Self::guarded_options(side_one_options, side_two_options);
        }

        if self.side_two.force_switch {
            self.side_two.add_switches(&mut side_two_options);
            if self.side_one.switch_out_move_second_saved_move == Choices::NONE {
                side_one_options.push(MoveChoice::None);
            } else {
                self.side_one.get_active_immutable().add_move_from_choice(
                    &mut side_one_options,
                    self.side_one.switch_out_move_second_saved_move,
                );
            }
            return Self::guarded_options(side_one_options, side_two_options);
        }

        let side_one_force_switch = self.side_one.get_active_immutable().hp <= 0;
        let side_two_force_switch = self.side_two.get_active_immutable().hp <= 0;

        if side_one_force_switch && side_two_force_switch {
            self.side_one.add_switches(&mut side_one_options);
            self.side_two.add_switches(&mut side_two_options);
            return Self::guarded_options(side_one_options, side_two_options);
        }
        if side_one_force_switch {
            self.side_one.add_switches(&mut side_one_options);
            side_two_options.push(MoveChoice::None);
            return Self::guarded_options(side_one_options, side_two_options);
        }
        if side_two_force_switch {
            side_one_options.push(MoveChoice::None);
            self.side_two.add_switches(&mut side_two_options);
            return Self::guarded_options(side_one_options, side_two_options);
        }

        if self
            .side_one
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE)
        {
            side_one_options.push(MoveChoice::None);
        } else if let Some(mv_index) = self.side_one.active_is_charging_move() {
            side_one_options.push(MoveChoice::Move(mv_index));
        } else {
            let encored = self
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            let taunted = self
                .side_one
                .volatile_statuses
                .contains(&PokemonVolatileStatus::TAUNT);
            self.side_one.get_active_immutable().add_available_moves(
                &mut side_one_options,
                &self.side_one.last_used_move,
                encored,
                taunted,
                self.side_one.can_use_tera(),
            );
            if !self.side_one.trapped(side_two_active) {
                self.side_one.add_switches(&mut side_one_options);
            }
        }

        if self
            .side_two
            .volatile_statuses
            .contains(&PokemonVolatileStatus::MUSTRECHARGE)
        {
            side_two_options.push(MoveChoice::None);
        } else if let Some(mv_index) = self.side_two.active_is_charging_move() {
            side_two_options.push(MoveChoice::Move(mv_index));
        } else {
            let encored = self
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::ENCORE);
            let taunted = self
                .side_two
                .volatile_statuses
                .contains(&PokemonVolatileStatus::TAUNT);
            self.side_two.get_active_immutable().add_available_moves(
                &mut side_two_options,
                &self.side_two.last_used_move,
                encored,
                taunted,
                self.side_two.can_use_tera(),
            );
            if !self.side_two.trapped(side_one_active) {
                self.side_two.add_switches(&mut side_two_options);
            }
        }

        if side_one_options.len() == 0 {
            side_one_options.push(MoveChoice::None);
        }
        if side_two_options.len() == 0 {
            side_two_options.push(MoveChoice::None);
        }

        (side_one_options, side_two_options)
    }

    pub fn reset_toxic_count(
        &mut self,
        side_ref: &SideReference,
        vec_to_add_to: &mut Vec<Instruction>,
    ) {
        let side = self.get_side(side_ref);
        if side.side_conditions.toxic_count > 0 {
            vec_to_add_to.push(Instruction::ChangeSideCondition(
                ChangeSideConditionInstruction {
                    side_ref: *side_ref,
                    side_condition: PokemonSideCondition::ToxicCount,
                    amount: -1 * side.side_conditions.toxic_count,
                },
            ));
            side.side_conditions.toxic_count = 0;
        }
    }

    pub fn remove_volatile_statuses_on_switch(
        &mut self,
        side_ref: &SideReference,
        instructions: &mut Vec<Instruction>,
        baton_passing: bool,
        shed_tailing: bool,
    ) {
        let side = self.get_side(side_ref);

        // Take ownership of the current set to avoid borrow conflicts
        // since we may need to modify the side in the loop
        let mut volatile_statuses = std::mem::take(&mut side.volatile_statuses);

        volatile_statuses.retain(|pkmn_volatile_status| {
            let should_retain = match pkmn_volatile_status {
                PokemonVolatileStatus::SUBSTITUTE => baton_passing || shed_tailing,
                PokemonVolatileStatus::LEECHSEED => baton_passing,
                PokemonVolatileStatus::TYPECHANGE => {
                    let active = side.get_active();
                    if active.base_types != active.types {
                        instructions.push(Instruction::ChangeType(ChangeType {
                            side_ref: *side_ref,
                            new_types: active.base_types,
                            old_types: active.types,
                        }));
                        active.types = active.base_types;
                    }
                    false
                }
                // While you can't switch out of a locked move you can be forced out in other ways
                PokemonVolatileStatus::LOCKEDMOVE => {
                    instructions.push(Instruction::ChangeVolatileStatusDuration(
                        ChangeVolatileStatusDurationInstruction {
                            side_ref: *side_ref,
                            volatile_status: *pkmn_volatile_status,
                            amount: -1 * side.volatile_status_durations.lockedmove,
                        },
                    ));
                    side.volatile_status_durations.lockedmove = 0;
                    false
                }
                PokemonVolatileStatus::YAWN => {
                    instructions.push(Instruction::ChangeVolatileStatusDuration(
                        ChangeVolatileStatusDurationInstruction {
                            side_ref: *side_ref,
                            volatile_status: *pkmn_volatile_status,
                            amount: -1 * side.volatile_status_durations.yawn,
                        },
                    ));
                    side.volatile_status_durations.yawn = 0;
                    false
                }
                PokemonVolatileStatus::TAUNT => {
                    instructions.push(Instruction::ChangeVolatileStatusDuration(
                        ChangeVolatileStatusDurationInstruction {
                            side_ref: *side_ref,
                            volatile_status: *pkmn_volatile_status,
                            amount: -1 * side.volatile_status_durations.taunt,
                        },
                    ));
                    side.volatile_status_durations.taunt = 0;
                    false
                }
                _ => false,
            };

            if !should_retain {
                instructions.push(Instruction::RemoveVolatileStatus(
                    RemoveVolatileStatusInstruction {
                        side_ref: *side_ref,
                        volatile_status: *pkmn_volatile_status,
                    },
                ));
            }
            should_retain
        });

        // Clean up by re-setting the volatile statuses
        side.volatile_statuses = volatile_statuses;
    }

    pub fn terrain_is_active(&self, terrain: &Terrain) -> bool {
        &self.terrain.terrain_type == terrain && self.terrain.turns_remaining > 0
    }

    pub fn get_terrain(&self) -> Terrain {
        if self.terrain.turns_remaining > 0 {
            self.terrain.terrain_type
        } else {
            Terrain::NONE
        }
    }

    pub fn weather_is_active(&self, weather: &Weather) -> bool {
        let s1_active = self.side_one.get_active_immutable();
        let s2_active = self.side_two.get_active_immutable();
        &self.weather.weather_type == weather
            && s1_active.ability != Abilities::AIRLOCK
            && s1_active.ability != Abilities::CLOUDNINE
            && s2_active.ability != Abilities::AIRLOCK
            && s2_active.ability != Abilities::CLOUDNINE
    }

    fn _state_contains_any_move(&self, moves: &[Choices]) -> bool {
        for s in [&self.side_one, &self.side_two] {
            for pkmn in s.pokemon.into_iter() {
                for mv in pkmn.moves.into_iter() {
                    if moves.contains(&mv.id) {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub fn set_damage_dealt_flag(&mut self) {
        if self._state_contains_any_move(&[
            Choices::COUNTER,
            Choices::MIRRORCOAT,
            Choices::METALBURST,
            Choices::COMEUPPANCE,
            Choices::FOCUSPUNCH,
            Choices::AVALANCHE,
        ]) {
            self.use_damage_dealt = true
        }
    }

    pub fn set_last_used_move_flag(&mut self) {
        if self._state_contains_any_move(&[
            Choices::ENCORE,
            Choices::FAKEOUT,
            Choices::FIRSTIMPRESSION,
            Choices::BLOODMOON,
            Choices::GIGATONHAMMER,
        ]) {
            self.use_last_used_move = true
        }
    }

    pub fn set_conditional_mechanics(&mut self) {
        /*
        These mechanics are not always relevant but when they are it
        is important that they are enabled. Enabling them all the time would
        suffer about a 20% performance hit.
        */
        self.set_damage_dealt_flag();
        self.set_last_used_move_flag();
    }
}

#[cfg(test)]
mod best_useful_tests {
    use super::*;
    use crate::choices::MOVES;
    use crate::state::{PokemonMoveIndex, State};

    const ALL: [PokemonIndex; 6] = [
        PokemonIndex::P0,
        PokemonIndex::P1,
        PokemonIndex::P2,
        PokemonIndex::P3,
        PokemonIndex::P4,
        PokemonIndex::P5,
    ];

    fn set_move(state: &mut State, slot: PokemonMoveIndex, id: Choices) {
        let active = state.side_one.get_active();
        active.replace_move(slot, id);
        active.moves[&slot].pp = 16;
    }

    /// Toxic with a statused ACTIVE but healthy reserves is switch-prediction
    /// and must survive; once every alive opponent is statused it is provable.
    #[test]
    fn toxic_dropped_only_when_every_alive_opponent_blocks() {
        let mut state = State::default();
        state.side_two.get_active().status = PokemonStatus::TOXIC;
        let toxic = MOVES.get(&Choices::TOXIC).unwrap();
        assert!(!state.move_is_strict_noop(toxic));
        for idx in ALL {
            state.side_two.pokemon[idx].status = PokemonStatus::POISON;
        }
        assert!(state.move_is_strict_noop(toxic));
    }

    /// Fainted reserves cannot be status targets; only alive mons gate.
    #[test]
    fn dead_reserves_do_not_keep_toxic_alive() {
        let mut state = State::default();
        state.side_two.get_active().status = PokemonStatus::TOXIC;
        for idx in ALL {
            if idx != state.side_two.active_index {
                state.side_two.pokemon[idx].hp = 0;
            }
        }
        assert!(state.move_is_strict_noop(MOVES.get(&Choices::TOXIC).unwrap()));
    }

    /// Thunder Wave vs an all-Ground side mirrors the engine's
    /// electric-vs-ground gate; one non-Ground reserve keeps it.
    #[test]
    fn thunder_wave_blocked_by_all_ground_side() {
        let mut state = State::default();
        for idx in ALL {
            state.side_two.pokemon[idx].types = (PokemonType::GROUND, PokemonType::TYPELESS);
        }
        let twave = MOVES.get(&Choices::THUNDERWAVE).unwrap();
        assert!(state.move_is_strict_noop(twave));
        state.side_two.pokemon[PokemonIndex::P4].types =
            (PokemonType::NORMAL, PokemonType::TYPELESS);
        assert!(!state.move_is_strict_noop(twave));
    }

    /// Calm Mind is a no-op only with BOTH touched stats clamped at +6,
    /// and never for a Contrary user.
    #[test]
    fn calm_mind_dropped_at_plus_six_only() {
        let mut state = State::default();
        let cm = MOVES.get(&Choices::CALMMIND).unwrap();
        state.side_one.special_attack_boost = 6;
        state.side_one.special_defense_boost = 5;
        assert!(!state.move_is_strict_noop(cm));
        state.side_one.special_defense_boost = 6;
        assert!(state.move_is_strict_noop(cm));
        state.side_one.get_active().ability = Abilities::CONTRARY;
        assert!(!state.move_is_strict_noop(cm));
    }

    /// Re-setting a maxed hazard is a no-op; below the cap it is not.
    #[test]
    fn maxed_hazards_dropped() {
        let mut state = State::default();
        let rocks = MOVES.get(&Choices::STEALTHROCK).unwrap();
        let spikes = MOVES.get(&Choices::SPIKES).unwrap();
        assert!(!state.move_is_strict_noop(rocks));
        state.side_two.side_conditions.stealth_rock = 1;
        assert!(state.move_is_strict_noop(rocks));
        state.side_two.side_conditions.spikes = 2;
        assert!(!state.move_is_strict_noop(spikes));
        state.side_two.side_conditions.spikes = 3;
        assert!(state.move_is_strict_noop(spikes));
    }

    /// The option filter drops the no-op Move but keeps switches and tera
    /// variants, and falls back to the original list rather than emptying it.
    #[test]
    fn filter_keeps_tera_and_never_empties() {
        let mut state = State::default();
        set_move(&mut state, PokemonMoveIndex::M0, Choices::TOXIC);
        set_move(&mut state, PokemonMoveIndex::M1, Choices::TACKLE);
        for idx in ALL {
            state.side_two.pokemon[idx].status = PokemonStatus::POISON;
        }
        let mut options = vec![
            MoveChoice::Move(PokemonMoveIndex::M0),
            MoveChoice::MoveTera(PokemonMoveIndex::M0),
            MoveChoice::Move(PokemonMoveIndex::M1),
        ];
        state.filter_side_one_noop_moves(&mut options);
        assert_eq!(
            options,
            vec![
                MoveChoice::MoveTera(PokemonMoveIndex::M0),
                MoveChoice::Move(PokemonMoveIndex::M1),
            ]
        );
        let mut only_noop = vec![MoveChoice::Move(PokemonMoveIndex::M0)];
        state.filter_side_one_noop_moves(&mut only_noop);
        assert_eq!(only_noop, vec![MoveChoice::Move(PokemonMoveIndex::M0)]);
    }
}
