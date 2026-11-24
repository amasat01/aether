//! What the text of a body needs from the unit around it: the helper functions, headers and
//! preludes its calls name. The code that
//! writes a helper call records the need at that moment; the unit code reads the set instead of
//! searching the finished text.
//! `Needs::scan` is the one place that still searches text, for a body the caller gave (it has no
//! records) and for the shadow check.
use std::collections::BTreeSet;

/// One helper, header or prelude a rendered body may call for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Need {
    /// `hawk_abi::store_compensated(`: a store through a correction plane.
    StoreCompensated,
    /// `hawk_abi::tiled<`: the tiled load helper, named by the read of a placed plane on a device.
    Tiled,
    /// `hawk_abi::warp_sum<`: the warp-wide sum helper, named by the sum over the lanes of a warp.
    WarpSum,
    /// `hawk_abi::spacing(`: the spacing helper.
    Spacing,
    /// `hawk_abi::mix_count(`: the mix counter.
    MixCount,
    /// `hawk_abi::int_`: an integer helper.
    Integers,
    /// `aether::linalg::`: the linear-algebra header.
    Linalg,
    /// `aether::rt::`: the run-time header.
    Rt,
    /// `aether::geometry::`: the geometry header.
    Geometry,
    /// `aether::linalg::team::`: the team header.
    Team,
    /// `hawk_abi::accum_or(`: the scattered or.
    AccumOr,
    /// `hawk_abi::accum_or_at(`: the scattered or at an element of a plane with axes.
    AccumOrAt,
    /// `hawk_abi::view_item<`: a value of rank above zero read out of a plane with axes.
    ViewItem,
    /// `aether::edge::`: the edge-rule helpers of an indexed access.
    Edge,
    /// `hawk_abi::accum_add_at(`: the scattered add at an element of a plane with axes.
    AccumAt,
    /// `hawk_abi::accum_add_compensated_at(`: the compensated scattered add at an element of a plane with axes.
    AccumCompensatedAt,
    /// `hawk_abi::accum_add_compensated(`: the compensated scattered add.
    AccumCompensated,
    /// `aether::accum::run::`: a run of commits against one target.
    AccumRun,
    /// `aether::accum::atomic::`: an atomic update of one element by a named library helper (min, max, and, xor, exchange,
    /// compare-exchange, fetch-add).
    ElementAtomic,
    /// `aether::accum::run::add...Compensated...`: a run of commits that carries a correction
    /// plane.
    CompensatedRun,
    /// `aether::random::`: the random header.
    Random,
    /// `hawk_abi::blend(`: the scalar blend helper.
    Blend,
    /// `hawk_abi::select_item<`: the item select helper.
    SelectItem,
    /// `hawk_abi::tile_lane<`: the lane tile helper.
    TileLane,
}

/// The needs of one body: the set, and the run states in the order the text first uses them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// The helpers, headers and preludes the text calls for.
    pub set: BTreeSet<Need>,
    /// The run states of the body's run commits, once each, in order of first use.
    pub runs: Vec<RunState>,
}

/// One open run of a commit: a lane's commits to the same element merge into it, a commit to another element flushes it. A target
/// that is committed at several elements by one lane (the components of a vector, two indices) holds one state per element, or
/// every commit would flush the run before it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RunState {
    /// The binding of the plane the run commits into.
    pub target: String,
    /// The text of the element the run commits to (what tells two states of one target apart).
    pub element: String,
    /// The state's name after `hawk_run_`: the target for its first element, `<target>__<n>` for the next.
    pub state: String,
}

/// The most run states one target holds in a body. A lane that commits to more elements than this (a long vector) commits the rest
/// without a run, so the entry's frame stays small: a state is four words.
pub const MAX_RUN_STATES: usize = 32;

/// A body's text with the needs recorded while it was written.
#[derive(Clone, Debug, Default)]
pub struct Rendered {
    /// The text of the body.
    pub text: String,
    /// The needs of that text.
    pub needs: Needs,
}

impl Needs {
    /// Records a need at the place its helper call is written.
    pub fn record(&mut self, need: Need) {
        self.set.insert(need);
    }

    /// Records the helpers the spelled text of an operation calls.
    pub fn record_spelled(&mut self, used: &[crate::spell::Helper]) {
        use crate::spell::Helper;
        for h in used {
            self.record(match h {
                Helper::Integers => Need::Integers,
                Helper::Random => Need::Random,
                Helper::Spacing => Need::Spacing,
                Helper::Blend => Need::Blend,
                Helper::SelectItem => Need::SelectItem,
            })
        }
    }

    /// The run state of `target` at the element text `element`, recorded on first use; none when the target already holds
    /// `MAX_RUN_STATES` (the commit then goes without a run).
    pub fn record_run(&mut self, target: &str, element: &str) -> Option<String> {
        if let Some(known) = self
            .runs
            .iter()
            .find(|r| r.target == target && r.element == element)
        {
            return Some(known.state.clone());
        }
        let held = self.runs.iter().filter(|r| r.target == target).count();
        if held >= MAX_RUN_STATES {
            return None;
        }
        let state = if held == 0 {
            target.to_string()
        } else {
            // the count leads: a target name never starts with a digit, so no other target can spell this state
            format!("{held}_{target}")
        };
        self.runs.push(RunState {
            target: target.to_string(),
            element: element.to_string(),
            state: state.clone(),
        });
        Some(state)
    }

    /// Records a state another body or the text already names, once.
    fn push_state(&mut self, run: &RunState) {
        if !self.runs.iter().any(|r| r.state == run.state) {
            self.runs.push(run.clone())
        }
    }

    /// Whether the need was recorded.
    pub fn has(&self, need: Need) -> bool {
        self.set.contains(&need)
    }

    /// Adds what another body needs; its run states follow this one's that are not here yet.
    pub fn merge(&mut self, other: &Needs) {
        self.set.extend(other.set.iter().copied());
        for run in &other.runs {
            self.push_state(run)
        }
    }

    /// The needs a finished text shows by its substrings: the only search of text left. It serves
    /// a body the caller gave, which has no records, and the shadow check of the recorded needs.
    pub fn scan(text: &str) -> Needs {
        let tests: [(Need, &str); 20] = [
            (Need::StoreCompensated, "hawk_abi::store_compensated("),
            (Need::Tiled, "hawk_abi::tiled<"),
            (Need::WarpSum, "hawk_abi::warp_sum<"),
            (Need::Spacing, "hawk_abi::spacing("),
            (Need::MixCount, "hawk_abi::mix_count("),
            (Need::Integers, "hawk_abi::int_"),
            (Need::Linalg, "aether::linalg::"),
            (Need::Rt, "aether::rt::"),
            (Need::Geometry, "aether::geometry::"),
            (Need::Team, "aether::linalg::team::"),
            (Need::AccumOr, "hawk_abi::accum_or("),
            (Need::AccumOrAt, "hawk_abi::accum_or_at("),
            (Need::ViewItem, "hawk_abi::view_item<"),
            (Need::Edge, "aether::edge::"),
            (Need::AccumAt, "hawk_abi::accum_add_at("),
            (
                Need::AccumCompensatedAt,
                "hawk_abi::accum_add_compensated_at(",
            ),
            (Need::AccumCompensated, "hawk_abi::accum_add_compensated("),
            (Need::AccumRun, "aether::accum::run::"),
            (Need::ElementAtomic, "aether::accum::atomic::"),
            (Need::Random, "aether::random::"),
        ];
        let mut needs = Needs::default();
        for (need, sub) in tests {
            if text.contains(sub) {
                needs.record(need)
            }
        }
        if text.contains("hawk_abi::blend(") {
            needs.record(Need::Blend)
        }
        if text.contains("hawk_abi::select_item<") {
            needs.record(Need::SelectItem)
        }
        if text.contains("hawk_abi::tile_lane<") {
            needs.record(Need::TileLane)
        }
        if text.contains("aether::accum::run::add") && text.contains("Compensated") {
            needs.record(Need::CompensatedRun)
        }
        for (i, _) in text.match_indices("hawk_run_") {
            if text[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
            {
                continue;
            }
            let rest = &text[i + 9..];
            let word: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !word.is_empty() && rest[word.len()..].starts_with(");") {
                needs.push_state(&RunState {
                    state: word,
                    ..RunState::default()
                })
            }
        }
        needs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scan finds a helper by its call and the run states in first-use order.
    #[test]
    fn scan_reads_calls_and_runs() {
        let text = "x = hawk_abi::blend(c, a, b);\naether::accum::run::addCompensated<4>(t, hawk_run_b);\nf(hawk_run_a);\ng(hawk_run_b);";
        let n = Needs::scan(text);
        assert!(n.has(Need::Blend) && n.has(Need::CompensatedRun) && n.has(Need::AccumRun));
        assert!(!n.has(Need::Rt));
        let states: Vec<&str> = n.runs.iter().map(|r| r.state.as_str()).collect();
        assert_eq!(states, ["b", "a"]);
    }

    /// A target committed at two elements holds two states, the same element again the state it had, and a target past
    /// `MAX_RUN_STATES` elements none: that commit goes without a run.
    #[test]
    fn a_state_is_kept_per_element_of_a_target_up_to_the_bound() {
        let mut n = Needs::default();
        assert_eq!(n.record_run("t", "t(0, k)").as_deref(), Some("t"));
        assert_eq!(n.record_run("t", "t(1, k)").as_deref(), Some("1_t"));
        assert_eq!(n.record_run("t", "t(0, k)").as_deref(), Some("t"));
        assert_eq!(n.record_run("u", "u(0)").as_deref(), Some("u"));
        for k in 2..MAX_RUN_STATES {
            assert!(n.record_run("t", &format!("t({k}, k)")).is_some());
        }
        assert_eq!(n.record_run("t", "t(past, k)"), None);
    }
}
