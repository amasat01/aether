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
    /// `hawk_abi::tiled<`: the tiled load helper. No writer emits this call; only a body the
    /// caller gave can name it, so only
    /// `Needs::scan` ever sets it.
    Tiled,
    /// `hawk_abi::warp_sum<`: the warp-wide sum helper. No writer emits this call; only a body the
    /// caller gave can name it, so only `Needs::scan` ever sets it.
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
    /// `hawk_abi::accum_add_compensated(`: the compensated scattered add.
    AccumCompensated,
    /// `aether::accum::run::`: a run of commits against one target.
    AccumRun,
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
    /// The targets the body's run commits name, once each, in order of first use.
    pub runs: Vec<String>,
}

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

    /// Records a run state, once, in order of first use.
    pub fn record_run(&mut self, target: &str) {
        if !self.runs.iter().any(|t| t == target) {
            self.runs.push(target.to_string())
        }
    }

    /// Whether the need was recorded.
    pub fn has(&self, need: Need) -> bool {
        self.set.contains(&need)
    }

    /// Adds what another body needs; its run states follow this one's that are not here yet.
    pub fn merge(&mut self, other: &Needs) {
        self.set.extend(other.set.iter().copied());
        for t in &other.runs {
            self.record_run(t)
        }
    }

    /// The needs a finished text shows by its substrings: the only search of text left. It serves
    /// a body the caller gave, which has no records, and the shadow check of the recorded needs.
    pub fn scan(text: &str) -> Needs {
        let tests: [(Need, &str); 14] = [
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
            (Need::AccumCompensated, "hawk_abi::accum_add_compensated("),
            (Need::AccumRun, "aether::accum::run::"),
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
                needs.record_run(&word)
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
        assert_eq!(n.runs, vec!["b".to_string(), "a".to_string()]);
    }
}
