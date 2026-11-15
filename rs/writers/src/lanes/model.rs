//! The data model of the lane features: a loop cut out of a body, a body cut at its interchanged
//! loops (lane segments) and a lane split. Each holds buffers (lines with the declarations and
//! marks they carry), not bare text, so that a reader asks the records and never rescans text.
use crate::body::{Buffer, Ty};

/// One carried slot of a cut loop.
#[derive(Clone)]
pub struct Carry {
    /// The slot's name.
    pub name: String,
    /// The slot's type.
    pub ty: Ty,
    /// The text of its initial value.
    pub init: String,
    /// The name of its next value.
    pub next: String,
    /// The declared names its initial value reads.
    pub init_reads: std::collections::BTreeSet<String>,
}

/// One loop cut out of the body: where the segment before it ends, one trip of it for one lane,
/// and what the arm needs to
/// run its trips outside and the lanes inside.
pub struct CutLoop {
    /// The number of lines emitted before the loop.
    pub mark: usize,
    /// One trip of the loop, rendered for one lane.
    pub(crate) step: Buffer,
    /// Per carried slot: its name, type, the text of its initial value and the name of its
    /// next value.
    pub carries: Vec<Carry>,
    /// The name of the loop's index.
    pub index: String,
    /// The index's first value.
    pub start: i64,
    /// The index's step.
    pub by: i64,
    /// The static trip count.
    pub trips: i64,
    /// The name of the break flag, or `false` for a loop without a break.
    pub exit: String,
    /// Whether one trip may run for a stopped lane and be discarded by a select.
    pub speculative: bool,
    /// A runtime count's header: index type, begin, comparison, end.
    pub bounds: Option<(String, String, String, String)>,
    /// The declared names the runtime count's text reads.
    pub bounds_reads: std::collections::BTreeSet<String>,
    /// The runtime trip count's text of a split loop.
    pub count: Option<String>,
}

/// A body cut at the interchanged loops: the segments around them, each loop, the types of the
/// names a later text reads.
pub struct LaneSegments {
    /// The body-wide mask condition, when the kernel is guarded; no segment contains the guard.
    pub guard: Option<String>,
    /// The text before the first cut, between cuts and after the last.
    pub(crate) segments: Vec<Buffer>,
    /// The cut loops, in order.
    pub loops: Vec<CutLoop>,
    /// The type of each name a later text reads.
    pub types: Vec<(String, Ty)>,
    /// The planes the placements write, as (proxy, plane, width): rows of the tile the arm declares.
    pub proxies: Vec<(String, String, usize)>,
}

/// The head's declarations of the values no sample changes: what a wrapper evaluates once at entry. Empty when a live value
/// that reads nothing per sample is not a one-line declaration.
#[derive(Default)]
pub struct SampleFree {
    /// The indexes in `LaneSplit::pre` of the one-line declarations, in the head's order.
    pub at: Vec<usize>,
    /// The names those declarations introduce.
    pub names: Vec<String>,
}

/// A fused-step body cut at its one loop: `pre` and `post` exclude the guard, `step` is one trip.
pub struct LaneSplit {
    /// The body-wide mask condition, when the kernel is guarded.
    pub guard: Option<String>,
    /// The text before the loop, without the guard.
    pub(crate) pre: Buffer,
    /// One trip of the loop.
    pub(crate) step: Buffer,
    /// The text after the loop, without the guard.
    pub(crate) post: Buffer,
    /// Per carried slot: its name, type, the text of its initial value and the name of its
    /// next value.
    pub carries: Vec<Carry>,
    /// The values the head names and one trip reads: their names and types.
    pub live_ins: Vec<(String, Ty)>,
    /// The runtime trip count's text, `None` for a static bound.
    pub count: Option<String>,
    /// The loop's static trip bound.
    pub stop: i64,
    /// The name of the loop's index.
    pub index: String,
    /// The name of the break flag.
    pub exit: String,
    /// Whether one trip may run for a stopped lane and be discarded.
    pub speculative: bool,
    /// The operations of the loop's own body nodes.
    pub op_count: usize,
    /// The carry the exit condition compares to a sample-free bound, and whether it rises
    /// towards it.
    pub exit_order: Option<(usize, bool)>,
    /// The live values that read nothing per sample and the values they read, as the one-line declarations of `pre`: a wrapper
    /// may evaluate them once instead of carrying them.
    pub sample_free: SampleFree,
}
