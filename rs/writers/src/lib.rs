//! The writers: C++ source text for numeric kernels, written from a recorded node walk.
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
pub mod body;
pub mod contract;
pub mod ir;
pub mod lanes;
pub mod refuse;
pub mod spell;
pub mod unit;

pub use body::render_body;
pub use ir::door;
pub use ir::door::RxCtx;
pub use lanes::{lane_segments, lane_split};
pub use refuse::{Refusal, RefusalKind};
pub use unit::device as cuda;
pub use unit::entry::{cuda_render, entry_body, host_render, HostSource};
#[cfg(feature = "old-contract")]
pub use unit::team_stage::is_team_stage;
pub use unit::team_stage::stage_body;
