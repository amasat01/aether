//! Wide writes: values a placement writes into a plane element by element.
use super::{Frame, Out};
use crate::refuse::nib;

/// A wide write: refused by name. Its only recorded units are team stages, whose text another
/// module composes, and no
/// recorded unit exercises a write outside a team stage, so nothing here is written unverified.
pub fn stmt(f: &mut Frame, _p: usize) -> Result<Out, String> {
    if f.unit.team.is_some() {
        return Err(nib("a wide write of a team stage"));
    }
    Err(nib("a wide write outside a team stage (no recorded unit)"))
}
