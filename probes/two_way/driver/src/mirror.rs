//! Where a project file is written in the shadow tree. A file reached through include directory
//! `X` as `X/rel` goes to `shadow/<index of X>/rel`, and `shadow/<index of X>` is put before `X`,
//! so every include spelling that found the original finds the rewritten file first.

use std::path::{Path, PathBuf};

/// The directories files are mirrored under: the include directories, then the roots.
#[derive(Debug)]
pub struct Mirror {
    bases: Vec<PathBuf>,
    include_count: usize,
}

impl Mirror {
    /// A mirror over the include directories (in order) and the roots.
    #[must_use]
    pub fn new(includes: &[PathBuf], roots: &[PathBuf]) -> Self {
        let bases = includes.iter().chain(roots).cloned().collect();
        Mirror {
            bases,
            include_count: includes.len(),
        }
    }

    /// Every `(index, relative path)` a file is written at: under each include directory that
    /// holds it, or, when none does, under the first root that holds it.
    #[must_use]
    pub fn placements(&self, file: &Path) -> Vec<(usize, PathBuf)> {
        let under = |i: usize| {
            file.strip_prefix(&self.bases[i])
                .ok()
                .map(|rel| (i, rel.to_path_buf()))
        };
        let in_includes: Vec<_> = (0..self.include_count).filter_map(under).collect();
        if !in_includes.is_empty() {
            return in_includes;
        }
        (self.include_count..self.bases.len())
            .find_map(under)
            .into_iter()
            .collect()
    }

    /// The path of a placement inside the shadow folder.
    #[must_use]
    pub fn shadow_path(shadow: &Path, placement: &(usize, PathBuf)) -> PathBuf {
        shadow.join(placement.0.to_string()).join(&placement.1)
    }

    /// The include flags that put each shadow directory before its original: `-I shadow/i -I X`.
    #[must_use]
    pub fn include_flags(&self, shadow: &Path) -> Vec<String> {
        let mut flags = Vec::new();
        for (i, dir) in self.bases.iter().take(self.include_count).enumerate() {
            flags.push(format!("-I{}", shadow.join(i.to_string()).display()));
            flags.push(format!("-I{}", dir.display()));
        }
        flags
    }
}
