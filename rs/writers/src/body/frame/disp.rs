//! The disposition of every position, computed once before it renders, and the one function that
//! tells whether a rendered value is named, in each of the three scopes.
use crate::body::frame::table::Disp;
use crate::body::{placed, Class, Frame, OpKind};
use crate::refuse::Refusal;

/// Where a value is being named.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// The top level of the walk.
    Top,
    /// The body of a loop.
    Loop,
    /// A case of a switch.
    Case,
}

impl Frame<'_> {
    /// Computes the disposition of the positions `from..to`. `per_part` marks the positions the
    /// commit plan stores per component (only the walk's own positions have an entry; the slots of
    /// a loop body never do).
    pub(crate) fn settle_disp(&mut self, from: usize, to: usize, per_part: &[bool]) {
        for p in from..to {
            let h = self.node(p).h;
            let class = self.node(p).class.clone();
            let literal = matches!(class, Class::Const | Class::RoleConst);
            let disp = Disp::of(
                matches!(
                    class,
                    Class::Assign
                        | Class::WarpStore
                        | Class::SampleSum
                        | Class::AccumWrite
                        | Class::Finish
                        | Class::TapeWrite
                ),
                placed::is_absorbed(self, p) && !literal,
                self.stores.skip.contains(&h),
                per_part.get(p) == Some(&true),
                self.stores.inline.contains(&h),
                self.is_forced(p),
            );
            self.set_disp(p, disp);
        }
    }

    /// Whether the value of position `p` gets a local of its own. The three scopes keep their own
    /// rules: the top level counts the node's reads and what forces it; a loop body counts its
    /// body reads; a case counts its case tally and ignores the inline mark.
    pub(crate) fn naming(
        &self,
        p: usize,
        scope: ScopeKind,
        role: Option<&str>,
        readers: usize,
    ) -> Result<bool, Refusal> {
        let class = &self.node(p).class;
        let leaf_valued = self.leaf_valued(p)?;
        let inline = self.disp(p) == Disp::Inline;
        let proxy_read =
            self.node(p).kind == Some(OpKind::Component) && self.is_lazy(self.opp(p)[0]);
        Ok(match scope {
            ScopeKind::Top => {
                let crossed = role == Some("mat_in")
                    && !self.users(p).is_empty()
                    && self
                        .users(p)
                        .iter()
                        .all(|&(u, k)| k == 0 && self.node(u).class == Class::At);
                !matches!(class, Class::Const | Class::RoleConst | Class::SampleIndex)
                    && !matches!(role, Some("uniform" | "nsamples"))
                    && !matches!(role, Some("lookup" | "wide_in"))
                    && !crossed
                    && !proxy_read
                    && !inline
                    && (readers >= 2 || leaf_valued || self.disp(p) == Disp::Forced)
            }
            ScopeKind::Loop => !proxy_read && !inline && (readers >= 2 || leaf_valued),
            ScopeKind::Case => {
                !matches!(class, Class::Const | Class::RoleConst | Class::SampleIndex)
                    && !matches!(role, Some("uniform" | "nsamples" | "lookup" | "wide_in"))
                    && (readers >= 2 || leaf_valued)
            }
        })
    }
}
