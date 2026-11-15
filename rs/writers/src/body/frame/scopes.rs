//! The scopes of a frame: the only code that pushes to or truncates the per-position tables, the
//! slot map or the line buffer. A loop scope appends slots for its index, carries and body and
//! drops them on close; a case scope restores the texts and scoped flags (not the lazy or forced
//! flags, not the namer); a buffer scope swaps in a fresh line buffer for a cut or a split.
use super::buffer::Buffer;
use crate::body::{ty_of, Class, Frame, Node};
use crate::ir::door;

/// What an open loop scope must undo when it closes.
#[must_use = "a loop scope must be closed"]
pub struct LoopScope {
    /// The indent the scope opened at, put back on close.
    saved_indent: String,
    /// The first slot the scope appended.
    pub base: usize,
    /// The handles of the nodes the scope appended, removed on close.
    added: Vec<u32>,
}

/// The line buffer a buffer scope put aside.
#[must_use = "a buffer scope must be closed"]
pub struct BufferScope {
    /// The buffer that was current when the scope opened.
    outer: Buffer,
}

impl Frame<'_> {
    /// The slot of a node of an open loop scope (the node must be one).
    pub fn slot_of(&self, h: u32) -> usize {
        self.slot[&h]
    }

    /// Whether the node is a slot of an open loop scope (it has no canonical position).
    pub fn is_slot(&self, h: u32) -> bool {
        self.slot.contains_key(&h)
    }

    /// Appends the slot of one node of a loop scope.
    fn add_slot(&mut self, h: u32) -> Result<usize, String> {
        let s = self.count();
        let node = Node {
            h,
            class: Class::of(self.ctx(), h)?,
            ty: ty_of(self.ctx(), h)?,
            ops: door::operands(self.ctx(), h)?,
        };
        self.push_slot(node);
        self.slot.insert(h, s);
        Ok(s)
    }

    /// Opens a loop's scope: one level deeper, with the index, the carries and the body's nodes
    /// appended to the tables.
    ///
    /// The rule a caller keeps: every open has its `close_loop` on the success path. On an error
    /// path the scope stays open, and
    /// that is safe only because an error leaves the render: it reaches `render_frame_over`, which
    /// returns the error and drops
    /// the frame (no site in the writers catches an error of a render and goes on with the same
    /// frame; checked by grep). A caller
    /// that would recover from an error must close the scope first. The explicit open and close
    /// are kept (not closure forms):
    /// the scope's body spans many locals and early `?` returns, and `in_case` already restores
    /// structurally.
    pub(crate) fn open_loop(
        &mut self,
        index_h: u32,
        carries: &[u32],
        body: &[u32],
    ) -> Result<LoopScope, String> {
        let (saved_indent, base) = (self.buf.indent.clone(), self.count());
        self.buf.indent.push_str("    ");
        let mut added = vec![index_h];
        added.extend(carries.iter().copied());
        added.extend(body.iter().copied());
        for &h in &added {
            self.add_slot(h)?;
        }
        for s in base..self.count() {
            let ps = self
                .node(s)
                .ops
                .clone()
                .into_iter()
                .map(|o| self.pos_of(o))
                .collect::<Result<Vec<_>, _>>()?;
            self.set_opp(s, ps);
        }
        self.settle_disp(base, self.count(), &[]);
        Ok(LoopScope {
            saved_indent,
            base,
            added,
        })
    }

    /// Closes a loop's scope: what it named does not outlive it.
    pub(crate) fn close_loop(&mut self, scope: LoopScope) {
        for h in &scope.added {
            self.slot.remove(h);
        }
        self.truncate(scope.base);
        self.buf.indent = scope.saved_indent;
    }

    /// Runs `run` in a case scope: the texts, the scoped flags and the indent come back as they
    /// were (the lazy and forced flags and the namer do not).
    pub(crate) fn in_case<R>(
        &mut self,
        run: impl FnOnce(&mut Frame) -> Result<R, String>,
    ) -> Result<R, String> {
        let saved = (self.buf.indent.clone(), self.save_scope());
        let out = run(self);
        self.buf.indent = saved.0;
        self.restore_scope(saved.1);
        out
    }

    /// Puts the line buffer aside for a fresh one (a cut or a split renders one trip into it).
    ///
    /// The rule a caller keeps: every open has its `close_buffer` on the success path; on an error
    /// path the fresh buffer stays in
    /// place, which is safe only because an error leaves the render and the frame is dropped (see
    /// `open_loop`).
    pub(crate) fn open_buffer(&mut self) -> BufferScope {
        let fresh = Buffer::new(vec![], self.buf.indent.clone());
        BufferScope {
            outer: std::mem::replace(&mut self.buf, fresh),
        }
    }

    /// Puts the outer line buffer back and returns the buffer written in the scope.
    pub(crate) fn close_buffer(&mut self, scope: BufferScope) -> Buffer {
        let indent = self.buf.indent.clone();
        let inner = std::mem::replace(&mut self.buf, scope.outer);
        self.buf.indent = indent;
        inner
    }
}
