//! The per-position storage of a frame. `Rows` are the walk's rows (nodes, operand positions,
//! fan-out, readers, spans); they are built once and never written after. The rows of an open loop
//! scope are kept apart, appended after the table's, and dropped with the scope. `Pos` is what
//! rendering records for one position. The fields are private: everything goes through methods.
use crate::body::{Class, Node};
use crate::ir::door::RxCtx;
use std::collections::BTreeSet;

/// What a position is, decided once before it renders; the first rule that matches wins. (A
/// position inside a switch case is told by `outside`, which stays a flag of its own: the case
/// renderer still asks whether such a position is absorbed or swallowed.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Disp {
    /// A statement class the sink pass writes.
    Sink,
    /// Written element by element by a placement, and not a literal.
    Absorbed,
    /// A store run writes it.
    Swallowed,
    /// Stored per component, never whole.
    PerPart,
    /// Rendered, never named.
    Inline,
    /// A loop reads it from outside: named.
    Forced,
    /// Named by its reader count.
    #[default]
    Plain,
}

impl Disp {
    /// The disposition for the facts about a position, in the order the renderer tests them.
    pub fn of(
        sink: bool,
        absorbed: bool,
        swallowed: bool,
        per_part: bool,
        inline: bool,
        forced: bool,
    ) -> Disp {
        if sink {
            Disp::Sink
        } else if absorbed {
            Disp::Absorbed
        } else if swallowed {
            Disp::Swallowed
        } else if per_part {
            Disp::PerPart
        } else if inline {
            Disp::Inline
        } else if forced {
            Disp::Forced
        } else {
            Disp::Plain
        }
    }
}

/// How far a position got.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum State {
    /// Not rendered yet.
    #[default]
    Pending,
    /// Rendered, or settled as never rendered whole.
    Done,
}

/// What rendering records for one position.
#[derive(Clone, Default)]
pub struct Pos {
    /// What the position is (see `Disp`).
    disp: Disp,
    /// How far it got.
    state: State,
    /// The text the node was rendered to (empty before it renders).
    text: String,
    /// Bound as the `[i]` proxy and read only element by element.
    lazy: bool,
    /// A loop reads it from outside, so it is named (an input to `Disp`).
    forced: bool,
    /// Some path from a sink reaches it without crossing the branches of a switch dispatch.
    outside: bool,
    /// Bound inside the scope being rendered (a case of a switch).
    scoped: bool,
    /// The needs of the position's own text (the helpers its spelling calls); they count once
    /// something reads the text.
    needs: super::needs::Needs,
    /// Whether anything has read the text. A cell for the reason `Positions::pending` is one.
    read: std::cell::Cell<bool>,
    /// The declared names the position's text reads: its own name when it is named, else the
    /// names its operands' texts and the names it took while rendering read.
    reads: BTreeSet<String>,
}

/// The clamped index span of a node: the position of the base value and the lowest and highest
/// offsets added to it, so that one range test of the base covers every read of the group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// The position of the base value the reads add their offsets to.
    pub base: usize,
    /// The lowest offset added to the base.
    pub lo: i64,
    /// The highest offset added to the base.
    pub hi: i64,
}

/// The rows of a walk, or of the open loop scopes.
#[derive(Default)]
pub struct Rows {
    /// The nodes, by position.
    nodes: Vec<Node>,
    /// Operand positions of every node (a duplicate answers its representative's position).
    opp: Vec<Vec<usize>>,
    /// How many operand reads each position has.
    fanout: Vec<usize>,
    /// (user position, operand index) of every reader of a node.
    users: Vec<Vec<(usize, usize)>>,
    /// The clamped index span of each position, where it has one.
    spans: Vec<Option<Span>>,
}

/// The tables of a frame: the immutable rows, the rows of open loop scopes and one `Pos` per
/// position of both.
pub struct Positions {
    /// The rows of the walk, fixed once built.
    table: Rows,
    /// The rows the open loop scopes appended after the walk's.
    scope: Rows,
    /// What rendering recorded for each position of both.
    pos: Vec<Pos>,
    /// The needs of read texts whose slots a closed loop scope dropped.
    retired: super::needs::Needs,
    /// The names read since the last line was emitted or text was set: what the next one reads.
    /// A cell because readers (`text`, `peek`, `add_read`) take `&self`: a line is built from several operand texts held at
    /// once (`format!("{} {}", f.text(a), f.text(b))`), which `&mut self` readers returning `&String` would not allow.
    /// The cells of the frame are these two (`pending` and `Pos::read`) and `Frame::needs`; nothing else is interior-mutable.
    pending: std::cell::RefCell<BTreeSet<String>>,
    /// Every name a declaration of the render introduced, in whichever buffer it sits.
    declared: std::collections::HashSet<String>,
}

/// The names the loops of a walk carry: the carried locals of each loop, the iteration counters
/// and the tape arrays. The
/// counters and tapes are fixed by the walk and written once, by `index_loops`; the carried locals
/// of a loop are recorded when
/// its statement is emitted (a value read of the loop looks them up after).
#[derive(Default)]
pub struct LoopNames {
    /// The C++ names of each loop's carried locals, by loop position.
    carries: Vec<(usize, Vec<String>)>,
    /// The per-sample iteration counter name of a loop, by loop position.
    counters: Vec<(usize, String)>,
    /// The tape array of a forward loop's carried slot, by (loop position, slot).
    tapes: Vec<((usize, usize), String)>,
}

/// The table of a walk, built once by `Table::build`: the context and walk it reads, the rows with
/// what rendering recorded for
/// each position, the sinks the walk commits, the suffix each loop's identifiers carry and the
/// names the loops carry. The
/// context, walk, rows and sinks are not written after the build; loop suffixes and counter and
/// tape names are written once, by
/// `index_loops`, and the carried names of a loop when it is emitted. A `Table` reads as its
/// `Positions`.
pub struct Table {
    /// The IR context the walk lives in.
    ctx: *mut RxCtx,
    /// The walk's handle.
    walk: u32,
    /// The rows and the per-position state.
    rows: Positions,
    /// The sinks of the walk, in commit order.
    sinks: Vec<u32>,
    /// The suffix each loop's identifiers carry, by loop handle.
    loop_pos: Vec<(u32, usize)>,
    /// The names the loops carry.
    names: LoopNames,
}

impl Table {
    /// The table of a walk from its context, handle, rows and sinks; no loop is indexed yet.
    pub fn build(ctx: *mut RxCtx, walk: u32, rows: Positions, sinks: Vec<u32>) -> Self {
        Table {
            ctx,
            walk,
            rows,
            sinks,
            loop_pos: vec![],
            names: LoopNames::default(),
        }
    }
    /// The IR context.
    pub fn ctx(&self) -> *mut RxCtx {
        self.ctx
    }
    /// The walk's handle.
    pub fn walk(&self) -> u32 {
        self.walk
    }
    /// The sinks of the walk.
    pub fn sinks(&self) -> &[u32] {
        &self.sinks
    }
    /// The suffix of each loop, by handle; empty until the loops are indexed.
    pub fn loop_pos(&self) -> &[(u32, usize)] {
        &self.loop_pos
    }
    /// Records that loop `h` carries the suffix `pos`.
    pub fn index_loop(&mut self, h: u32, pos: usize) {
        self.loop_pos.push((h, pos))
    }
    /// The carried locals' names of each emitted loop, by loop position.
    pub fn carry_names(&self) -> &[(usize, Vec<String>)] {
        &self.names.carries
    }
    /// Records the carried locals' names of the loop at `pos`.
    pub fn record_carries(&mut self, pos: usize, names: Vec<String>) {
        self.names.carries.push((pos, names))
    }
    /// The iteration counter name of each loop that has one, by loop position.
    pub fn count_names(&self) -> &[(usize, String)] {
        &self.names.counters
    }
    /// Records the iteration counter name of the loop at `pos`.
    pub fn add_counter(&mut self, pos: usize, name: String) {
        self.names.counters.push((pos, name))
    }
    /// The tape array name of each forward loop slot that has one, by (loop position, slot).
    pub fn tape_names(&self) -> &[((usize, usize), String)] {
        &self.names.tapes
    }
    /// Records the tape array name of slot `slot` of the loop at `pos`.
    pub fn add_tape(&mut self, pos: usize, slot: usize, name: String) {
        self.names.tapes.push(((pos, slot), name))
    }
}

impl std::ops::Deref for Table {
    type Target = Positions;
    fn deref(&self) -> &Positions {
        &self.rows
    }
}

impl std::ops::DerefMut for Table {
    fn deref_mut(&mut self) -> &mut Positions {
        &mut self.rows
    }
}

impl Positions {
    /// The tables of a walk: nothing rendered, nothing scoped, nothing lazy.
    pub fn new(
        nodes: Vec<Node>,
        opp: Vec<Vec<usize>>,
        fanout: Vec<usize>,
        users: Vec<Vec<(usize, usize)>>,
        spans: Vec<Option<Span>>,
        outside: Vec<bool>,
        forced: Vec<bool>,
    ) -> Self {
        let pos = outside
            .iter()
            .zip(&forced)
            .map(|(&outside, &forced)| Pos {
                outside,
                forced,
                ..Pos::default()
            })
            .collect();
        Positions {
            retired: Default::default(),
            pending: Default::default(),
            declared: Default::default(),
            table: Rows {
                nodes,
                opp,
                fanout,
                users,
                spans,
            },
            scope: Rows::default(),
            pos,
        }
    }

    /// The number of positions (the table's and the open scopes').
    pub fn count(&self) -> usize {
        self.pos.len()
    }

    /// The number of positions of the walk itself (a position below it is outside every open scope).
    pub fn walk_count(&self) -> usize {
        self.table.nodes.len()
    }

    /// The node at a position.
    pub fn node(&self, p: usize) -> &Node {
        let table = self.table.nodes.len();
        if p < table {
            &self.table.nodes[p]
        } else {
            &self.scope.nodes[p - table]
        }
    }

    /// Every node of the table, in position order.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.table.nodes.iter().chain(&self.scope.nodes)
    }

    /// The operand positions of the node at `p`.
    pub fn opp(&self, p: usize) -> &Vec<usize> {
        let table = self.table.nodes.len();
        if p < table {
            &self.table.opp[p]
        } else {
            &self.scope.opp[p - table]
        }
    }

    /// Sets the operand positions of the node at `p`, a slot of an open loop scope.
    pub fn set_opp(&mut self, p: usize, positions: Vec<usize>) {
        let table = self.table.nodes.len();
        if self.scope.nodes[p - table].class != Class::Primitive {
            for (k, &q) in positions.iter().enumerate() {
                if q >= table {
                    self.scope.users[q - table].push((p, k))
                }
            }
        }
        self.scope.opp[p - table] = positions
    }

    /// How many operand reads the node at `p` has.
    pub fn fanout(&self, p: usize) -> usize {
        let table = self.table.nodes.len();
        if p < table {
            self.table.fanout[p]
        } else {
            self.scope.fanout[p - table]
        }
    }

    /// The (user position, operand index) pairs that read the node at `p`.
    pub fn users(&self, p: usize) -> &Vec<(usize, usize)> {
        let table = self.table.nodes.len();
        if p < table {
            &self.table.users[p]
        } else {
            &self.scope.users[p - table]
        }
    }

    /// The span a static component read covers, when the node has one.
    pub fn span(&self, p: usize) -> Option<Span> {
        let table = self.table.nodes.len();
        if p < table {
            self.table.spans[p]
        } else {
            self.scope.spans[p - table]
        }
    }

    /// The text the node at `p` was rendered to (empty before it renders).
    pub fn text(&self, p: usize) -> &String {
        self.pos[p].read.set(true);
        self.pending
            .borrow_mut()
            .extend(self.pos[p].reads.iter().cloned());
        &self.pos[p].text
    }

    /// The text of the node at `p` without recording that the text being written reads it: for a
    /// writer that inspects an operand's text (to compare it or to choose a spelling) and does not
    /// put it into its own text. The text is still marked as read, so the helpers it calls are
    /// kept.
    pub fn peek(&self, p: usize) -> &String {
        self.pos[p].read.set(true);
        &self.pos[p].text
    }

    /// The declared names the text of the node at `p` reads.
    pub fn reads(&self, p: usize) -> &BTreeSet<String> {
        &self.pos[p].reads
    }

    /// Notes that a declaration introduced `name`.
    pub fn note_declared(&mut self, name: &str) {
        self.declared.insert(name.to_string());
    }

    /// Whether a declaration introduced `name`.
    pub fn is_declared(&self, name: &str) -> bool {
        self.declared.contains(name)
    }

    /// Notes that the text being written reads `name`.
    pub fn add_read(&self, name: &str) {
        self.pending.borrow_mut().insert(name.to_string());
    }

    /// Takes the names read since the last line or text was recorded.
    pub fn take_reads(&self) -> BTreeSet<String> {
        std::mem::take(&mut *self.pending.borrow_mut())
    }

    /// Records the helpers the spelled text of the node at `p` calls.
    pub fn hold_needs(&mut self, p: usize, used: &[crate::spell::Helper]) {
        self.pos[p].needs.record_spelled(used)
    }

    /// The needs of the texts something has read, in position order.
    pub fn read_needs(&self) -> super::needs::Needs {
        let mut all = self.retired.clone();
        for pos in self.pos.iter().filter(|pos| pos.read.get()) {
            all.merge(&pos.needs)
        }
        all
    }

    /// Records the text of the node at `p`.
    /// The names read since the last record are the position's reads, or its own name when it
    /// is `named`.
    pub fn set_text(&mut self, p: usize, text: String, named: bool) {
        let taken = self.take_reads();
        self.pos[p].reads = if named {
            BTreeSet::from([text.clone()])
        } else {
            taken
        };
        self.pos[p].text = text
    }

    /// Whether `p` is reached without crossing a switch's branches.
    pub fn is_outside(&self, p: usize) -> bool {
        self.pos[p].outside
    }

    /// Marks whether `p` is reached without crossing a switch's branches.
    pub fn set_outside(&mut self, p: usize, on: bool) {
        self.pos[p].outside = on
    }

    /// Whether `p` is bound inside the scope being rendered.
    pub fn is_scoped(&self, p: usize) -> bool {
        self.pos[p].scoped
    }

    /// Marks `p` as bound inside the scope being rendered (or not).
    pub fn set_scoped(&mut self, p: usize, on: bool) {
        self.pos[p].scoped = on
    }

    /// Whether `p` is bound as the `[i]` proxy.
    pub fn is_lazy(&self, p: usize) -> bool {
        self.pos[p].lazy
    }

    /// Marks `p` as bound as the `[i]` proxy (or not).
    pub fn set_lazy(&mut self, p: usize, on: bool) {
        self.pos[p].lazy = on
    }

    /// Whether a loop reads `p` from outside (an input to its disposition).
    pub fn is_forced(&self, p: usize) -> bool {
        self.pos[p].forced
    }

    /// The disposition of `p`.
    pub fn disp(&self, p: usize) -> Disp {
        self.pos[p].disp
    }

    /// Records the disposition of `p`.
    pub fn set_disp(&mut self, p: usize, disp: Disp) {
        self.pos[p].disp = disp
    }

    /// Whether `p` has been rendered (or settled as never rendered whole).
    pub fn is_done(&self, p: usize) -> bool {
        self.pos[p].state == State::Done
    }

    /// Marks `p` as done.
    pub fn set_done(&mut self, p: usize) {
        self.pos[p].state = State::Done
    }

    /// Appends one slot for a loop scope's node: not yet rendered, outside, no operands.
    pub fn push_slot(&mut self, node: Node) {
        self.scope.nodes.push(node);
        self.scope.opp.push(vec![]);
        self.scope.fanout.push(0);
        self.scope.users.push(vec![]);
        self.scope.spans.push(None);
        self.pos.push(Pos {
            outside: true,
            ..Pos::default()
        });
    }

    /// Drops every slot from `base` on (a loop scope closing).
    pub fn truncate(&mut self, base: usize) {
        let keep = base.saturating_sub(self.table.nodes.len());
        self.scope.nodes.truncate(keep);
        self.scope.opp.truncate(keep);
        self.scope.fanout.truncate(keep);
        self.scope.users.truncate(keep);
        self.scope.spans.truncate(keep);
        for dropped in self.pos.iter().skip(base).filter(|p| p.read.get()) {
            self.retired.merge(&dropped.needs)
        }
        self.pos.truncate(base);
    }

    /// The texts and the scoped flags, for a scope that restores them when it closes.
    pub fn save_scope(&self) -> Vec<(String, bool)> {
        self.pos
            .iter()
            .map(|p| (p.text.clone(), p.scoped))
            .collect()
    }

    /// Puts back the texts and the scoped flags `save_scope` took (not the lazy or forced flags).
    pub fn restore_scope(&mut self, saved: Vec<(String, bool)>) {
        for (pos, (text, scoped)) in self.pos.iter_mut().zip(saved) {
            pos.text = text;
            pos.scoped = scoped;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{Class, Ty};

    fn node(h: u32) -> Node {
        Node {
            h,
            class: Class::Op,
            kind: None,
            ty: Ty {
                dtype: "f64".into(),
                shape: vec![],
            },
            ops: vec![],
        }
    }

    fn two_rows() -> Positions {
        Positions::new(
            vec![node(1), node(2)],
            vec![vec![], vec![0]],
            vec![1, 0],
            vec![vec![(1, 0)], vec![]],
            vec![None, None],
            vec![true, false],
            vec![false, true],
        )
    }

    #[test]
    fn a_loop_scope_drops_the_slots_it_pushed() {
        let mut t = two_rows();
        t.push_slot(node(7));
        t.push_slot(node(8));
        t.set_opp(3, vec![2]);
        assert_eq!((t.count(), t.node(3).h, t.opp(3).clone()), (4, 8, vec![2]));
        t.truncate(2);
        assert_eq!((t.count(), t.nodes().count(), t.node(1).h), (2, 2, 2));
    }

    #[test]
    fn a_case_scope_restores_text_and_scoped_but_not_lazy_or_forced() {
        let mut t = two_rows();
        t.set_text(0, "a".into(), false);
        let saved = t.save_scope();
        t.set_text(0, "b".into(), false);
        t.set_scoped(1, true);
        t.set_lazy(0, true);
        t.restore_scope(saved);
        assert_eq!(t.text(0), "a");
        assert!(!t.is_scoped(1));
        assert!(t.is_lazy(0), "lazy stays set after a case scope");
        assert!(t.is_forced(1), "forced is never touched by a case scope");
    }

    #[test]
    fn disposition_follows_the_order_of_the_renderer_tests() {
        // (sink, absorbed, swallowed, per_part, inline, forced)
        let of = Disp::of;
        assert_eq!(of(true, true, true, true, true, true), Disp::Sink);
        assert_eq!(of(false, true, true, true, true, true), Disp::Absorbed);
        assert_eq!(of(false, false, true, true, true, true), Disp::Swallowed);
        assert_eq!(of(false, false, false, true, true, true), Disp::PerPart);
        assert_eq!(of(false, false, false, false, true, true), Disp::Inline);
        assert_eq!(of(false, false, false, false, false, true), Disp::Forced);
        assert_eq!(of(false, false, false, false, false, false), Disp::Plain);
    }

    #[test]
    fn done_is_set_once_and_independent_of_the_case_scope() {
        let mut t = two_rows();
        assert!(!t.is_done(0));
        t.set_done(0);
        let saved = t.save_scope();
        t.restore_scope(saved);
        assert!(t.is_done(0));
    }
}
