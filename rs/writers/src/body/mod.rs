//! The renderer core: the body of a unit from its walk. Values are numbered by their canonical
//! position in the walk (a duplicate
//! answers its representative's position), and temporaries are named by that position, as the
//! numbering the text carries.
//! This file is the frame (the contract, the node table, fan-out, naming, the dispatch over node
//! classes); each class of node has its own file.
pub(crate) mod dispatch;
pub(crate) mod frame;
mod linalg;
pub(crate) mod loops;
mod ops;
mod placed;
mod render;
pub mod stores;
mod team;
mod values;

use crate::contract::types::{Form, Guard, SinkPolicy, Unit};
use crate::ir::door::{self, RxCtx};
pub(crate) use crate::kind::OpKind;
use crate::refuse::{fault, nib, Refusal};
use crate::spell::binding_name;
pub(crate) use frame::buffer::{Buffer, Mark};
pub(crate) use frame::class::Class;
use frame::disp::ScopeKind;
use frame::namer::{Name, NameKind, Namer};
use frame::table::Disp;
pub(crate) use frame::table::Span;
use frame::table::{Positions, Table};

/// The type of a node: its element type name and its shape.
#[derive(Clone)]
pub struct Ty {
    /// The element type name (`f64`, `i32`, ...).
    pub dtype: String,
    /// The extent of each dimension; empty for a scalar.
    pub shape: Vec<usize>,
}

/// The canonical position of a node in the walk; a node inside a loop body has none, which is
/// refused by name.
fn walk_position(ctx: *mut RxCtx, walk: u32, h: u32) -> Result<usize, Refusal> {
    door::position(ctx, walk, h).map_err(|e| {
        if door::in_loop_body(ctx, walk, h).unwrap_or(false) {
            nib("an operand inside a loop body")
        } else {
            e
        }
    })
}

/// The type of a node, as the door's typed reader answers it.
pub(crate) fn ty_of(ctx: *mut RxCtx, h: u32) -> Result<Ty, Refusal> {
    let t = door::ttype_of(ctx, h)?;
    Ok(Ty {
        dtype: t.dtype,
        shape: t.shape.iter().map(|&d| d as usize).collect(),
    })
}

/// The text a node's field holds, without quotes; empty when the field holds nothing.
pub(crate) fn field_text(ctx: *mut RxCtx, h: u32, f: &str) -> Result<String, Refusal> {
    Ok(door::field_text(ctx, h, f)?.unwrap_or_default())
}

/// The kind of a node of class `class`: read for an operation, none for any other class.
pub(crate) fn kind_of(ctx: *mut RxCtx, h: u32, class: &Class) -> Result<Option<OpKind>, Refusal> {
    (*class == Class::Op)
        .then(|| OpKind::of(ctx, h))
        .transpose()
}

/// A node of the walk: its handle, class, type and operand handles.
pub struct Node {
    /// The node's handle in the IR.
    pub h: u32,
    /// The node's class, as the writers tell classes apart.
    pub class: Class,
    /// The kind of an operation node, read once; none for any other class.
    pub kind: Option<OpKind>,
    /// The node's type.
    pub ty: Ty,
    /// The handles of the node's operands, in order.
    pub ops: Vec<u32>,
}
impl Node {
    /// The number of dimensions of the node's type.
    pub fn rank(&self) -> usize {
        self.ty.shape.len()
    }
    /// The number of elements of the node's type.
    pub fn numel(&self) -> usize {
        self.ty.shape.iter().product()
    }
}

/// What a class writer gives back for a node: its text and the slot role of a leaf, or the news
/// that it bound the rendering itself.
pub enum Out {
    /// The node's text, and the slot role when the node is a leaf.
    Value(String, Option<String>),
    /// The class writer bound the rendering itself: nothing is left to name.
    Bound,
}

/// The correction plane of the compensated sink policy and the sink its commit is spelled for.
#[derive(Clone)]
pub struct CompensatedPlan {
    /// The correction plane's name.
    pub plane: String,
    /// The target the commit is spelled for, when the sinks name one.
    pub target: Option<CompensatedTarget>,
}

/// The target of a compensated commit.
#[derive(Clone)]
pub struct CompensatedTarget {
    /// The target's name.
    pub name: String,
    /// Whether the target is the own column of a struct.
    pub own_column: bool,
}

/// A lane proxy the text reads in place of a placed plane.
pub struct Proxy {
    /// The text the body reads (`plane[i]` or `plane_i`).
    pub name: String,
    /// The plane the proxy stands for.
    pub plane: String,
    /// The width of the plane's row; 0 for a mutable's own plane, which has no row to declare.
    pub width: usize,
}

/// What a caller that asks for a cut of the body supplies: the writer of each cut loop and the step after the walk.
/// The body knows only this trait; what the writer records is the caller's own state.
pub trait LoopEmitter {
    /// Writes the loop `lp` when the walk meets it (one of the loops the caller asked to cut): the loop's own text is not
    /// emitted unless this does it.
    fn emit(&mut self, f: &mut Frame, lp: u32) -> Result<(), Refusal>;
    /// Called once with the finished frame and the guard's condition (when the kernel has one), after the last position.
    fn finish(&mut self, f: &Frame, guard: Option<String>) -> Result<(), Refusal>;
    /// Whether the cut renders one trip of a loop into a text of its own and leaves the walk's placements unwritten as
    /// planes; false for a cut whose placements are written as their planes. A typed flag, not a lane fact.
    fn one_trip(&self) -> bool {
        false
    }
}

/// The state of one render: the node table in walk order, operand positions, readers, renderings
/// so far and the lines emitted.
pub struct Frame<'e> {
    /// The walk's table: its rows and what rendering recorded for each position, its sinks and its
    /// loop suffixes.
    table: Table,
    /// The lines emitted, the indent of the next and the declarations written.
    pub buf: Buffer,
    /// What the text emitted so far needs from the unit around it, recorded where each helper call
    /// is written.
    /// A cell for the same reason as the table's read marks: `need` is called from `&self` readers while operand texts are
    /// held.
    pub needs: std::cell::RefCell<frame::needs::Needs>,
    /// Whether the sinks are written as one block fold.
    pub block_fold: bool,
    /// The typed unit the frame writes: its kind, mode, build settings and machine facts.
    pub unit: Unit,
    /// Counter of the temporaries a loop body names (`tb<n>`); never reset.
    pub names: Namer,
    /// Slot of a loop-body node (or a loop's index or carry) by handle: body nodes are appended to
    /// the tables while their scope is open.
    slot: std::collections::HashMap<u32, usize>,
    /// The top-level loops the caller asked to cut out, handed to its emitter when the walk meets them.
    pub cut: Vec<u32>,
    /// The caller's loop emitter; `None` without a cut. It is taken out while it runs.
    loop_emitter: Option<&'e mut dyn LoopEmitter>,
    /// Under the compensated sink policy: the correction plane, and the sink its commit is spelled
    /// for with whether it is the own column.
    pub compensated: Option<CompensatedPlan>,
    /// The walk's placements are written as their planes (the interchanged host text): values the
    /// placements write element by
    /// element, by handle, and the lane proxies the text reads, as (proxy, plane, width).
    pub placed: bool,
    /// The handles of the nodes a placed write absorbed: their text is written by that write.
    pub absorbed: std::collections::HashSet<u32>,
    /// The lane proxies the text reads.
    pub proxies: Vec<Proxy>,
    /// The runtime-index stores of the body: their runs and the nodes the runs swallow.
    pub(crate) stores: stores::storeplan::Plan,
    /// The carried slots a loop without a break lets a store run write in place, by handle.
    pub mutable_carries: std::collections::HashSet<u32>,
    /// The banded `sincos` pairs bound so far, by the position of the shared operand: the names of the sine and the cosine.
    sincos: Vec<(usize, String, String)>,
}

/// The frame reads and writes its table directly: the table's accessors (`f.node(p)`, `f.text(p)`, `f.set_done(p)`, ...)
/// are the frame's. The frame spells out only what adds to a table call: a read recorded for the line, a scope rule, the IR
/// context.
impl std::ops::Deref for Frame<'_> {
    type Target = Table;
    fn deref(&self) -> &Table {
        &self.table
    }
}

impl std::ops::DerefMut for Frame<'_> {
    fn deref_mut(&mut self) -> &mut Table {
        &mut self.table
    }
}

impl<'e> Frame<'e> {
    /// Takes the caller's loop emitter out of the frame (while it writes a loop, which needs the frame).
    pub(crate) fn take_loop_emitter(&mut self) -> Option<&'e mut dyn LoopEmitter> {
        self.loop_emitter.take()
    }
    /// Puts the loop emitter back after it wrote a loop.
    pub(crate) fn put_loop_emitter(&mut self, emit: &'e mut dyn LoopEmitter) {
        self.loop_emitter = Some(emit);
    }
}

impl Frame<'_> {
    /// The text a node's field holds, without quotes; empty when the field holds nothing.
    pub fn field_text(&self, h: u32, f: &str) -> Result<String, Refusal> {
        field_text(self.ctx(), h, f)
    }
    /// The kind of the operation at position `p`; a node of another class is a fault.
    pub fn op_kind(&self, p: usize) -> Result<OpKind, Refusal> {
        self.node(p)
            .kind
            .ok_or_else(|| fault(format!("node {} is not an operation", self.node(p).h)))
    }
    /// The canonical position of a node in the walk (a node inside a loop body is refused by name).
    pub fn at(&self, h: u32) -> Result<usize, Refusal> {
        walk_position(self.ctx(), self.walk(), h)
    }
    /// The rendering of the node with handle `h`, read like `peek`: no read is recorded.
    pub fn peek_arg(&self, h: u32) -> Result<String, Refusal> {
        Ok(self.peek(self.pos_of(h)?).clone())
    }
    /// Notes that the text being written holds the text of the node at `p`: for a writer that
    /// took the text with `peek` and then put it into its own text.
    pub fn record_read(&self, p: usize) {
        self.table.text(p);
    }
    /// Notes the names the node at `p` reads that the text being written still holds: for a writer
    /// that took operand texts with `peek` and folded some of them away while building its text.
    pub fn record_names_in(&self, p: usize, text: &str) {
        for name in self.table.reads(p) {
            if crate::body::frame::buffer::tokens(text).any(|word| word == name) {
                self.table.add_read(name)
            }
        }
    }
    /// Appends a line that holds no name. The names the next line is built from stay pending, so
    /// they are recorded on that line.
    pub fn emit_bare(&mut self, line: String) {
        let held = self.table.take_reads();
        self.emit(line);
        for name in held {
            self.table.add_read(&name)
        }
    }
    /// Records the text of the node at `p`.
    pub fn set_text(&mut self, p: usize, text: String) {
        let named = self.table.is_declared(&text);
        self.table.set_text(p, text, named)
    }
    /// Records the name of a loop's carried slot as the text of the node at `p`: a name the
    /// slot's readers read, whether or not a declaration introduced it.
    pub fn set_name_text(&mut self, p: usize, name: String) {
        self.table.note_declared(&name);
        self.table.set_text(p, name, true)
    }
    /// The declared names the text of the node with handle `h` reads.
    pub fn reads_of(&self, h: u32) -> Result<std::collections::BTreeSet<String>, Refusal> {
        Ok(self.reads(self.pos_of(h)?).clone())
    }
    /// The position of a node: its slot while a loop scope holds it, else its canonical position
    /// in the walk.
    pub fn pos_of(&self, h: u32) -> Result<usize, Refusal> {
        match self.slot.get(&h) {
            Some(&s) => Ok(s),
            None => self.at(h),
        }
    }
    /// The rendering of a node as an operand.
    pub fn arg(&self, h: u32) -> Result<String, Refusal> {
        Ok(self.text(self.pos_of(h)?).clone())
    }
    /// A value that is itself an aether leaf (a row of a plane, an assembled vector): always named.
    pub fn leaf_valued(&self, s: usize) -> Result<bool, Refusal> {
        let n = &self.node(s);
        Ok(
            ((n.class == Class::Leaf || n.class == Class::At) && n.rank() > 0)
                || matches!(n.kind, Some(OpKind::Splat | OpKind::Vec)),
        )
    }
    /// Records a need of the text being written; the helper call it names is written next to this
    /// call.
    pub fn need(&self, need: frame::needs::Need) {
        self.needs.borrow_mut().record(need)
    }
    /// Records the helpers the spelled text of an operation calls.
    pub fn need_spelled(&self, used: &[crate::spell::Helper]) {
        self.needs.borrow_mut().record_spelled(used)
    }
    /// What the text written so far needs: the recorded needs and those of every value text that
    /// was read.
    pub fn recorded_needs(&self) -> frame::needs::Needs {
        let mut all = self.needs.borrow().clone();
        all.merge(&self.table.read_needs());
        all
    }
    /// Records a run state of the text being written.
    pub fn need_run(&self, target: &str) {
        self.needs.borrow_mut().record_run(target)
    }
    /// Appends a line at the current indent.
    pub fn emit(&mut self, line: String) {
        self.buf.lines.push(format!("{}{line}", self.buf.indent));
        let reads = self.table.take_reads();
        self.buf.reads.push(reads)
    }
    /// Binds a text to a fresh named local ahead of the current line.
    pub fn hoist(&mut self, text: String) -> String {
        let name = self.names.hoist();
        self.declare_auto(name.clone(), &text);
        self.add_read(&name.text);
        name.text
    }
}

/// One position of the flat walk: rendered once (a part may be rendered ahead of its turn), then
/// the stores due at it.
pub(crate) fn render_pos(
    f: &mut Frame,
    plan: &mut stores::commits::Plan,
    p: usize,
) -> Result<(), Refusal> {
    if !f.is_outside(p) || f.disp(p) == Disp::Sink {
        return Ok(());
    }
    if f.disp(p) == Disp::PerPart {
        f.set_done(p);
        return Ok(());
    } // committed per part, never whole
    if !f.is_done(p) {
        let group = dispatch::group_of(f, p)?;
        if !group.is_empty() {
            dispatch::emit_group(f, plan, &group)?;
            return flush_group(f, plan, &group);
        }
    }
    if !f.is_done(p) {
        let out = node_out(f, p)?;
        f.set_done(p);
        if let Out::Value(text, role) = out {
            name_value(f, p, text, role)?
        }
    }
    if !plan.due.is_empty() {
        stores::commits::flush(f, plan, p)?
    }
    Ok(())
}

/// The stores due at each member of a fused group, once the group is rendered.
fn flush_group(
    f: &mut Frame,
    plan: &mut stores::commits::Plan,
    group: &[usize],
) -> Result<(), Refusal> {
    if plan.due.is_empty() {
        return Ok(());
    }
    for &member in group {
        stores::commits::flush(f, plan, member)?
    }
    Ok(())
}

/// Binds a rendering to its node: named when two readers use it (or it is a leaf value, or a loop
/// reads it), else kept inline.
pub(crate) fn name_value(
    f: &mut Frame,
    p: usize,
    text: String,
    role: Option<String>,
) -> Result<(), Refusal> {
    let n_h = f.node(p).h;
    let leaf_valued = f.leaf_valued(p)?;
    // a uniform leaf and a literal are never named; a row leaf, a vec and a splat always are; any
    // other value once two readers use it
    let named = f.naming(p, ScopeKind::Top, role.as_deref(), f.fanout(p))?;
    // A loop value whose text is a plane proxy stays unnamed: a run-time index read recognises the
    // proxy by its text and reads the plane in place; a named copy would be a local item of the
    // plane's full width that the read then indexes.
    let named =
        named && !(f.node(p).class == Class::LoopValue && f.proxies.iter().any(|x| x.name == text));
    if named {
        let name = match &role {
            Some(r) => Name::new(
                format!("{}_i", binding_name(r, &f.field_text(n_h, "name")?)?),
                if text == format!("{}[i]", binding_name(r, &f.field_text(n_h, "name")?)?) {
                    NameKind::LeafView
                } else {
                    NameKind::RowLeaf
                },
            ),
            None => f.names.temp_at(p),
        };
        let ty = f.node(p).ty.clone();
        let held = f.fanout(p) >= 2 && !leaf_valued;
        let raw = f.flows_raw(p, &f.sink_values()?);
        f.declare(name.clone(), &ty.dtype, &ty.shape, &text, held, raw)?;
        f.set_text(p, name.text);
    } else {
        f.set_text(p, text)
    }
    Ok(())
}

/// The text of one node by its class (a statement class writes its lines and says it bound the
/// rendering itself).
pub(crate) fn node_out(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    // a literal a placement absorbs may be shared with another reader (equal literals are one
    // position): it is
    // never named (its disposition is not `Absorbed`), so its text is always there for whoever
    // reads it
    if matches!(f.disp(p), Disp::Absorbed | Disp::Swallowed) {
        return Ok(Out::Bound);
    }
    Ok(match f.node(p).class.clone() {
        Class::Leaf => values::leaf(f, p)?,
        Class::At => values::at(f, p)?,
        Class::PlaneExtent => values::plane_extent(f, p)?,
        Class::SampleIndex => values::sample_index()?,
        Class::SamplePosition => values::sample_position(f)?,
        Class::Select => values::select(f, p)?,
        Class::Const | Class::RoleConst => values::constant(f, p)?,
        Class::Op => ops::op(f, p)?,
        Class::Loop => loops::stmt(f, p)?,
        Class::LoopValue => loops::value(f, p)?,
        Class::LoopCount => loops::count(f, p)?,
        Class::TapeRead => loops::tape_read(f, p)?,
        Class::Dispatch | Class::PredicatedDispatch | Class::AutoDispatch => dispatch::stmt(f, p)?,
        Class::TeamRead
        | Class::TeamCopy
        | Class::TeamStore
        | Class::ScatterLog
        | Class::LogStore => team::stmt(f, p)?,
        Class::BoxStore | Class::WarpStore => Out::Bound,
        Class::WarpLane => Out::Value(crate::spell::WARP_LANE.to_string(), None),
        // an extension primitive stands for its forward expression
        Class::Primitive => {
            let forward = door::field_refs(f.ctx(), f.node(p).h, "forward")?.unwrap_or_default();
            let h = *forward
                .first()
                .ok_or_else(|| nib("a primitive with no forward expression"))?;
            f.set_text(p, f.arg(h)?);
            Out::Bound
        }
        other => return Err(nib(format!("node class {other}"))),
    })
}

/// The body of the unit: one statement per line, each led by `indent`. The walk handle `u32::MAX`
/// stands for a record with no walk.
pub fn render_body(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
) -> Result<String, Refusal> {
    Ok(render_body_with_needs(ctx, walk, unit, indent)?.text)
}

/// `render_body` with the needs recorded while its text was written.
pub fn render_body_with_needs(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
) -> Result<frame::needs::Rendered, Refusal> {
    let frame = render::render_frame_over(ctx, walk, unit, indent, &[], None, None)?.0;
    Ok(frame::needs::Rendered {
        text: frame.buf.lines.join("\n"),
        needs: frame.recorded_needs(),
    })
}

/// The body of a walk over the given sinks, as one statement per line led by `indent`. It is for a
/// unit's sub-renders (a boxed
/// body, a single step), whose sinks are not the contract's: a box or a single step in the
/// contract is not refused here.
pub fn render_body_over(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
    sinks: &[u32],
) -> Result<String, Refusal> {
    Ok(render_body_over_with_needs(ctx, walk, unit, indent, sinks)?.text)
}

/// `render_body_over` with the needs recorded while its text was written.
pub fn render_body_over_with_needs(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
    sinks: &[u32],
) -> Result<frame::needs::Rendered, Refusal> {
    let frame = render::render_frame_over(ctx, walk, unit, indent, &[], None, Some(sinks))?.0;
    Ok(frame::needs::Rendered {
        text: frame.buf.lines.join("\n"),
        needs: frame.recorded_needs(),
    })
}

/// One render of the body with a loop emitter: the walk hands each loop of `cut` to it, and at the end it gets the finished
/// frame (its lines are the body) and the guard's condition.
pub(crate) fn render_cut(
    ctx: *mut RxCtx,
    walk: u32,
    unit: &Unit,
    indent: &str,
    cut: &[u32],
    emit: &mut dyn LoopEmitter,
) -> Result<(), Refusal> {
    let (mut f, guard) = render::render_frame_over(ctx, walk, unit, indent, cut, Some(emit), None)?;
    if let Some(emit) = f.loop_emitter.take() {
        emit.finish(&f, guard)?;
    }
    Ok(())
}

/// The mask names that skip a lane's commit and the correction plane of the compensated sink
/// policy.
fn guard_and_policy(unit: &Unit) -> (Vec<String>, Option<String>) {
    let masks = match &unit.kind.guard {
        Guard::Default => vec!["terminated".to_string()],
        Guard::DataOnly => vec![],
        Guard::Masks(masks) => masks.clone(),
    };
    let plane = match &unit.kind.sink {
        SinkPolicy::Compensated { plane, .. } => Some(plane.clone()),
        SinkPolicy::Plain => None,
    };
    (masks, plane)
}

/// Whether the plain frame turns the block fold on: no segment, active set, box, team, two-path
/// entry, sample sum, staged rows or mask.
fn block_fold_of(ctx: *mut RxCtx, walk: u32, unit: &Unit, sums: bool) -> Result<bool, Refusal> {
    let keyed = unit.segment.is_some()
        || unit.active.is_some()
        || unit.thread_box.is_some()
        || unit.one_step.is_some()
        || unit.team.is_some();
    let masked = door::walk_args(ctx, walk)?
        .iter()
        .any(|(role, _)| role == "terminated");
    Ok(!(keyed || sums || unit.rows > 1 || masked))
}
