//! Stores at a runtime index into a vector (`set_component_at`, `add_at`). The node is functional:
//! the vector with one component
//! replaced. A run of them, each the only user of the one before, is one mutable local written in
//! place: the loop's own carried
//! local when nothing else reads it, a plane's lane proxy when the value lives in a plane, else one
//! copy made for the run. Each
//! store is written in place (`v(k) += e`, one indexed access) or as selects over the components
//! the index can reach (registers
//! only, work proportional to the reach); the machine's select weight and the key's warp uniformity
//! pick between them.
use crate::body::values::{component_index, literal_of, text_field};
use crate::body::{placed, Frame, Out};
use crate::body::{Class, Name, NameKind};
use crate::ir::door::Scalar;
use crate::ir::door::{self, same_index, through, RxCtx};
use crate::kind::OpKind;
use crate::refuse::{fault, nib, Refusal};
use crate::spell::carry_type;
use std::collections::{HashMap, HashSet};

/// A run is written as selects when (components its selects touch x the select weight) is at most
/// this many times its indexed
/// accesses (a read-modify-write counts two).
const SELECT_COST_RATIO: f64 = 32.0;
/// What an indexed access costs when its key is proven uniform over the warp.
const UNIFORM_ACCESS: f64 = 1.0 / 16.0;

/// One store: `node` writes `value` (added to the slot when `acc`) at `key`, which is `base +
/// offset` (`base` none for a static
/// key) and reaches the components `lo..=hi`.
#[derive(Clone)]
pub(super) struct Link {
    /// The store node.
    pub node: u32,
    /// The index the store writes at.
    pub key: u32,
    /// The runtime part of the index, none for a static index.
    pub base: Option<u32>,
    /// The constant part of the index.
    pub offset: i64,
    /// The lowest component the store can write.
    pub lo: i64,
    /// The highest component the store can write.
    pub hi: i64,
    /// Whether the store adds to the slot instead of replacing it.
    pub acc: bool,
    /// The value written, or the added value of an accumulate.
    pub value: u32,
    /// The read of the slot an accumulate pattern is built from, when the store is one.
    pub read: Option<u32>,
    /// The add of that accumulate pattern.
    pub add: Option<u32>,
}

/// What the stores of a body are, found once per render: the runs (by their top store), the nodes a
/// run swallows, and the nodes
/// that read each node.
#[derive(Default)]
pub(crate) struct Plan {
    /// The nodes that read each node.
    uses: HashMap<u32, Vec<u32>>,
    /// The loop each node lies in, none outside any loop.
    scope: HashMap<u32, Option<u32>>,
    /// The nodes a run swallows: they render no text of their own.
    pub skip: HashSet<u32>,
    /// The keys read only by their store, spelled where they are used.
    pub inline: HashSet<u32>,
    /// The runs of stores, by their last store.
    runs: HashMap<u32, Vec<Link>>,
    /// The planes a loop body accumulates into at runtime indices, as (binding, width), each once:
    /// zeroed whole before the walk.
    pub accumulated: Vec<(String, usize)>,
}

/// The value of a numeric constant node; refuses a literal that is not a number.
fn number(ctx: *mut RxCtx, h: u32) -> Result<f64, Refusal> {
    match literal_of(ctx, h)? {
        Scalar::Int(v) => Ok(v as f64),
        Scalar::Real(v) => Ok(v),
        other => Err(fault(format!(
            "a literal of {} that is not a number",
            other.kind_name()
        ))),
    }
}

/// The bounds of an index-valued node from its clamps and constant offsets alone; unknown is
/// unbounded.
fn interval(ctx: *mut RxCtx, h: u32) -> Result<(f64, f64), Refusal> {
    let h = through(ctx, h)?;
    let class = Class::of(ctx, h)?;
    if class == Class::Const {
        let v = number(ctx, h)?;
        return Ok((v, v));
    }
    let unknown = (f64::NEG_INFINITY, f64::INFINITY);
    if class != Class::Op {
        return Ok(unknown);
    }
    let kind = OpKind::of(ctx, h)?;
    if !matches!(kind, OpKind::Add | OpKind::Min | OpKind::Max | OpKind::Cast) {
        return Ok(unknown);
    }
    let parts = door::operands(ctx, h)?
        .into_iter()
        .map(|o| interval(ctx, o))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(match kind {
        OpKind::Add => (parts[0].0 + parts[1].0, parts[0].1 + parts[1].1),
        OpKind::Min => (parts[0].0.min(parts[1].0), parts[0].1.min(parts[1].1)),
        OpKind::Max => (parts[0].0.max(parts[1].0), parts[0].1.max(parts[1].1)),
        _ => (
            if parts[0].0.is_finite() {
                parts[0].0.trunc()
            } else {
                parts[0].0
            },
            if parts[0].1.is_finite() {
                parts[0].1.trunc()
            } else {
                parts[0].1
            },
        ),
    })
}

/// `(base, offset)` of `key = base + constants`; the base is none when the key is all constant.
fn split_key(ctx: *mut RxCtx, key: u32) -> Result<(Option<u32>, i64), Refusal> {
    let (mut key, mut offset) = (through(ctx, key)?, 0i64);
    while OpKind::of_op(ctx, key)? == Some(OpKind::Add) {
        let operands = door::operands(ctx, key)?;
        let (a, b) = (through(ctx, operands[0])?, through(ctx, operands[1])?);
        if door::class_is(ctx, b, "Const")? {
            key = a;
            offset += number(ctx, b)? as i64
        } else if door::class_is(ctx, a, "Const")? {
            key = b;
            offset += number(ctx, a)? as i64
        } else {
            break;
        }
    }
    if door::class_is(ctx, key, "Const")? {
        return Ok((None, offset + number(ctx, key)? as i64));
    }
    Ok((Some(key), offset))
}

/// Whether a node is a store at a runtime index.
fn is_store(f: &Frame, h: u32) -> Result<bool, Refusal> {
    Ok(matches!(
        OpKind::of_op(f.ctx(), h)?,
        Some(OpKind::SetComponentAt | OpKind::AddAt)
    ))
}

impl Plan {
    /// The plan of a frame: every store of the body and of the loop bodies, its run, and what the
    /// runs swallow.
    pub fn build(f: &Frame) -> Result<Plan, Refusal> {
        let mut plan = Plan::default();
        let nodes = plan.walk(f)?;
        let mut links: Vec<Link> = vec![];
        for &node in &nodes {
            // a chain over a vector part is committed link by link by its sink: no run takes it, a
            // whole read copies it
            if plan.is_committed(f, node)? {
                continue;
            }
            links.push(plan.link(f, node)?);
        }
        plan.chain(f, links)?;
        plan.accumulating_sinks(f)?;
        Ok(plan)
    }

    /// Records that `by` reads `child`.
    fn use_of(&mut self, ctx: *mut RxCtx, child: u32, by: u32) -> Result<(), Refusal> {
        let child = through(ctx, child)?;
        self.uses.entry(child).or_default().push(by);
        Ok(())
    }

    /// The use graph and scopes; the stores found, in the walk's order.
    fn walk(&mut self, f: &Frame) -> Result<Vec<u32>, Refusal> {
        let ctx = f.ctx();
        let mut stores = vec![];
        let mut seen: HashSet<u32> = HashSet::new();
        let mut stack: Vec<(u32, Option<u32>)> = f.nodes().map(|n| (n.h, None)).collect();
        stack.reverse();
        while let Some((h, scope)) = stack.pop() {
            if !seen.insert(h) {
                continue;
            }
            self.scope.insert(h, scope);
            if is_store(f, h)? {
                stores.push(h)
            }
            for child in door::operands(ctx, h)? {
                self.use_of(ctx, child, h)?
            }
            if door::class_is(ctx, h, "Loop")? {
                let mut roots = door::field_refs(ctx, h, "nexts")?.unwrap_or_default();
                roots.extend(door::field_refs(ctx, h, "exit_values")?.unwrap_or_default());
                roots.extend(door::field_refs(ctx, h, "exit_cond")?.unwrap_or_default());
                for root in roots {
                    self.use_of(ctx, root, h)?
                }
                for sink in door::field_refs(ctx, h, "body_sinks")?.unwrap_or_default() {
                    for operand in door::operands(ctx, sink)? {
                        self.use_of(ctx, operand, sink)?
                    }
                }
                let mut body = door::body_nodes(ctx, h)?;
                body.reverse();
                stack.extend(body.into_iter().map(|b| (b, Some(h))));
            }
        }
        Ok(stores)
    }

    /// Whether the store belongs to an accumulate chain over a `vec` or `splat` a sink commits link
    /// by link.
    fn is_committed(&self, f: &Frame, node: u32) -> Result<bool, Refusal> {
        let (mut chain, mut base) = (false, node);
        while OpKind::of_op(f.ctx(), base)? == Some(OpKind::AddAt) {
            chain = true;
            base = through(f.ctx(), door::operands(f.ctx(), base)?[0])?;
        }
        Ok(chain
            && matches!(
                OpKind::of_op(f.ctx(), base)?,
                Some(OpKind::Vec | OpKind::Splat)
            ))
    }

    /// The loop-body sinks that accumulate per iteration (an own-column assign of an `add_at` chain
    /// over a structural zero): the
    /// chain is committed link by link by its sink, never rendered as a value, and its plane is
    /// zeroed once before the walk.
    fn accumulating_sinks(&mut self, f: &Frame) -> Result<(), Refusal> {
        let mut pending: Vec<u32> = f
            .nodes()
            .filter(|n| n.class == Class::Loop)
            .map(|n| n.h)
            .collect();
        while let Some(lp) = pending.pop() {
            for h in door::body_nodes(f.ctx(), lp)? {
                if door::class_is(f.ctx(), h, "Loop")? {
                    pending.push(h)
                }
            }
            for sink in door::field_refs(f.ctx(), lp, "body_sinks")?.unwrap_or_default() {
                let Some(links) = accumulate_links(f, sink)? else {
                    continue;
                };
                let mut node = door::operands(f.ctx(), sink)?[0];
                for _ in &links {
                    self.skip.insert(node);
                    node = door::operands(f.ctx(), node)?[0]
                }
                let target = crate::spell::binding_name(
                    &text_field(f.ctx(), sink, "role")?,
                    &text_field(f.ctx(), sink, "name")?,
                )?;
                if !self.accumulated.iter().any(|a| a.0 == target) {
                    let width = crate::body::ty_of(f.ctx(), sink)?.shape.iter().product();
                    self.accumulated.push((target, width))
                }
            }
        }
        Ok(())
    }

    /// The nodes that read a node.
    fn users(&self, h: u32) -> &[u32] {
        self.uses.get(&h).map_or(&[][..], |u| &u[..])
    }

    /// The link a store node makes: its key split into a base and an offset, the components it
    /// reaches, and whether it is an accumulate.
    fn link(&self, f: &Frame, node: u32) -> Result<Link, Refusal> {
        let ctx = f.ctx();
        let operands = door::operands(ctx, node)?;
        let (base, key, mut value) = (
            through(ctx, operands[0])?,
            through(ctx, operands[1])?,
            through(ctx, operands[2])?,
        );
        let mut acc = OpKind::of_op(f.ctx(), node)? == Some(OpKind::AddAt);
        let (mut read, mut add) = (None, None);
        if !acc {
            if let Some((x, r, a)) = self.accumulate(f, node, base, key, value)? {
                acc = true;
                value = x;
                read = Some(r);
                add = Some(a)
            }
        }
        let (b, offset) = split_key(ctx, key)?;
        let width = crate::body::ty_of(ctx, node)?.shape[0] as i64;
        let (lo, hi) = match b {
            None => (offset, offset),
            Some(b) => {
                let (ilo, ihi) = interval(ctx, b)?;
                (
                    if ilo == f64::NEG_INFINITY {
                        0
                    } else {
                        0.max(ilo as i64 + offset)
                    },
                    if ihi == f64::INFINITY {
                        width - 1
                    } else {
                        (width - 1).min(ihi as i64 + offset)
                    },
                )
            }
        };
        Ok(Link {
            node,
            key,
            base: b,
            offset,
            lo,
            hi,
            acc,
            value,
            read,
            add,
        })
    }

    /// `(x, read, add)` when the store writes `base[key] + x` at `key`: `read` is the read of that
    /// slot the `add` alone uses.
    fn accumulate(
        &self,
        f: &Frame,
        node: u32,
        base: u32,
        key: u32,
        value: u32,
    ) -> Result<Option<(u32, u32, u32)>, Refusal> {
        let ctx = f.ctx();
        if !(OpKind::of_op(ctx, value)? == Some(OpKind::Add) && self.users(value) == [node]) {
            return Ok(None);
        }
        let operands = door::operands(ctx, value)?;
        for (read, x) in [(operands[0], operands[1]), (operands[1], operands[0])] {
            let read = through(ctx, read)?;
            if !(door::class_is(ctx, read, "Op")?
                && self.users(read) == [value]
                && through(ctx, door::operands(ctx, read)?[0])? == base)
            {
                continue;
            }
            let kind = OpKind::of(f.ctx(), read)?;
            let same = if kind == OpKind::ComponentAt {
                same_index(ctx, through(ctx, door::operands(ctx, read)?[1])?, key)?
            } else if kind == OpKind::Component && door::class_is(ctx, key, "Const")? {
                let wanted = number(ctx, key)?;
                matches!(component_index(ctx, read)?.as_slice(), [only] if *only as f64 == wanted)
            } else {
                false
            };
            if same {
                return Ok(Some((through(ctx, x)?, read, value)));
            }
        }
        Ok(None)
    }

    /// The users of a link's base that belong to the link itself are its only users.
    fn sole(&self, vec: u32, link: &Link) -> bool {
        let mut have: Vec<u32> = self.users(vec).to_vec();
        let mut owned = vec![link.node];
        owned.extend(link.read);
        have.sort();
        owned.sort();
        have == owned
    }

    /// Whether the plain base of a run is read only by the run's first store.
    pub(super) fn sole_reader(&self, vec: u32, first: u32, run: &[Link]) -> bool {
        run.iter()
            .find(|l| l.node == first)
            .is_some_and(|l| self.sole(vec, l))
    }

    /// Joins links into runs: a store whose only reader is the next store of the same scope belongs
    /// to that store's run.
    fn chain(&mut self, f: &Frame, links: Vec<Link>) -> Result<(), Refusal> {
        let ctx = f.ctx();
        let by_node: HashMap<u32, usize> =
            links.iter().enumerate().map(|(i, l)| (l.node, i)).collect();
        let mut below: HashMap<u32, u32> = HashMap::new();
        for link in &links {
            let base = through(ctx, door::operands(ctx, link.node)?[0])?;
            if by_node.contains_key(&base)
                && self.sole(base, link)
                && self.scope.get(&base) == self.scope.get(&link.node)
            {
                below.insert(base, link.node);
            }
        }
        for link in &links {
            if below.contains_key(&link.node) {
                self.skip.insert(link.node);
            }
            if let (Some(read), Some(add)) = (link.read, link.add) {
                self.skip.insert(read);
                self.skip.insert(add);
            }
            if link.base.is_some()
                && self
                    .users(link.key)
                    .iter()
                    .all(|&u| u == link.node || Some(u) == link.read)
            {
                self.inline.insert(link.key);
            }
        }
        for link in &links {
            if below.contains_key(&link.node) {
                continue;
            }
            let mut run = vec![link.clone()];
            loop {
                let base = through(ctx, door::operands(ctx, run[0].node)?[0])?;
                match by_node.get(&base) {
                    Some(&i) if below.get(&base) == Some(&run[0].node) => {
                        run.insert(0, links[i].clone())
                    }
                    _ => break,
                }
            }
            self.runs.insert(link.node, run);
        }
        Ok(())
    }
}

/// The (key, element) links of a loop-body sink that accumulates per iteration, or none for any
/// other sink.
pub(crate) fn accumulate_links(f: &Frame, sink: u32) -> Result<Option<Vec<(u32, u32)>>, Refusal> {
    let ctx = f.ctx();
    if !door::class_is(ctx, sink, "Assign")? || door::operands(ctx, sink)?.len() != 1 {
        return Ok(None);
    }
    let (mut node, mut links) = (door::operands(ctx, sink)?[0], vec![]);
    while OpKind::of_op(ctx, node)? == Some(OpKind::AddAt) {
        let operands = door::operands(ctx, node)?;
        links.push((operands[1], operands[2]));
        node = operands[0];
    }
    links.reverse();
    if links.is_empty() || !door::class_is(ctx, node, "Op")? {
        return Ok(None);
    }
    let slots = match OpKind::of(f.ctx(), node)? {
        OpKind::Splat | OpKind::Vec => door::operands(ctx, node)?,
        _ => return Ok(None),
    };
    for slot in slots {
        if !door::class_is(ctx, slot, "Const")? || number(ctx, slot)? != 0.0 {
            return Ok(None);
        }
    }
    Ok(Some(links))
}

/// The statements of a loop's body sinks: each link of a per-iteration accumulate on the own column
/// where it stands, and each log store of an ordered accumulate into its scratch plane. Any other
/// body sink is refused by name.
pub(crate) fn body_sinks(f: &mut Frame, lp: u32) -> Result<(), Refusal> {
    for sink in door::field_refs(f.ctx(), lp, "body_sinks")?.unwrap_or_default() {
        if door::class_is(f.ctx(), sink, "AccumWrite")? && door::operands(f.ctx(), sink)?.len() == 2
        {
            super::sinks::accum_of(f, sink)?;
            continue;
        }
        if door::class_is(f.ctx(), sink, "TapeWrite")? {
            super::sinks::tape_write_of(f, sink)?;
            continue;
        }
        if door::class_is(f.ctx(), sink, "LogStore")? {
            crate::body::team::store_in_loop(f, sink)?;
            continue;
        }
        let Some(links) = accumulate_links(f, sink)? else {
            return Err(nib(
                "a loop body sink that is not a per-iteration accumulate",
            ));
        };
        let target = crate::spell::binding_name(
            &text_field(f.ctx(), sink, "role")?,
            &text_field(f.ctx(), sink, "name")?,
        )?;
        for (key, element) in links {
            f.emit(format!(
                "{target}[i].at({}) += {};",
                f.arg(key)?,
                f.arg(element)?
            ))
        }
    }
    Ok(())
}

/// How a run's stores are written.
#[derive(Clone, Copy, PartialEq)]
enum Pick {
    /// Each store is one indexed access.
    Index,
    /// Each store is selects over the components it can reach.
    Select,
    /// One accumulate rebuilds the whole short vector.
    Lean,
}

/// What one add written as selects costs against an indexed access, for the scalar type, as the
/// unit's machine facts answer it.
fn select_weight(f: &Frame, scalar_type: &str) -> Result<f64, Refusal> {
    let weight = &f.unit.machine.select_weight;
    let held = if scalar_type == "float64" {
        weight.float64
    } else {
        weight.float32
    };
    held.ok_or_else(|| nib("a runtime-index store without the machine's select weight"))
}

/// The select weight when `Real` is eight bytes wide, and when four: the scalar type is chosen when
/// the unit is compiled.
fn weights(f: &Frame, dtype: &str) -> Result<(f64, f64), Refusal> {
    Ok(match dtype {
        "f64" => (select_weight(f, "float64")?, select_weight(f, "float32")?),
        "f32" => {
            let w = select_weight(f, "float32")?;
            (w, w)
        }
        _ => (1.0, 1.0),
    })
}

/// The stores of a run in order, each group a list: accumulates that share one runtime base and
/// take distinct offsets fuse.
fn groups(ctx: *mut RxCtx, links: &[Link]) -> Result<Vec<Vec<Link>>, Refusal> {
    let mut groups: Vec<Vec<Link>> = vec![];
    for link in links {
        if let Some(last) = groups.last_mut() {
            if link.acc
                && link.base.is_some()
                && last[0].acc
                && same_index(ctx, last[0].base.unwrap_or(0), link.base.unwrap_or(0))?
                && last.iter().all(|other| other.offset != link.offset)
            {
                last.push(link.clone());
                continue;
            }
        }
        groups.push(vec![link.clone()]);
    }
    Ok(groups)
}

/// The components some store of the group can write, ascending.
fn reach(group: &[Link]) -> Vec<i64> {
    let mut all: Vec<i64> = group.iter().flat_map(|l| l.lo..=l.hi).collect();
    all.sort();
    all.dedup();
    all
}

/// Element `k` of the mutable local: an item's own subscript, or the element of a plane's proxy.
fn at(f: &Frame, name: &str, k: &str) -> String {
    if f.proxies.iter().any(|x| x.name == name) {
        format!("{name}.at({k})")
    } else {
        format!("{name}({k})")
    }
}

/// Whether every key of the dynamic stores is proven equal over the lanes of an aligned run.
fn uniform(f: &Frame, keys: &[u32]) -> Result<bool, Refusal> {
    if keys.is_empty() {
        return Ok(false);
    }
    let flags =
        crate::ir::analysis::warp_uniform(f.ctx(), f.sinks(), f.unit.build.form.uniform_run, keys)?;
    Ok(flags.iter().all(|&x| x))
}

/// A store node: nothing for a store a run swallows, the run's statements at its last store, else
/// one copy written at the index.
pub(crate) fn store(f: &mut Frame, p: usize) -> Result<Out, Refusal> {
    let h = f.node(p).h;
    let Some(run) = f.stores.runs.get(&h).cloned() else {
        if f.stores.skip.contains(&h) {
            f.set_text(p, String::new());
            return Ok(Out::Bound);
        }
        let ty = f.node(p).ty.clone();
        let copy = f.names.store_copy();
        let name = copy.text.clone();
        let args: Vec<String> = f.opp(p).iter().map(|&q| f.text(q).clone()).collect();
        f.declare_var(copy, carry_type(&ty.dtype, &ty.shape)?, Some(&args[0]));
        let op = if OpKind::of(f.ctx(), h)? == OpKind::AddAt {
            "+="
        } else {
            "="
        };
        f.emit(format!(
            "{name}({}) {op} {};",
            args[1],
            placed::cast(&args[2], &ty.dtype)?
        ));
        return Ok(Out::Value(name, None));
    };
    emit_run(f, p, &run)
}

/// The statements of the run that ends at node `p`, and the name of the local that holds the
/// result.
fn emit_run(f: &mut Frame, p: usize, run: &[Link]) -> Result<Out, Refusal> {
    let ctx = f.ctx();
    let ty = f.node(p).ty.clone();
    let base = through(ctx, door::operands(ctx, run[0].node)?[0])?;
    let token_name = f.names.store_copy();
    let token = token_name.text.clone();
    let held = f.arg(base)?;
    let in_proxy = f.proxies.iter().any(|x| x.name == held);
    let carry = door::class_is(ctx, base, "LoopCarry")?
        && f.mutable_carries.contains(&base)
        && (f.stores.sole_reader(base, run[0].node, run) || in_proxy);
    let name = if carry || (in_proxy && f.stores.sole_reader(base, run[0].node, run)) {
        held.clone()
    } else {
        f.declare_var(token_name, carry_type(&ty.dtype, &ty.shape)?, Some(&held));
        token.clone()
    };
    let zero = if matches!(ty.dtype.as_str(), "f64" | "f32") {
        "-0.0"
    } else {
        "0"
    };
    let groups = groups(ctx, run)?;
    let picks = pick_styles(f, &ty, &name, &groups)?;
    let spelling = Spelling {
        ty: &ty,
        name: &name,
        token: &token,
        elem: &carry_type(&ty.dtype, &[])?,
        zero,
    };
    if picks.wide == picks.narrow {
        write(f, &spelling, &groups, picks.wide)?;
    } else {
        f.emit("if constexpr (sizeof(Real) == 8) {".into());
        f.buf.indent.push_str("    ");
        write(f, &spelling, &groups, picks.wide)?;
        f.buf.indent.truncate(f.buf.indent.len() - 4);
        f.emit("} else {".into());
        f.buf.indent.push_str("    ");
        write(f, &spelling, &groups, picks.narrow)?;
        f.buf.indent.truncate(f.buf.indent.len() - 4);
        f.emit("}".into());
    }
    Ok(Out::Value(name, None))
}

/// How a run is written when `Real` is eight bytes wide and when it is four.
struct Picks {
    /// The style when `Real` is eight bytes wide.
    wide: Pick,
    /// The style when `Real` is four bytes wide.
    narrow: Pick,
}

/// Chooses between indexed writes, selects and the lean form for a run, once for each width of
/// `Real`: selects when what they
/// touch costs no more than the budget the indexed accesses allow, indexed writes for a run held in
/// a plane.
fn pick_styles(
    f: &Frame,
    ty: &crate::body::Ty,
    name: &str,
    groups: &[Vec<Link>],
) -> Result<Picks, Refusal> {
    let dynamic: Vec<&Vec<Link>> = groups.iter().filter(|g| g[0].base.is_some()).collect();
    let accesses: usize = dynamic
        .iter()
        .flat_map(|g| g.iter())
        .map(|l| if l.acc { 2 } else { 1 })
        .sum();
    let touched: usize = dynamic.iter().map(|g| reach(g).len()).sum();
    let keys: Vec<u32> = dynamic
        .iter()
        .flat_map(|g| g.iter())
        .map(|l| l.key)
        .collect();
    let budget = SELECT_COST_RATIO
        * accesses as f64
        * if uniform(f, &keys)? {
            UNIFORM_ACCESS
        } else {
            1.0
        };
    let pick = |w: f64| {
        if !dynamic.is_empty() && touched as f64 * w <= budget {
            Pick::Select
        } else {
            Pick::Index
        }
    };
    let (w8, w4) = if dynamic.is_empty() {
        (1.0, 1.0)
    } else {
        weights(f, &ty.dtype)?
    };
    let (mut pick8, mut pick4) = (pick(w8), pick(w4));
    let lean_width = f
        .unit
        .build
        .form
        .lean_width
        .ok_or_else(|| nib("a runtime-index store without the form's lean width"))?;
    let lean = dynamic.len() == 1
        && dynamic[0].len() == 1
        && dynamic[0][0].acc
        && ty.shape[0] as i64 <= lean_width
        && matches!(ty.dtype.as_str(), "f64" | "f32");
    let in_plane = f.proxies.iter().any(|x| x.name == name);
    if in_plane {
        pick8 = Pick::Index;
        pick4 = Pick::Index
    }
    if lean && !in_plane {
        pick4 = Pick::Lean;
        if ty.dtype == "f32" {
            pick8 = pick4
        }
    }
    Ok(Picks {
        wide: pick8,
        narrow: pick4,
    })
}

/// What the statements of a run are spelled with: the value's type, the local written (`name`), the
/// prefix of the names the
/// run declares (`token`), the element type and the spelling of a zero of it.
struct Spelling<'a> {
    /// The type of the vector written.
    ty: &'a crate::body::Ty,
    /// The local (or plane proxy) written.
    name: &'a str,
    /// The prefix of the names the run declares.
    token: &'a str,
    /// The element type.
    elem: &'a str,
    /// The spelling of a zero of the element type.
    zero: &'a str,
}

/// `+=` for an accumulate, `=` for a plain store.
fn assign_op(acc: bool) -> &'static str {
    if acc {
        "+="
    } else {
        "="
    }
}

/// The statements of the groups of a run: as selects over the reachable components, or as indexed
/// writes.
fn write(f: &mut Frame, sp: &Spelling, groups: &[Vec<Link>], pick: Pick) -> Result<(), Refusal> {
    for (j, group) in groups.iter().enumerate() {
        let tag = format!("{}_{j}", sp.token);
        let mut names = vec![];
        for (i, link) in group.iter().enumerate() {
            let v = format!("{tag}_v{i}");
            let value = placed::cast(&f.arg(link.value)?, &sp.ty.dtype)?;
            f.declare_as(
                Name::new(v.clone(), NameKind::Local),
                sp.elem.to_string(),
                Some(&value),
            );
            names.push(v);
        }
        let first = &group[0];
        if first.base.is_none() {
            f.emit(format!(
                "{} {} {};",
                at(f, sp.name, &first.lo.to_string()),
                assign_op(first.acc),
                names[0]
            ));
        } else if pick == Pick::Lean {
            write_lean(f, sp, &tag, first, &names[0])?;
        } else if pick == Pick::Index {
            for (link, v) in group.iter().zip(&names) {
                let key = f.arg(link.key)?;
                f.emit(format!(
                    "{} {} {v};",
                    at(f, sp.name, &key),
                    assign_op(link.acc)
                ));
            }
        } else if group.len() == 1 {
            write_selects(f, sp.name, &tag, first, &names[0])?;
        } else {
            write_fused_selects(f, sp, &tag, group, &names)?;
        }
    }
    Ok(())
}

/// One accumulate at a runtime key into a short vector: the vector is rebuilt from the old one with
/// the sum in the key's slot.
fn write_lean(
    f: &mut Frame,
    sp: &Spelling,
    tag: &str,
    first: &Link,
    value: &str,
) -> Result<(), Refusal> {
    let (ty, name) = (sp.ty, sp.name);
    let (k, old) = (format!("{tag}_k"), format!("{tag}_old"));
    let key = f.arg(first.key)?;
    f.declare_auto(Name::new(k.clone(), NameKind::Key), &key);
    f.declare_as(
        Name::new(old.clone(), NameKind::Local),
        carry_type(&ty.dtype, &ty.shape)?,
        Some(name),
    );
    f.declare_as(
        Name::new(format!("{tag}_sum"), NameKind::Local),
        sp.elem.to_string(),
        Some(&format!("{old}({k}) + {value}")),
    );
    let parts: Vec<String> = (0..ty.shape[0])
        .map(|c| format!("({k} == {c} ? {tag}_sum : {old}({c}))"))
        .collect();
    f.emit(format!(
        "{name} = {}{{{}}};",
        carry_type(&ty.dtype, &ty.shape)?,
        parts.join(", ")
    ));
    Ok(())
}

/// One store at a runtime key as selects: each component the key can reach takes the new value when
/// the key names it.
fn write_selects(
    f: &mut Frame,
    name: &str,
    tag: &str,
    first: &Link,
    value: &str,
) -> Result<(), Refusal> {
    let k = format!("{tag}_k");
    let key = f.arg(first.key)?;
    f.declare_auto(Name::new(k.clone(), NameKind::Key), &key);
    for c in first.lo..=first.hi {
        let new = if first.acc {
            format!("{name}({c}) + {value}")
        } else {
            value.to_string()
        };
        f.emit(format!("{name}({c}) = {k} == {c} ? {new} : {name}({c});"));
    }
    Ok(())
}

/// Accumulates that share one runtime base at distinct offsets, fused: each reachable component
/// adds the value whose offset
/// the base names.
fn write_fused_selects(
    f: &mut Frame,
    sp: &Spelling,
    tag: &str,
    group: &[Link],
    names: &[String],
) -> Result<(), Refusal> {
    let (name, elem, zero) = (sp.name, sp.elem, sp.zero);
    let k = format!("{tag}_k");
    let base = f.arg(group[0].base.unwrap_or(0))?;
    f.declare_auto(Name::new(k.clone(), NameKind::Key), &base);
    for c in reach(group) {
        let mut chosen = format!("static_cast<{elem}>({zero})");
        for (link, v) in group.iter().zip(names).rev() {
            if link.lo <= c && c <= link.hi {
                chosen = format!("{k} == {} ? {v} : {chosen}", c - link.offset)
            }
        }
        f.emit(format!("{name}({c}) += {chosen};"));
    }
    Ok(())
}
