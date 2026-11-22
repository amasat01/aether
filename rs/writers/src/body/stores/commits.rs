//! Per-component commits: an own-column assign of a `vec` (or a `splat`) stores component k right
//! after the statement that defines
//! the part, so the wide value is never live whole. Planned only when no leaf reads the plane (its
//! prior value would see the early
//! store).
use crate::body::values::text_field;
use crate::body::Class;
use crate::body::{render_pos, Frame};
use crate::kind::OpKind;
use crate::refuse::Refusal;
use crate::spell::binding_name;

/// What the plan decided: the stores due after a position renders, the positions no longer rendered
/// whole, the sinks committed.
pub struct Plan {
    /// The stores due once a position has rendered.
    pub due: Vec<Due>,
    /// Whether a position is not rendered whole, by position.
    pub skip: Vec<bool>,
    /// The sinks (by handle) committed per component.
    pub held: Vec<u32>,
    /// The (plane, component) pairs the prologue zeroes before an accumulate.
    pub zero: Vec<(String, String)>,
}

/// One store due once a position is rendered: the component it writes (`at`), the value's position,
/// how it writes (`=`, `+=`, or
/// `at+=` for an accumulate at a runtime index, whose key is `key`).
#[derive(Clone)]
pub struct Due {
    /// The position whose render makes the store due.
    pub when: usize,
    /// The binding of the plane written.
    pub target: String,
    /// The subscript text of the component written (empty for an accumulate at a runtime key).
    pub at: String,
    /// The position of the value written.
    pub part: usize,
    /// The assignment operator: `=`, `+=`, or `at+=`.
    pub op: &'static str,
    /// The position of the runtime key of an `at+=` store.
    pub key: Option<usize>,
}

/// The component indices of a shape in row-major order, each as the text of its comma-separated
/// subscripts.
fn flat_indices(shape: &[usize]) -> Vec<String> {
    let mut out: Vec<Vec<usize>> = vec![vec![]];
    for &e in shape {
        out = out
            .iter()
            .flat_map(|i| {
                (0..e).map(move |k| {
                    let mut v = i.clone();
                    v.push(k);
                    v
                })
            })
            .collect()
    }
    out.iter()
        .map(|i| {
            i.iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .collect()
}

/// The position whose render makes a node renderable: the latest of what it reads (-1: it reads
/// none), a loop at its own position.
fn ready_times(f: &Frame) -> Vec<i64> {
    let mut ready: Vec<i64> = vec![];
    for pos in 0..f.count() {
        let ps = f.opp(pos);
        let at = ps.iter().map(|&c| ready[c]).max().unwrap_or(-1);
        ready.push(if f.node(pos).class == Class::Loop {
            at.max(pos as i64)
        } else {
            at
        });
    }
    ready
}

/// The position at whose render a part can be rendered and stored.
fn ready_at(f: &Frame, part: usize, ready: &[i64], late: i64, memo: &mut Vec<Option<i64>>) -> i64 {
    if part as i64 <= late {
        return if ready[part] < 0 {
            part as i64
        } else {
            ready[part].min(part as i64)
        };
    }
    if let Some(m) = memo[part] {
        return m;
    }
    let at = f
        .opp(part)
        .iter()
        .map(|&c| ready_at(f, c, ready, late, memo))
        .max()
        .unwrap_or(0);
    memo[part] = Some(at);
    at
}

/// A sink that commits per component: its chain of positions from the value down to the base, its
/// static parts, its runtime
/// links (key, value), the plane it writes and the component indices of each part.
struct Candidate {
    /// The sink (by handle) that commits the value.
    sink: u32,
    /// The positions of the accumulate chain, from the sink value down to its base.
    chain: Vec<usize>,
    /// The positions of the static parts, one per component.
    parts: Vec<usize>,
    /// The runtime accumulates as (key position, value position), in the order they apply.
    links: Vec<(usize, usize)>,
    /// The binding of the plane the sink writes.
    target: String,
    /// The subscript text of each component, parallel to `parts`.
    indices: Vec<String>,
}

/// What the candidates' parts and links are due to write, and how many candidates read each
/// position.
struct Schedule {
    /// The stores and the positions they are due at.
    due: Vec<Due>,
    /// The (plane, component) pairs the prologue zeroes before an accumulate.
    zero: Vec<(String, String)>,
    /// How many candidates read each position through their chain.
    users: Vec<usize>,
}

/// The sinks that commit per component: an own-column assign of a `vec` (or `splat`) under a chain
/// of runtime accumulates. Each
/// static part is stored where it is ready (a plain store before the first accumulate, else an
/// accumulate over a zero the
/// prologue wrote), each link where its index and value are.
pub(crate) fn plan(f: &Frame, read: &[(String, String)]) -> Result<Plan, Refusal> {
    let ready = ready_times(f);
    let planned = candidates(f, read)?;
    let schedule = schedule(f, &planned, &ready);
    Ok(Plan {
        due: schedule.due,
        skip: skipped(f, &planned, &schedule.users),
        held: planned.iter().map(|x| x.sink).collect(),
        zero: schedule.zero,
    })
}

/// The sinks that commit per component, in the order of the frame's sinks.
fn candidates(f: &Frame, read: &[(String, String)]) -> Result<Vec<Candidate>, Refusal> {
    let mut planned = vec![];
    for &s in f.sinks() {
        if let Some(candidate) = candidate(f, s, read)? {
            planned.push(candidate)
        }
    }
    Ok(planned)
}

/// The sink as a per-component candidate, or none when it is committed whole: not an own-column
/// assign of a `vec` or `splat`,
/// its plane read by a leaf, a plane the compensated kind commits whole, or a part or link computed
/// inside a scope.
fn candidate(f: &Frame, s: u32, read: &[(String, String)]) -> Result<Option<Candidate>, Refusal> {
    let p = f.at(s)?;
    let n = &f.node(p);
    // a warp store whose value every lane holds whole commits like an assign
    let alike = n.class == Class::WarpStore && f.node(f.opp(p)[0]).ty.shape == n.ty.shape;
    if !(n.class == Class::Assign || alike) || n.ops.len() != 1 {
        return Ok(None);
    }
    let (mut node, mut chain, mut links) = (f.opp(p)[0], vec![f.opp(p)[0]], vec![]);
    // a chain of stores at one index each; an accumulate at several indices is not a link
    while f.node(node).kind == Some(OpKind::AddAt) && f.node(f.opp(node)[0]).rank() == 1 {
        links.push((f.opp(node)[1], f.opp(node)[2]));
        node = f.opp(node)[0];
        chain.push(node);
    }
    links.reverse();
    let base = node;
    let parts: Vec<usize> = match f.node(base).kind {
        Some(OpKind::Vec) => f.opp(base).clone(),
        Some(OpKind::Splat) => vec![f.opp(base)[0]; f.node(base).numel()],
        _ => return Ok(None),
    };
    let (role, name) = (
        text_field(f.ctx(), n.h, "role")?,
        text_field(f.ctx(), n.h, "name")?,
    );
    if read.contains(&(role.clone(), name.clone())) {
        return Ok(None);
    }
    // under the compensated kind the correction plane and the kind's own-column target are
    // committed whole
    if let Some(comp) = &f.compensated {
        if comp.plane == name
            || comp
                .target
                .as_ref()
                .is_some_and(|t| t.own_column && t.name == name)
        {
            return Ok(None);
        }
    }
    if parts.iter().any(|&q| !f.is_outside(q))
        || links
            .iter()
            .any(|&(k, e)| !f.is_outside(k) || !f.is_outside(e))
    {
        return Ok(None);
    }
    for &(k, e) in &links {
        if f.node(k).class == Class::Dispatch && f.node(e).class == Class::Dispatch {
            return Err(crate::refuse::nib(
                "an accumulate link inside the cases of a fused switch",
            ));
        }
    }
    Ok(Some(Candidate {
        sink: s,
        chain,
        parts,
        links,
        target: binding_name(&role, n.ty.shape.len(), &name)?,
        indices: flat_indices(&n.ty.shape),
    }))
}

/// The position each store is due at: a static part where it is ready, a link where its key and
/// value are.
fn schedule(f: &Frame, planned: &[Candidate], ready: &[i64]) -> Schedule {
    let late = planned
        .iter()
        .map(|x| x.chain.last().map_or(-1, |&b| b as i64))
        .min()
        .unwrap_or(-1);
    let mut memo = vec![None; f.count()];
    let (mut due, mut zero) = (vec![], vec![]);
    let mut users = vec![0usize; f.count()];
    for candidate in planned {
        let Candidate {
            chain,
            parts,
            links,
            target,
            indices,
            ..
        } = candidate;
        let due_links: Vec<i64> = links
            .iter()
            .map(|&(k, e)| {
                ready_at(f, k, ready, late, &mut memo).max(ready_at(f, e, ready, late, &mut memo))
            })
            .collect();
        let first = due_links.iter().min().copied();
        for (at, &part) in indices.iter().zip(parts) {
            let when = ready_at(f, part, ready, late, &mut memo);
            let plain = first.is_none_or(|first| when < first);
            if !plain {
                zero.push((target.clone(), at.clone()))
            }
            due.push(Due {
                when: when as usize,
                target: target.clone(),
                at: at.clone(),
                part,
                op: if plain { "=" } else { "+=" },
                key: None,
            });
        }
        for (&when, &(k, e)) in due_links.iter().zip(links) {
            due.push(Due {
                when: when as usize,
                target: target.clone(),
                at: String::new(),
                part: e,
                op: "at+=",
                key: Some(k),
            });
        }
        for &q in chain {
            users[q] += 1
        }
    }
    Schedule { due, zero, users }
}

/// The positions no longer rendered whole: those only the candidates' chains read. A node read
/// whole needs every node under it
/// rendered.
fn skipped(f: &Frame, planned: &[Candidate], users: &[usize]) -> Vec<bool> {
    let mut skip: Vec<bool> = (0..f.count())
        .map(|p| users[p] > 0 && users[p] == f.fanout(p))
        .collect();
    for candidate in planned {
        let mut whole = false;
        for &q in &candidate.chain {
            whole = whole || !skip[q];
            if whole {
                skip[q] = false
            }
        }
    }
    skip
}

/// The stores due now that `pos` is rendered. A part whose inputs are all there is rendered here
/// (with what it still needs, in walk
/// order) rather than at its turn in the walk.
pub(crate) fn flush(f: &mut Frame, plan: &mut Plan, pos: usize) -> Result<(), Refusal> {
    let mut mine = vec![];
    plan.due.retain(|d| {
        if d.when == pos {
            mine.push(d.clone());
            false
        } else {
            true
        }
    });
    for d in mine {
        let (mut need, mut stack) = (vec![], vec![d.part]);
        stack.extend(d.key);
        while let Some(p) = stack.pop() {
            if f.is_done(p) || need.contains(&p) {
                continue;
            }
            need.push(p);
            stack.extend(f.opp(p).iter().copied());
        }
        need.sort();
        for p in need {
            render_pos(f, plan, p)?
        }
        let line = match d.key {
            Some(key) => format!("{}[i].at({}) += {};", d.target, f.text(key), f.text(d.part)),
            None => format!(
                "{}({}, i.global()) {} {};",
                d.target,
                d.at,
                d.op,
                f.text(d.part)
            ),
        };
        f.emit(line);
    }
    Ok(())
}
