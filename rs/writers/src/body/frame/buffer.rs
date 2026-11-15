//! The buffer a frame emits into: the lines written so far, the indent of the next one and the
//! declarations those lines hold. The declaration table lives with its buffer and is never
//! truncated when a loop or case scope closes (the lines of a closed scope stay in the buffer and
//! so do their declarations); a buffer scope puts both aside together.
use super::namer::Name;
use crate::body::{Class, Frame};
use crate::contract::types::Scalar;
use crate::ir::door;
use crate::refuse::Refusal;
use crate::spell;
use std::collections::BTreeSet;

/// One declaration the frame wrote: what it named, its C++ type, where it sits and what it holds.
#[derive(Clone)]
pub struct Decl {
    /// The declared name and its kind.
    pub name: Name,
    /// The C++ type as written (`auto`, or an `aether::Item<..>`).
    pub cty: String,
    /// The index of the declaration's line in the buffer.
    pub line: usize,
    /// The text the name was initialised with; none for an unset declaration.
    pub value: Option<String>,
    /// Whether the declaration is a single line.
    pub one_line: bool,
    /// Whether the declaration is a constant (`const ...`), not a variable.
    pub constant: bool,
}

/// A line of a buffer the writer marked for a later reader.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mark {
    /// The line that counts a finished sample.
    FinishCount,
}

/// The lines of one render, the indent of the next line and the declarations written.
#[derive(Default, Clone)]
pub struct Buffer {
    /// The lines written, each led by the indent it was written at.
    pub lines: Vec<String>,
    /// The declared names each line was built from, one set per line.
    pub reads: Vec<BTreeSet<String>>,
    /// The indent of the next line.
    pub indent: String,
    /// The declarations the lines hold, in order.
    pub decls: Vec<Decl>,
    /// The lines the writer marked, with their index.
    pub marks: Vec<(Mark, usize)>,
}

impl Buffer {
    /// A buffer with the given lines and indent and no declarations.
    pub fn new(lines: Vec<String>, indent: String) -> Buffer {
        Buffer {
            reads: vec![BTreeSet::new(); lines.len()],
            lines,
            indent,
            decls: vec![],
            marks: vec![],
        }
    }

    /// The lines joined by newlines.
    pub fn text(&self) -> String {
        self.lines.join("\n")
    }

    /// Whether the buffer has no lines.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// The lines from `start` to `end` as a buffer of their own, with the declarations and marks
    /// they hold (their line indexes counted from `start`).
    pub fn slice(&self, start: usize, end: usize) -> Buffer {
        let inside = |line: usize| line >= start && line < end;
        Buffer {
            lines: self.lines[start..end].to_vec(),
            reads: self.reads[start..end].to_vec(),
            indent: self.indent.clone(),
            decls: self
                .decls
                .iter()
                .filter(|d| inside(d.line))
                .map(|d| Decl {
                    line: d.line - start,
                    ..d.clone()
                })
                .collect(),
            marks: self
                .marks
                .iter()
                .filter(|m| inside(m.1))
                .map(|m| (m.0, m.1 - start))
                .collect(),
        }
    }

    /// The names of the constants the buffer declares with a value.
    pub fn declared(&self) -> BTreeSet<String> {
        self.decls
            .iter()
            .filter(|d| d.constant && d.value.is_some())
            .map(|d| d.name.text.clone())
            .collect()
    }

    /// The names of every declaration of the buffer, set or unset.
    pub fn declared_all(&self) -> BTreeSet<String> {
        self.decls.iter().map(|d| d.name.text.clone()).collect()
    }

    /// The first declaration of a name that has a value.
    pub fn decl_of(&self, name: &str) -> Option<&Decl> {
        self.decls
            .iter()
            .find(|d| d.name.text == name && d.constant && d.value.is_some())
    }

    /// Which of `names` the lines read, from the records of what each line was built from.
    pub fn reads_of(&self, names: &BTreeSet<String>) -> BTreeSet<String> {
        self.reads
            .iter()
            .flatten()
            .filter(|name| names.contains(*name))
            .cloned()
            .collect()
    }

    /// The declarations, in the buffer's order, of `names` and of every value declared in the
    /// buffer that their values read: empty when one of them is missing or not a single line.
    pub fn one_line_chain(&self, names: &[String]) -> Vec<&Decl> {
        let (mut want, mut stack) = (BTreeSet::new(), names.to_vec());
        while let Some(name) = stack.pop() {
            let Some(decl) = self.decl_of(&name) else {
                return vec![];
            };
            if !decl.one_line {
                return vec![];
            }
            if want.insert(name) {
                let value = decl.value.as_deref().unwrap_or("");
                stack.extend(
                    tokens(value)
                        .filter(|w| self.decl_of(w).is_some())
                        .map(String::from),
                );
            }
        }
        self.decls
            .iter()
            .filter(|d| d.constant && d.value.is_some() && want.contains(&d.name.text))
            .collect()
    }

    /// The text with the marked finish-count line rewritten: `from` becomes `to` on that line.
    pub fn with_finish_counted(&self, to: &str, from: &str) -> String {
        let lines: Vec<String> = self
            .lines
            .iter()
            .enumerate()
            .map(|(at, line)| {
                if self.marks.contains(&(Mark::FinishCount, at)) {
                    line.replace(from, to)
                } else {
                    line.clone()
                }
            })
            .collect();
        lines.join("\n")
    }

    /// The text without the lines at the given indexes.
    pub fn text_without(&self, lines: &[usize]) -> String {
        let kept: Vec<&str> = (0..self.lines.len())
            .filter(|at| !lines.contains(at))
            .map(|at| self.lines[at].as_str())
            .collect();
        kept.join("\n")
    }
}

/// The words of a text.
pub(crate) fn tokens(text: &str) -> impl Iterator<Item = &str> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| !w.is_empty())
}

impl Frame<'_> {
    /// Whether the unit's scalar mode is the banded one.
    pub(crate) fn banded(&self) -> bool {
        self.unit.scalar == Scalar::Banded
    }

    /// Whether a banded local of the node at `p` flows on unnormalised: a computed value whose every reader is an operation that
    /// takes a raw band (a plane element is decoded once, and a sink, carry or exit reads limbs in order).
    pub(crate) fn flows_raw(&self, p: usize, held: &[u32]) -> bool {
        let n = self.node(p);
        self.banded()
            && !matches!(n.class, Class::Leaf | Class::At)
            && !held.contains(&n.h)
            && !self.users(p).is_empty()
            && self
                .users(p)
                .iter()
                .all(|&(u, _)| self.node(u).kind.is_some_and(spell::reads_raw))
    }

    /// The handles of the values the walk's sinks commit (a sink reads limbs in order).
    pub(crate) fn sink_values(&self) -> Result<Vec<u32>, Refusal> {
        let mut out = vec![];
        for &s in self.sinks() {
            out.extend(door::operands(self.ctx(), s)?);
        }
        Ok(out)
    }

    /// Declares a typed local through `spell::declare` and records it in the buffer.
    pub(crate) fn declare(
        &mut self,
        name: Name,
        dtype: &str,
        shape: &[usize],
        value: &str,
        held: bool,
        raw: bool,
    ) -> Result<(), Refusal> {
        let mode = if self.banded() && !raw {
            spell::Mode::Banded
        } else {
            spell::Mode::Plain
        };
        let line = spell::declare(&name.text, dtype, shape, value, held, mode)?;
        let cty = spell::declared_type(dtype, shape, held, mode)?;
        self.record(name, cty, Some(value.to_string()), line, true);
        Ok(())
    }

    /// Declares a local of inferred type and records it.
    pub(crate) fn declare_auto(&mut self, name: Name, value: &str) {
        self.declare_as(name, "auto".to_string(), Some(value));
    }

    /// Declares a local of the given C++ type (unset when `value` is none) and records it.
    pub(crate) fn declare_as(&mut self, name: Name, cty: String, value: Option<&str>) {
        let line = match value {
            Some(value) => format!("const {cty} {} = {value};", name.text),
            None => format!("{cty} {};", name.text),
        };
        self.record(name, cty, value.map(str::to_string), line, true);
    }

    /// Declares a variable (not a constant) of the given C++ type, set from `value` when there is
    /// one.
    pub(crate) fn declare_var(&mut self, name: Name, cty: String, value: Option<&str>) {
        let line = match value {
            Some(value) => format!("{cty} {} = {value};", name.text),
            None => format!("{cty} {};", name.text),
        };
        self.record(name, cty, value.map(str::to_string), line, false);
    }

    /// Marks the line just emitted for a later reader.
    pub(crate) fn mark_last(&mut self, mark: Mark) {
        let at = self.buf.lines.len() - 1;
        self.buf.marks.push((mark, at));
    }

    /// Writes the declaration line and notes it in the buffer's table.
    fn record(
        &mut self,
        name: Name,
        cty: String,
        value: Option<String>,
        line: String,
        constant: bool,
    ) {
        let decl = Decl {
            name,
            cty,
            line: self.buf.lines.len(),
            value,
            one_line: !line.contains('\n'),
            constant,
        };
        let own = decl.name.text.clone();
        self.table.note_declared(&own);
        self.emit(line);
        if let Some(last) = self.buf.reads.last_mut() {
            last.insert(own);
        }
        self.buf.decls.push(decl);
    }
}
