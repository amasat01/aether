//! The class of a node, built once from the door's class name. A class the writers do not name
//! stays `Other` with its name, so it is refused exactly as an unnamed class always was.
use crate::refuse::Refusal;

/// The node classes the writers tell apart.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Class {
    /// A kernel argument: a plane or a scalar read by its role and name.
    Leaf,
    /// An indexed read of a plane.
    At,
    /// The extent of one axis of a plane.
    AxisExtent,
    /// The index of the sample being computed.
    SampleIndex,
    /// The position of the sample within its batch.
    SamplePosition,
    /// A choice between two values by a condition.
    Select,
    /// A literal value.
    Const,
    /// A value fixed by the role of an argument.
    RoleConst,
    /// An arithmetic, comparison or reduction operation, named by its kind.
    Op,
    /// A counted loop with carried values.
    Loop,
    /// The value a loop gives back after its last trip.
    LoopValue,
    /// The trip counter of a loop.
    LoopCount,
    /// The index variable of a loop body.
    LoopIndex,
    /// A value carried from one trip of a loop to the next.
    LoopCarry,
    /// A read of a tape row recorded by a loop.
    TapeRead,
    /// A write of a tape row by a loop.
    TapeWrite,
    /// An update of the operation `store`: a lane-wide value written into a plane.
    StoreUpdate,
    /// A choice among branches by a clamped selector.
    Dispatch,
    /// A dispatch that evaluates every branch and chooses by a chain.
    PredicatedDispatch,
    /// A dispatch whose strategy is picked and counted at run time.
    AutoDispatch,
    /// A read of a value shared by the lanes of a team.
    TeamRead,
    /// A copy into a value shared by the lanes of a team.
    TeamCopy,
    /// A store of the lanes' results into a shared value.
    TeamStore,
    /// A log of scattered writes, replayed at commit.
    ScatterLog,
    /// A store that goes through a write log.
    LogStore,
    /// A store of a boxed value.
    BoxStore,
    /// The lane's place in its warp.
    WarpLane,
    /// A store of a value held sliced across the lanes of a warp.
    WarpStore,
    /// A primitive a loop body calls for its effect.
    Primitive,
    /// An assignment to a plane or a mutable argument.
    Assign,
    /// A sum over the samples of a batch.
    SampleSum,
    /// An update that accumulates into a plane: an add or a bit-or.
    AccumUpdate,
    /// The end of a sample: counts a finished sample.
    Finish,
    /// Any other class: only its name is kept.
    Other(String),
}

impl Class {
    /// The class of the node class name the door answers.
    pub fn from_name(name: &str) -> Class {
        match name {
            "Leaf" => Class::Leaf,
            "At" => Class::At,
            "AxisExtent" => Class::AxisExtent,
            "SampleIndex" => Class::SampleIndex,
            "SamplePosition" => Class::SamplePosition,
            "Select" => Class::Select,
            "Const" => Class::Const,
            "RoleConst" => Class::RoleConst,
            "Op" => Class::Op,
            "Loop" => Class::Loop,
            "LoopValue" => Class::LoopValue,
            "LoopCount" => Class::LoopCount,
            "LoopIndex" => Class::LoopIndex,
            "LoopCarry" => Class::LoopCarry,
            "TapeRead" => Class::TapeRead,
            "TapeWrite" => Class::TapeWrite,
            "Dispatch" => Class::Dispatch,
            "PredicatedDispatch" => Class::PredicatedDispatch,
            "AutoDispatch" => Class::AutoDispatch,
            "TeamRead" => Class::TeamRead,
            "TeamCopy" => Class::TeamCopy,
            "TeamStore" => Class::TeamStore,
            "ScatterLog" => Class::ScatterLog,
            "LogStore" => Class::LogStore,
            "BoxStore" => Class::BoxStore,
            "WarpLane" => Class::WarpLane,
            "WarpStore" => Class::WarpStore,
            "Primitive" => Class::Primitive,
            "Assign" => Class::Assign,
            "SampleSum" => Class::SampleSum,
            "Finish" => Class::Finish,
            other => Class::Other(other.to_string()),
        }
    }

    /// The class of a node, by the name the door answers for it.
    pub fn of(ctx: *mut crate::ir::door::RxCtx, node: u32) -> Result<Class, Refusal> {
        let name = crate::ir::door::class_name(ctx, node)?;
        if name == "Update" {
            // one IR class, two emitters: a store writes the element, every other operation accumulates into it
            let op = crate::ir::door::text_field(ctx, node, "op")?;
            return Ok(if op == "store" {
                Class::StoreUpdate
            } else {
                Class::AccumUpdate
            });
        }
        Ok(Class::from_name(&name))
    }

    /// The door's name of the class.
    pub fn as_str(&self) -> &str {
        match self {
            Class::Leaf => "Leaf",
            Class::At => "At",
            Class::AxisExtent => "AxisExtent",
            Class::SampleIndex => "SampleIndex",
            Class::SamplePosition => "SamplePosition",
            Class::Select => "Select",
            Class::Const => "Const",
            Class::RoleConst => "RoleConst",
            Class::Op => "Op",
            Class::Loop => "Loop",
            Class::LoopValue => "LoopValue",
            Class::LoopCount => "LoopCount",
            Class::LoopIndex => "LoopIndex",
            Class::LoopCarry => "LoopCarry",
            Class::TapeRead => "TapeRead",
            Class::TapeWrite => "TapeWrite",
            Class::StoreUpdate => "store update",
            Class::Dispatch => "Dispatch",
            Class::PredicatedDispatch => "PredicatedDispatch",
            Class::AutoDispatch => "AutoDispatch",
            Class::TeamRead => "TeamRead",
            Class::TeamCopy => "TeamCopy",
            Class::TeamStore => "TeamStore",
            Class::ScatterLog => "ScatterLog",
            Class::LogStore => "LogStore",
            Class::BoxStore => "BoxStore",
            Class::WarpLane => "WarpLane",
            Class::WarpStore => "WarpStore",
            Class::Primitive => "Primitive",
            Class::Assign => "Assign",
            Class::SampleSum => "SampleSum",
            Class::AccumUpdate => "accumulating update",
            Class::Finish => "Finish",
            Class::Other(name) => name,
        }
    }
}

impl std::fmt::Display for Class {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.write_str(self.as_str())
    }
}
