//! The typed description of one render request: what the writers need to know about a unit to
//! write its source text. Closed sets are enums, optional parts are `Option`, and no field holds
//! JSON text or a name from another program. Nothing here knows how a request is recorded.

/// The group of an active set whose request leaves it out.
pub const DEFAULT_GROUP: i64 = 1;
/// The first position of a run segment whose request leaves it out.
pub const DEFAULT_RUN_START: i64 = 0;
/// The threads of a thread box, and of a team, whose request leaves them out.
pub const DEFAULT_THREADS: i64 = 1;
/// The stride of a thread box whose request leaves it out.
pub const DEFAULT_STRIDE: i64 = 1;
/// The samples written per row when the request leaves it out.
pub const DEFAULT_ROWS: i64 = 1;
/// The warps of a team, and of a build, whose request leaves them out.
pub const DEFAULT_WARPS: i64 = 1;
/// The banded-mode row width when the request leaves it out: none.
pub const DEFAULT_CELLS_ROW: i64 = 0;

/// The version of the typed contract: carried in the JSON as `"v"` and checked when the library is
/// opened. It moves whenever a field or a member is added, removed or renamed.
pub const CONTRACT_VERSION: u32 = 1;

/// The machine and kind of build a unit is written for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Target {
    /// A host function, compiled for the machine the build runs on.
    Host,
    /// A device kernel.
    Device,
}

/// The scalar type of the unit's values: what `Real` is spelled as in the emitted code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// `double`.
    Float64,
    /// `float`.
    Float32,
    /// A banded accumulator type; the unit carries cells instead of plain numbers.
    Banded,
}

/// The element type of one argument slot of the entry function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype {
    /// 64-bit float, `Real` when the scalar is `Float64`.
    F64,
    /// 32-bit float, `Real` when the scalar is `Float32`.
    F32,
    /// The unit's `Int` as a signed 32-bit value.
    I32,
    /// The unit's `Int` as a signed 64-bit value.
    I64,
    /// A plane of `std::int32_t`.
    Int32,
    /// A plane of `std::uint32_t`.
    Uint32,
    /// A plane of `bool`.
    Bool,
}

/// What an argument slot of the entry function holds; decides how it is declared and read.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    /// A plane the unit reads and writes in place.
    Mutable,
    /// A plane read per sample, whatever the rank of its value (the slot's shape says it).
    PlaneIn,
    /// A table indexed by a run-time value.
    Lookup,
    /// A plane the unit writes: a store target, or an accumulate target the runtime zeroes before the run.
    PlaneOut,
    /// A value shared by all samples, passed by value.
    Uniform,
    /// A mask that marks a sample as finished.
    Terminated,
    /// An input with several values per sample.
    WideIn,
    /// The number of samples, passed by value.
    Nsamples,
    /// A plain output plane.
    Out,
    /// The extent of one runtime axis of a plane, passed by value; the runtime reads it from the bound array.
    Extent,
}

/// How the unit's loads and stores relate one sample to the others.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Access {
    /// Each sample reads and writes only its own position.
    SampleLocal,
    /// A sample reads positions of other samples.
    CrossSampleRead,
    /// A sample writes positions of other samples.
    CrossSampleWrite,
}

/// One argument of the entry function, in declaration order.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    /// What the slot holds.
    pub role: Role,
    /// The slot's name; the C++ binding is spelled from it.
    pub name: String,
    /// The element type.
    pub dtype: Dtype,
    /// The extents of the slot, empty for a scalar.
    pub shape: Vec<i64>,
    /// The position of the slot in the entry's argument list.
    pub index: i64,
    /// The axes of a plane the slot declares, in storage order; none for a plane declared without axes.
    pub axes: Option<Vec<Axis>>,
    /// Whether the runtime zeroes the plane before the run (a `plane_out` slot an accumulate writes); false for every other slot.
    pub zeroed: bool,
}

/// One axis of a plane: its label and its extent, none when the extent is known only at run time.
#[derive(Debug, Clone, PartialEq)]
pub struct Axis {
    /// The axis's label.
    pub label: String,
    /// The axis's extent, at least 1; none for a runtime extent.
    pub extent: Option<i64>,
}

/// The argument list of a unit and how it is accessed.
#[derive(Debug, Clone, PartialEq)]
pub struct Walk {
    /// How the unit's accesses relate samples.
    pub access: Access,
    /// The name of the slot that receives the finishing value, when the unit has one.
    pub finish: Option<String>,
    /// The arguments in declaration order.
    pub slots: Vec<Slot>,
}

/// The mask names that skip a lane's commit.
#[derive(Debug, Clone, PartialEq)]
pub enum Guard {
    /// The `terminated` mask only.
    Default,
    /// No mask: every lane commits.
    DataOnly,
    /// These masks; a lane is skipped when any is set.
    Masks(Vec<String>),
}

/// How the unit's accumulating outputs are committed.
#[derive(Debug, Clone, PartialEq)]
pub enum SinkPolicy {
    /// Plain adds.
    Plain,
    /// Adds with a running correction kept in the named plane.
    Compensated {
        /// The plane that holds the corrections.
        plane: String,
        /// The slot the kind's declared output is returned in, when it names one.
        output_slot: Option<String>,
        /// The names of the kind's declared outputs.
        output_names: Vec<String>,
    },
}

/// The kind of unit: how it guards, indexes and commits.
#[derive(Debug, Clone, PartialEq)]
pub struct Kind {
    /// The masks that skip a lane.
    pub guard: Guard,
    /// Whether the unit runs over an index set of lanes.
    pub lanes: bool,
    /// How accumulating outputs are committed.
    pub sink: SinkPolicy,
}

/// The size of one word of an active-set map.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MapBits {
    /// 32-bit words.
    B32,
    /// 64-bit words.
    B64,
}

/// The active set: the launch runs only the samples the map lists.
#[derive(Debug, Clone, PartialEq)]
pub struct Active {
    /// The binding of the map plane; none for the prefix form, where lane `t` runs lane `t`.
    pub map_binding: Option<String>,
    /// The binding of the plane that holds the number of active samples.
    pub count_binding: String,
    /// The lanes per compaction group.
    pub group: i64,
    /// The word size of the map.
    pub map_bits: MapBits,
    /// Whether the active lanes form a lane set, which lets integers narrow to 32 bits.
    pub lane_set: bool,
}

/// A run segment: the launch covers one run of a segmented input.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// The binding of the plane that holds the run offsets.
    pub offsets_binding: String,
    /// The index of the run.
    pub j: i64,
}

/// A thread box: each thread serves several samples.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadBox {
    /// The sample axis the box runs along, when the unit states one.
    pub axis: Option<i64>,
    /// The samples per thread.
    pub t: i64,
    /// The distance between a thread's samples.
    pub stride: i64,
}

/// The single-step variant of a unit written as two paths: the entry picks it when only one step
/// is asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct OneStep {
    /// The node handles the single step writes.
    pub sinks: Vec<u32>,
    /// The single step's own arguments.
    pub walk: Option<Walk>,
}

/// The cooperating threads of a unit.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamShape {
    /// The threads of one team.
    pub threads: i64,
    /// The warps of one team.
    pub warps: i64,
}

/// How a team stage splits its work across the launch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Grid {
    /// One team per lane.
    Lanes,
    /// One team per key.
    Keys,
    /// One team per chunk of samples.
    Chunks,
    /// One team per target.
    Targets,
}

/// The operation one step of a team stage performs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StepOp {
    /// LU factorisation.
    Lu,
    /// Solve with an LU factorisation.
    LuSolve,
    /// Transposed solve with an LU factorisation.
    LuSolveT,
    /// Inverse from an LU factorisation.
    LuInverse,
    /// Determinant from an LU factorisation.
    LuDet,
    /// Cholesky factorisation.
    Chol,
    /// Cholesky factorisation from a prepared matrix.
    CholFrom,
    /// Solve with a Cholesky factor.
    ChoSolve,
    /// Solve with a lower triangle.
    TriLower,
    /// Transposed solve with an upper triangle.
    TriUpperT,
    /// Symmetric eigen decomposition.
    Eigh,
    /// Symmetric eigen decomposition across a warp.
    EighWarp,
    /// QR factorisation.
    Qr,
    /// Least-squares solve by QR.
    QrSolve,
    /// Gram-matrix solve by QR.
    QrGramSolve,
    /// Sum of a block of rows.
    BlockSum,
    /// Partial sums of a block of rows.
    BlockPart,
    /// Sum in lane order.
    OrderedFold,
    /// Zero a scatter target.
    ScatterZero,
    /// Scatter into chunks.
    ScatterChunks,
    /// Scatter into cells.
    ScatterCells,
}

/// One lane axis of a keyed sum whose keys do not own contiguous lane ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyAxis {
    /// The axis's extent; none for the run-time first axis.
    pub extent: Option<i64>,
    /// The flat-lane distance of one step along the axis.
    pub stride: i64,
    /// Whether the sum keeps the axis (its key moves along it); else the sum runs over it.
    pub kept: bool,
}

/// One step of a team stage.
#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// The operation.
    pub op: StepOp,
    /// The matrix order or element count.
    pub n: i64,
    /// The right-hand sides.
    pub k: i64,
    /// The rows, for the operations that take a rectangle.
    pub m: i64,
    /// The most chunks a scatter may use.
    pub max_chunks: i64,
    /// The scratch planes the step uses, as (role in the step, plane name).
    pub planes: Vec<(String, String)>,
    /// The lane axes of a keyed sum whose keys' lanes are not contiguous, all of the launch's axes from the slowest; none when
    /// every key owns the lanes `key * rows ..`.
    pub key_axes: Option<Vec<KeyAxis>>,
}

/// A unit that is one stage run by a team of threads instead of a loop over samples.
#[derive(Debug, Clone, PartialEq)]
pub struct TeamStage {
    /// How the launch is split.
    pub grid: Grid,
    /// The scratch planes the stage owns, as (name, values per lane).
    pub planes: Vec<(String, i64)>,
    /// The node handles of the inputs the stage reads in place.
    pub reads: Vec<u32>,
    /// The steps, in order.
    pub steps: Vec<Step>,
    /// The map the stage reads itself; a stage over keys, chunks or targets still reads it though
    /// the launch takes none.
    pub active: Option<Active>,
}

/// One lane of a fused group.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedLane {
    /// The lane's name, unique within the group.
    pub name: String,
    /// The node handles the lane writes.
    pub sinks: Vec<u32>,
}

/// What the unit's body is.
#[derive(Debug, Clone, PartialEq)]
pub enum Form {
    /// A loop over samples written from the unit's own nodes.
    Plain,
    /// One stage run by a team.
    TeamStage(TeamStage),
    /// Several lanes written one after the other in one body.
    Fused(Vec<FusedLane>),
    /// A warp stage: its body is always supplied by the caller.
    WarpStage,
}

/// The build's numbers for the shape of the written body.
#[derive(Debug, Clone, PartialEq)]
pub struct FormSettings {
    /// The widest leaf value that is computed where it is used instead of up front.
    pub lazy_leaf_width: i64,
    /// The fewest terms for which a fold is written as a loop.
    pub rolled_fold_min: i64,
    /// The widest vector that is kept lean, when the build states it.
    pub lean_width: Option<i64>,
    /// The width of a run of lanes that agree, when the build states it.
    pub uniform_run: Option<i64>,
}

/// How a fold is written.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum FoldForm {
    /// Chosen per fold.
    #[default]
    Auto,
    /// Unrolled.
    Static,
    /// A loop.
    Rolled,
}

/// The build's numbers for runs of lanes.
#[derive(Debug, Clone, PartialEq)]
pub struct LaneSettings {
    /// The most lanes in one run.
    pub run_max: Option<i64>,
    /// The lanes in one block of a run.
    pub block_run: Option<i64>,
    /// The smallest and largest tile of a host lane split, when the build states both.
    pub tile: Option<(i64, i64)>,
}

/// The device's folding limits: how many rows one warp folds in one batched sum.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldLimits {
    /// The most rows folded together.
    pub k_max: i64,
    /// The registers a fold uses before any column.
    pub regs_base: i64,
    /// The registers each further column uses.
    pub regs_column: i64,
    /// The registers per thread above which residency drops.
    pub residency_regs: i64,
}

/// The device's rule for holding wide vectors sliced.
#[derive(Debug, Clone, PartialEq)]
pub struct WideCard {
    /// The scalar names (`float32`, `float64`) the card has a row of its own for.
    pub row_scalars: Vec<String>,
    /// The narrowest vector held sliced, by default.
    pub slice_min: Option<i64>,
    /// Whether the build forces a wide setting.
    pub forced: bool,
}

/// The build's settings and the numbers of the target it writes for.
#[derive(Debug, Clone, PartialEq)]
pub struct Build {
    /// Numbers for the shape of the body.
    pub form: FormSettings,
    /// How folds are written.
    pub fold_form: FoldForm,
    /// Numbers for runs of lanes.
    pub lanes: LaneSettings,
    /// The largest matrix order the target writes inline.
    pub linalg_max_dim: Option<i64>,
    /// The samples of one row of cells; zero when the build states none.
    pub cells_row: i64,
    /// Whether selects are written as blends.
    pub blend: bool,
    /// The warps per team the build asks for.
    pub warps: i64,
    /// The launch bounds, as (threads, blocks per multiprocessor), when the build names them.
    pub launch_bounds: Option<(i64, i64)>,
    /// The rows per warp fold the build forces, when it forces one.
    pub fold_k: Option<i64>,
    /// The device's folding limits; none for a host build.
    pub fold_limits: Option<FoldLimits>,
    /// The device's wide-vector rule; none for a host build.
    pub wide_card: Option<WideCard>,
}

/// The instruction set of the machine the host code is written for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Isa {
    /// 64-bit ARM.
    Arm64,
    /// 64-bit x86.
    X86_64,
}

/// The cache sizes of the machine, used to size tiles.
#[derive(Debug, Clone, PartialEq)]
pub struct Caches {
    /// The first-level data cache, in bytes.
    pub l1d_bytes: i64,
    /// The second-level cache, in bytes.
    pub l2_bytes: i64,
    /// Where the sizes came from, for the comment the emitted code carries.
    pub source: String,
}

/// What one add written as selects costs against an indexed access, per scalar type.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectWeight {
    /// For `double` values, when known.
    pub float64: Option<f64>,
    /// For `float` values, when known.
    pub float32: Option<f64>,
}

/// What the writers need to know about the machine.
#[derive(Debug, Clone, PartialEq)]
pub struct Machine {
    /// The instruction set; none when the build states none, as a device build does.
    pub isa: Option<Isa>,
    /// The vector instruction set of a host (`neon`, `sse2`, `avx2`, `avx512`); none for a device build and where the build states
    /// none.
    pub vector_isa: Option<String>,
    /// Whether the math library gives correctly rounded results for the vector spellings.
    pub exact_libm: bool,
    /// Whether powers go through the routed spelling the arm64 compiler needs.
    pub routed_pow: bool,
    /// The cache sizes, when the machine reports both.
    pub caches: Option<Caches>,
    /// The cost of a select against an indexed access.
    pub select_weight: SelectWeight,
}

/// The export block of the unit.
#[derive(Debug, Clone, PartialEq)]
pub enum Exports {
    /// Layout sizes derived from the build.
    Derived,
    /// These five layout sizes in place of the derived ones.
    Sizes([u64; 5]),
    /// A finished block of C++ text, written as given.
    Text(String),
}

/// Where the unit's body comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum BodySource {
    /// The writers write the body from the unit's walk.
    Walk,
    /// The caller's own body text, used in place of writing one. Only a replay of a recorded
    /// body needs it: it goes when the library lowers kernels itself.
    Given(GivenBody),
}

/// A body the caller wrote for the unit, which the writers use in place of writing one.
#[derive(Debug, Clone, PartialEq)]
pub struct GivenBody {
    /// The body text.
    pub text: String,
    /// Whether the body draws random numbers.
    pub draws_random: bool,
}

/// One render request: everything the writers read about a unit apart from its nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Unit {
    /// The unit's name, which is also its entry function.
    pub name: String,
    /// Whether the unit is a host function or a device kernel.
    pub target: Target,
    /// The scalar type.
    pub scalar: Scalar,
    /// The kind of unit.
    pub kind: Kind,
    /// What the body is.
    pub form: Form,
    /// The node handles the unit writes.
    pub sinks: Vec<u32>,
    /// The unit's arguments; none for a stage that has no walk of its own.
    pub walk: Option<Walk>,
    /// Where the body comes from.
    pub body: BodySource,
    /// The active set, when the launch runs only listed samples.
    pub active: Option<Active>,
    /// The run segment, when the launch covers one run.
    pub segment: Option<Segment>,
    /// The thread box, when each thread serves several samples.
    pub thread_box: Option<ThreadBox>,
    /// The single-step variant, when the unit is written as two paths.
    pub one_step: Option<OneStep>,
    /// The samples written per row, one unless the unit stages rows.
    pub rows: i64,
    /// The team the unit runs on, when it has one.
    pub team: Option<TeamShape>,
    /// The build's settings.
    pub build: Build,
    /// The machine facts.
    pub machine: Machine,
    /// The export block.
    pub exports: Exports,
    /// Whether the device text is followed by its table of literals.
    pub literal_table: bool,
    /// The functions the unit's calls name, each after the functions it calls; none when no call is made.
    pub functions: Option<Vec<FunctionUnit>>,
    /// The scratch planes of the unit's group that the runtime holds instead of the unit's text: each is read through the slot
    /// `spell::group::scratch_slot` names, at the slice of its group. Every other scratch plane is an array of the text.
    pub planed_scratch: Vec<String>,
}

/// A parameter of a function the unit writes.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionParam {
    /// The parameter's name in the function's text.
    pub name: String,
    /// The element type, as the walk spells it (`f64`).
    pub dtype: String,
    /// The extents, empty for a scalar.
    pub shape: Vec<i64>,
    /// The `Param` node the body reads the value through.
    pub node: u32,
}

/// One function of the unit: written once as a definition, then called.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionUnit {
    /// The name the calls know the function by.
    pub name: String,
    /// The digest that tells functions of one name apart.
    pub digest: String,
    /// The parameters, in the order a call passes its arguments.
    pub params: Vec<FunctionParam>,
    /// The element type of the result.
    pub result: String,
    /// The extents of the result, empty for a scalar.
    pub result_shape: Vec<i64>,
    /// What the function's body is.
    pub body: FunctionBody,
}

/// The body of a function the unit writes.
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionBody {
    /// A body traced from the parameters.
    Traced(TracedBody),
    /// The author's text for the target being written: the statements of the body, which see the parameters only.
    Native(String),
}

/// A traced function body: a walk of its own, taken over one result sink that stands for the returned value.
#[derive(Debug, Clone, PartialEq)]
pub struct TracedBody {
    /// The walk handle of the body.
    pub walk: u32,
    /// The result sink the body's walk is taken over.
    pub sink: u32,
    /// The arguments of the body's walk.
    pub slots: Walk,
}

impl Role {
    /// The name of the role as the walk spells it.
    pub fn name(self) -> &'static str {
        match self {
            Role::Mutable => "mutable",
            Role::PlaneIn => "plane_in",
            Role::Lookup => "lookup",
            Role::PlaneOut => "plane_out",
            Role::Uniform => "uniform",
            Role::Terminated => "terminated",
            Role::WideIn => "wide_in",
            Role::Nsamples => "nsamples",
            Role::Out => "out",
            Role::Extent => "extent",
        }
    }
}

impl Dtype {
    /// The name of the element type as the walk spells it.
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "f64",
            Dtype::F32 => "f32",
            Dtype::I32 => "i32",
            Dtype::I64 => "i64",
            Dtype::Int32 => "int32",
            Dtype::Uint32 => "uint32",
            Dtype::Bool => "bool",
        }
    }
}

impl Access {
    /// The name of the access class as the walk spells it.
    pub fn name(self) -> &'static str {
        match self {
            Access::SampleLocal => "sample_local",
            Access::CrossSampleRead => "cross_sample_read",
            Access::CrossSampleWrite => "cross_sample_write",
        }
    }
}

impl StepOp {
    /// Every operation.
    pub const ALL: [StepOp; 21] = [
        StepOp::Lu,
        StepOp::LuSolve,
        StepOp::LuSolveT,
        StepOp::LuInverse,
        StepOp::LuDet,
        StepOp::Chol,
        StepOp::CholFrom,
        StepOp::ChoSolve,
        StepOp::TriLower,
        StepOp::TriUpperT,
        StepOp::Eigh,
        StepOp::EighWarp,
        StepOp::Qr,
        StepOp::QrSolve,
        StepOp::QrGramSolve,
        StepOp::BlockSum,
        StepOp::BlockPart,
        StepOp::OrderedFold,
        StepOp::ScatterZero,
        StepOp::ScatterChunks,
        StepOp::ScatterCells,
    ];

    /// The operation a step spells with the name, when there is one.
    pub fn named(name: &str) -> Option<StepOp> {
        StepOp::ALL.into_iter().find(|op| op.name() == name)
    }

    /// The name of the operation as a step spells it.
    pub fn name(self) -> &'static str {
        match self {
            StepOp::Lu => "lu",
            StepOp::LuSolve => "lu_solve",
            StepOp::LuSolveT => "lu_solve_t",
            StepOp::LuInverse => "lu_inverse",
            StepOp::LuDet => "lu_det",
            StepOp::Chol => "chol",
            StepOp::CholFrom => "chol_from",
            StepOp::ChoSolve => "cho_solve",
            StepOp::TriLower => "tri_lower",
            StepOp::TriUpperT => "tri_upper_t",
            StepOp::Eigh => "eigh",
            StepOp::EighWarp => "eigh_warp",
            StepOp::Qr => "qr",
            StepOp::QrSolve => "qr_solve",
            StepOp::QrGramSolve => "qr_gram_solve",
            StepOp::BlockSum => "block_sum",
            StepOp::BlockPart => "block_part",
            StepOp::OrderedFold => "ordered_fold",
            StepOp::ScatterZero => "scatter_zero",
            StepOp::ScatterChunks => "scatter_chunks",
            StepOp::ScatterCells => "scatter_cells",
        }
    }
}

impl FoldForm {
    /// The name of the fold form as the IR analysis spells it.
    pub fn name(self) -> &'static str {
        match self {
            FoldForm::Auto => "auto",
            FoldForm::Static => "static",
            FoldForm::Rolled => "rolled",
        }
    }
}
