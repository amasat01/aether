//! The vocabulary of operation kinds: one enum, the text each is spelled as, and the kind a text
//! names. It uses nothing else of the crate.
use std::fmt;

/// Declares the kinds: the enum, the text each is spelled as, and the reader of that text.
macro_rules! op_kinds {
    ($($(#[$doc:meta])* $variant:ident = $text:literal,)*) => {
        /// The kind of an operation node.
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum OpKind {
            $($(#[$doc])* $variant,)*
        }

        impl OpKind {
            /// The text the kind is spelled as in the IR and in the emitted names.
            pub fn name(self) -> &'static str {
                match self {
                    $(OpKind::$variant => $text,)*
                }
            }

            /// The kind a text names, if it names one.
            pub(crate) fn parse(text: &str) -> Option<OpKind> {
                match text {
                    $($text => Some(OpKind::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

op_kinds! {
    /// Sum of two values.
    Add = "add",
    /// Difference of two values.
    Sub = "sub",
    /// Product of two values.
    Mul = "mul",
    /// Quotient of two values.
    Div = "div",
    /// First value raised to the power of the second.
    Pow = "pow",
    /// The smaller of two values.
    Min = "min",
    /// The larger of two values.
    Max = "max",
    /// Angle of the point whose coordinates are the two values.
    Atan2 = "atan2",
    /// Length of the hypotenuse of the two values.
    Hypot = "hypot",
    /// First value with the sign of the second.
    Copysign = "copysign",
    /// Remainder of a division that rounds toward zero.
    Fmod = "fmod",
    /// Remainder of a division that rounds to nearest.
    Remainder = "remainder",
    /// Positive difference of two values, or zero.
    Fdim = "fdim",
    /// Negation.
    Neg = "neg",
    /// Absolute value.
    Abs = "abs",
    /// Square root.
    Sqrt = "sqrt",
    /// Reciprocal of the square root.
    Rsqrt = "rsqrt",
    /// Exponential.
    Exp = "exp",
    /// Natural logarithm.
    Log = "log",
    /// Sine.
    Sin = "sin",
    /// Cosine.
    Cos = "cos",
    /// Tangent.
    Tan = "tan",
    /// Hyperbolic tangent.
    Tanh = "tanh",
    /// Inverse sine.
    Asin = "asin",
    /// Inverse cosine.
    Acos = "acos",
    /// Inverse tangent.
    Atan = "atan",
    /// Largest whole number not above the value.
    Floor = "floor",
    /// Two raised to the value.
    Exp2 = "exp2",
    /// Exponential minus one, exact near zero.
    Expm1 = "expm1",
    /// Base-two logarithm.
    Log2 = "log2",
    /// Base-ten logarithm.
    Log10 = "log10",
    /// Logarithm of one plus the value, exact near zero.
    Log1p = "log1p",
    /// Cube root.
    Cbrt = "cbrt",
    /// Hyperbolic sine.
    Sinh = "sinh",
    /// Hyperbolic cosine.
    Cosh = "cosh",
    /// Inverse hyperbolic sine.
    Asinh = "asinh",
    /// Inverse hyperbolic cosine.
    Acosh = "acosh",
    /// Inverse hyperbolic tangent.
    Atanh = "atanh",
    /// Smallest whole number not below the value.
    Ceil = "ceil",
    /// The value with its fraction dropped.
    Trunc = "trunc",
    /// The value rounded to the nearest whole number, halves away from zero.
    Round = "round",
    /// The value rounded to the nearest whole number, halves to even.
    Rint = "rint",
    /// Sign of the value.
    Sign = "sign",
    /// Error function.
    Erf = "erf",
    /// Complementary error function.
    Erfc = "erfc",
    /// Distance from the value to the next representable number.
    Spacing = "spacing",
    /// First value is less than the second.
    Lt = "lt",
    /// First value is less than or equal to the second.
    Le = "le",
    /// First value is greater than the second.
    Gt = "gt",
    /// First value is greater than or equal to the second.
    Ge = "ge",
    /// The two values are equal.
    Eq = "eq",
    /// The two values differ.
    Ne = "ne",
    /// Both flags hold.
    Land = "land",
    /// At least one flag holds.
    Lor = "lor",
    /// The flag does not hold.
    Lnot = "lnot",
    /// Dot product of two vectors.
    Dot = "dot",
    /// Cross product of two vectors.
    Cross = "cross",
    /// Euclidean norm of a vector.
    Norm = "norm",
    /// Sum of the elements of a value.
    Sum = "sum",
    /// One element of a value, at a static index.
    Component = "component",
    /// One element of a value, at an index known only at run time.
    ComponentAt = "component_at",
    /// A copy of a value with one element, at a run-time index, replaced.
    SetComponentAt = "set_component_at",
    /// A copy of a value with one element, at a run-time index, increased.
    AddAt = "add_at",
    /// A vector or matrix assembled from its elements.
    Vec = "vec",
    /// One value repeated to fill a vector or matrix.
    Splat = "splat",
    /// Product of a matrix and a vector.
    Mv = "mv",
    /// Product of two matrices.
    Mm = "mm",
    /// Transpose of a matrix.
    Transpose = "transpose",
    /// Outer product of two vectors.
    Outer = "outer",
    /// LU factorisation of a matrix.
    Lu = "lu",
    /// Cholesky factorisation of a matrix.
    Chol = "chol",
    /// Eigen decomposition of a symmetric matrix.
    Eigh = "eigh",
    /// Cholesky factorisation seeded from a given triangle.
    CholFrom = "chol_from",
    /// QR factorisation of a matrix.
    Qr = "qr",
    /// Solve against an LU factor.
    LuSolve = "lu_solve",
    /// Solve against the transpose of an LU factor.
    LuSolveT = "lu_solve_t",
    /// Solve against a Cholesky factor.
    ChoSolve = "cho_solve",
    /// Solve against a lower triangle.
    TriLower = "tri_lower",
    /// Solve against the transpose of an upper triangle.
    TriUpperT = "tri_upper_t",
    /// Solve against a QR factor.
    QrSolve = "qr_solve",
    /// Solve of the normal equations of a QR factor.
    QrGramSolve = "qr_gram_solve",
    /// Status of an LU factorisation.
    LuInfo = "lu_info",
    /// Determinant from an LU factor.
    LuDet = "lu_det",
    /// Status of a Cholesky factorisation.
    CholInfo = "chol_info",
    /// The triangle of a Cholesky factor.
    CholFactor = "chol_factor",
    /// The eigenvalues of an eigen decomposition.
    EighW = "eigh_w",
    /// The eigenvectors of an eigen decomposition.
    EighV = "eigh_v",
    /// Status of an eigen decomposition.
    EighInfo = "eigh_info",
    /// Status of a QR factorisation.
    QrInfo = "qr_info",
    /// Least-squares solve.
    Lstsq = "lstsq",
    /// Least-squares solve through the normal equations.
    LstsqGramSolve = "lstsq_gram_solve",
    /// Status of a least-squares solve.
    LstsqInfo = "lstsq_info",
    /// Ray and triangle intersection.
    RayTri = "ray_tri",
    /// Ray and axis-aligned box intersection.
    RayAabb = "ray_aabb",
    /// Ray and sphere intersection.
    RaySphere = "ray_sphere",
    /// Product of two quaternions.
    QuatMul = "quat_mul",
    /// Conjugate of a quaternion.
    QuatConj = "quat_conj",
    /// Reciprocal of a quaternion.
    QuatRecip = "quat_recip",
    /// Rotation of a vector by a quaternion.
    QuatRotate = "quat_rotate",
    /// A vector viewed as a quaternion with no real part.
    AsPure = "as_pure",
    /// The vector part of a quaternion.
    AsVec3 = "as_vec3",
    /// A uniform random draw.
    RandomUniform = "random_uniform",
    /// A normal random draw.
    RandomNormal = "random_normal",
    /// A choice between two values by a condition.
    Select = "select",
    /// A choice among branches by a selector.
    Dispatch = "dispatch",
    /// An indexed read of a plane.
    At = "at",
    /// Fused multiply-add.
    Fma = "fma",
    /// The value limited to a range.
    Clip = "clip",
    /// The value is not a number.
    Isnan = "isnan",
    /// The value is infinite.
    Isinf = "isinf",
    /// The value is neither infinite nor not a number.
    Isfinite = "isfinite",
    /// Integer division rounding down.
    Floordiv = "floordiv",
    /// Integer remainder with the sign of the divisor.
    Mod = "mod",
    /// Bitwise and.
    Band = "band",
    /// Bitwise or.
    Bor = "bor",
    /// Bitwise exclusive or.
    Bxor = "bxor",
    /// Shift left.
    Shl = "shl",
    /// Shift right.
    Shr = "shr",
    /// An index limited to the extent it reads.
    ClampIndex = "clamp_index",
    /// Bitwise complement.
    Invert = "invert",
    /// Conversion to another element type.
    Cast = "cast",
    /// A value placed in its plane.
    Resident = "resident",
    /// A block of the target language spliced in as written.
    RawDevice = "raw_device",
    /// Sum of one value over the lanes of a warp.
    WarpSum = "warp_sum",
}

impl fmt::Display for OpKind {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(self.name())
    }
}
