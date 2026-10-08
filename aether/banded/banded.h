// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file banded.h
 * @brief Opt-in umbrella for the `aether::banded` module — the emulated
 *        53-bit carrier, its 8-byte storage codec and the numeric face
 *        over it.
 *
 * @code
 * #include <aether/banded/banded.h>
 * using aether::banded::Band;
 * using aether::banded::BandedReal;
 *
 * BandedReal a = BandedReal::fromDouble(1.0);   // host ingest, explicit
 * BandedReal b = BandedReal::fromDouble(3.0);
 * BandedReal q = a / b;                          // one pack, at the `=`
 * double back  = q.toDouble();                   // named egress, never implicit
 * @endcode
 *
 * @section why_banded What this module is for
 * The target hardware for the emulation lane has no FP64 units, or has
 * them at 1:32 of the FP32 rate. `Band` delivers 53 certified significand
 * bits out of three `float` limbs and a handful of integer ops, so a
 * kernel that needs double-class accuracy can have it on a card that
 * cannot spell `double` quickly. That is a portability warrant, not a
 * micro-optimisation: the SASS gate `tests/sass/check_band_fp64free.sh`
 * asserts zero FP64 instructions in the banded kernels and carries a
 * seeded positive control so a clean result cannot be a blind scan.
 *
 * @section types The three types, three roles
 *  - `Band` (`Band.h`) — the register carrier: `float hi, lo, tail`, 12
 *    bytes, no memory form, `certifiedBits == 53`. All chain arithmetic
 *    lives here.
 *  - `BandCell8` (`BandCell8.h`) — the codec word: one 64-bit word, depth
 *    56, `operator==` is a bit compare (right for "is this the fill
 *    value", wrong for arithmetic).
 *  - `BandedReal` (`BandedReal.h`) — the storage value type: the codec
 *    word with numeric equality and ordering, an explicit `double` egress
 *    and a `std::numeric_limits` specialization. Defined at array bias 0.
 *
 * @section diet Not included from `aether/aether.h`, deliberately
 * This umbrella reaches the whole codec and carrier. Putting it in the
 * library umbrella would hand every TU that computes `sin(double)` about
 * two thousand lines it has no use for — exactly the cost `tests/headers/
 * check_header_diet.sh` exists to police (its BANDED-* arms pin the
 * absence, and a control arm proves the grep can see the header when it is
 * included). The `aether::math` facade routing for the banded family is
 * reached instead through the declaration-only `aether/math/detail/
 * BandedFwd.h`, which `math/detail/MathDispatch.h` includes and which
 * names no definitions at all. A consumer of the type includes this
 * header explicitly.
 *
 * @section scope_banded What this module covers
 * The carrier and its certified core (`neg`/`add`/`sub`/`mul`/`recip`/
 * `div`), the four exact sign/order ops the facade routes to (`abs`/
 * `copysign`/`fmax`/`fmin`), the tier-1 and escape codecs, the rebias
 * pair, the admission predicates, the storage face, its operator set and
 * its limits; `BandAccum`/`BandAccumVec<N>` (`Accum.h`) and the `Ff1`/
 * `Ff2` interior working-carrier rungs (`Ff1.h`/`Ff2.h`) — certified
 * 24-bit and 45-bit carriers demoted from `Band` at region entry, packing
 * back through the same two-name `Accum.h` egress; the banded-domain half
 * of the demand `Ladder` (`Ladder.h` — `BandedLadder`,
 * `IsBandedStorageDomain`, `BandedDemandType`/`BandedDemandT`); the
 * reciprocal-square-root family (`RsqrtCore.h` — `rsqrtCore`/`rsqrt`/
 * `sqrt_`/`rsqrtCube`) and the atan-reduction sector lookup
 * (`AtanSector.h` — `AtanSector`/`atanSector`, table only); the Band-typed
 * half of the coefficient-table element (`BandTable.h` — `BandPair`, its
 * `band()` decode and Band-typed seam operators, the host-only ingest
 * `bandPairFromDouble`/`bandPairToDouble`/`bandPairAdmits`/
 * `bandPairCertifiedBits`/`bandPairTableFromDoubles`; `compDD()` and the
 * texture-fetch wiring are not ported — @see that file's own scope notes)
 * and the 16-byte wide-range storage cell (`BandCell.h` — `BandCell`,
 * `cellFromBand`/`bandFromCell`/`loadCell`/`storeCell`; a different,
 * wider-range sibling of the tier-1-windowed `BandCell8`); `exp`/`log`/
 * `pow` and the 16-entry `log` table (`BandExpLog.h`), `pow = exp(y*log
 * x)`; `cbrt`/`hypot` and the `rsqrt(+Inf)=+0` IEEE fix (`BandRoot.h`);
 * `trigReduce`/`sincos`/`sin`/`cos` (`BandTrig.h`), the Cody-Waite `pi/2`
 * reduction and `bandTrigAdmits`; `floor`/`ceil`/`round`/`fmod`/`trunc`
 * and the Band `fma` (`BandRound.h`; `fdim` is in `Band.h`); `atan`/
 * `atan2`/`asin`/`acos` (`BandInvTrig.h`), built on `AtanSector.h`'s
 * table. Still deferred: the hyperbolic/exp2-log2/tan constructions, the
 * expression-template/consumer integration, texture arrays and non-zero
 * array biases.
 */

#include "aether/banded/Band.h"
#include "aether/banded/BandExpLog.h"
#include "aether/banded/BandTrig.h"
#include "aether/banded/BandRound.h"
#include "aether/banded/BandCell8.h"
#include "aether/banded/BandedReal.h"
#include "aether/banded/BandedRealOps.h"
#include "aether/banded/BandedLimits.h"
#include "aether/banded/Accum.h"
#include "aether/banded/Ff2.h"
#include "aether/banded/Ff1.h"
#include "aether/banded/Ladder.h"
#include "aether/banded/RsqrtCore.h"
#include "aether/banded/BandRoot.h"
#include "aether/banded/AtanSector.h"
#include "aether/banded/BandInvTrig.h"
#include "aether/banded/BandTable.h"
#include "aether/banded/BandCell.h"
