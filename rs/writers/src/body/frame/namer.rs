//! The namer: every name the writers generate comes from here. Two counters are shared by
//! several families and never reset (`hoists`: hz, la, it, lf, sv; `temps`: tb and the loop-body
//! dispatch tag `b`); the other families are numbered by a position the caller gives.

/// What a generated name stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NameKind {
    /// `t<position>`: a value named at the top level.
    Temp,
    /// `tb<n>`: a value named inside a loop body or a switch case.
    BodyTemp,
    /// `hz<n>`: a text bound ahead of the current line.
    Hoist,
    /// `hawk_la<n>`: an unset item a library call fills.
    FreshItem,
    /// `hawk_it<n>`: a held copy of an operand item.
    HeldItem,
    /// `hawk_lf<n>`: a factorisation struct.
    Factor,
    /// `hawk_sv<n>`: a store run's copy or token.
    StoreCopy,
    /// `hawk_v<pos>_<j>_<name>`: a carried local.
    Carry,
    /// `hawk_n<pos>_<j>`: the next value of a carry.
    Next,
    /// `hawk_e<pos>_<j>`: the value held at a break.
    ExitValue,
    /// `hawk_x<pos>`: the break flag.
    ExitFlag,
    /// `hawk_r<pos>_<j>`: the speculated next value of a carry.
    Result,
    /// `hawk_w<pos>_<j>`: the working copy of a carry in a skipped trip.
    WriteBack,
    /// `hawk_tp<pos>_<slot>`: a tape array.
    Tape,
    /// `hawk_c<pos>`: the iteration counter.
    Counter,
    /// `hawk_k<pos>_<name>`: a loop index.
    Index,
    /// `hawk_disp<tag>_k`: a dispatch selector.
    Selector,
    /// `hawk_disp<tag>`: a dispatch's output.
    Switch,
    /// `<tag>_k`: the key of an indexed store.
    Key,
    /// `<binding>_i`: a row leaf read whole.
    RowLeaf,
    /// `<binding>_i`: a plane read at the sample, bound as `<plane>[i]`.
    LeafView,
    /// A local the body names by hand (a store run's value, a clamped selector, a tape row).
    Local,
}

/// A generated name and what it stands for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name {
    /// The C++ identifier.
    pub text: String,
    /// What the name stands for.
    pub kind: NameKind,
}

/// The counters behind the names; one per frame, never reset per scope.
#[derive(Default)]
pub struct Namer {
    /// How many hoisted temporaries have been named.
    hoists: usize,
    /// How many body temporaries have been named.
    temps: usize,
}

/// A name of the given text and kind.
fn name(text: String, kind: NameKind) -> Name {
    Name { text, kind }
}

impl Name {
    /// A name the namer does not number (it is made from a binding or a tag).
    pub fn new(text: String, kind: NameKind) -> Name {
        name(text, kind)
    }
}

impl Namer {
    /// The next number of the `hoists` counter.
    fn next_hoist(&mut self) -> usize {
        self.hoists += 1;
        self.hoists - 1
    }

    /// The next number of the `temps` counter.
    fn next_temp(&mut self) -> usize {
        self.temps += 1;
        self.temps - 1
    }

    /// `hz<n>`: a text bound ahead of the current line.
    pub fn hoist(&mut self) -> Name {
        name(format!("hz{}", self.next_hoist()), NameKind::Hoist)
    }

    /// `hawk_la<n>`: an unset item for a library call.
    pub fn fresh_item(&mut self) -> Name {
        name(format!("hawk_la{}", self.next_hoist()), NameKind::FreshItem)
    }

    /// `hawk_it<n>`: a held copy of an operand item.
    pub fn held_item(&mut self) -> Name {
        name(format!("hawk_it{}", self.next_hoist()), NameKind::HeldItem)
    }

    /// `hawk_lf<n>`: a factorisation struct.
    pub fn factor(&mut self) -> Name {
        name(format!("hawk_lf{}", self.next_hoist()), NameKind::Factor)
    }

    /// `hawk_sv<n>`: a store run's copy or token.
    pub fn store_copy(&mut self) -> Name {
        name(format!("hawk_sv{}", self.next_hoist()), NameKind::StoreCopy)
    }

    /// `tb<n>`: a value named inside a loop body or a switch case.
    pub fn body_temp(&mut self) -> Name {
        name(format!("tb{}", self.next_temp()), NameKind::BodyTemp)
    }

    /// `b<n>`: the tag of a dispatch inside a loop body, which has no position of its own.
    pub fn body_tag(&mut self) -> String {
        format!("b{}", self.next_temp())
    }

    /// `t<position>`: a value named at the top level.
    pub fn temp_at(&self, position: usize) -> Name {
        name(format!("t{position}"), NameKind::Temp)
    }

    /// `hawk_v<pos>_<j>_<ident>`: a carried local.
    pub fn carry(&self, pos: usize, j: usize, ident: &str) -> Name {
        name(format!("hawk_v{pos}_{j}_{ident}"), NameKind::Carry)
    }

    /// `hawk_n<pos>_<j>`: the next value of a carry.
    pub fn next(&self, pos: usize, j: usize) -> Name {
        name(format!("hawk_n{pos}_{j}"), NameKind::Next)
    }

    /// `hawk_e<pos>_<j>`: the value held at a break.
    pub fn exit_value(&self, pos: usize, j: usize) -> Name {
        name(format!("hawk_e{pos}_{j}"), NameKind::ExitValue)
    }

    /// `hawk_x<pos>`: the break flag.
    pub fn exit_flag(&self, pos: usize) -> Name {
        name(format!("hawk_x{pos}"), NameKind::ExitFlag)
    }

    /// `hawk_r<pos>_<j>`: the speculated next value of a carry.
    pub fn result(&self, pos: usize, j: usize) -> Name {
        name(format!("hawk_r{pos}_{j}"), NameKind::Result)
    }

    /// `hawk_w<pos>_<j>`: the working copy of a carry in a skipped trip.
    pub fn write_back(&self, pos: usize, j: usize) -> Name {
        name(format!("hawk_w{pos}_{j}"), NameKind::WriteBack)
    }

    /// `hawk_tp<pos>_<slot>`: a tape array.
    pub fn tape(&self, pos: usize, slot: usize) -> Name {
        name(format!("hawk_tp{pos}_{slot}"), NameKind::Tape)
    }

    /// `hawk_c<pos>`: the iteration counter.
    pub fn counter(&self, pos: usize) -> Name {
        name(format!("hawk_c{pos}"), NameKind::Counter)
    }

    /// `hawk_k<pos>_<ident>`: a loop index.
    pub fn index(&self, pos: usize, ident: &str) -> Name {
        name(format!("hawk_k{pos}_{ident}"), NameKind::Index)
    }

    /// `hawk_disp<tag>_k`: a dispatch selector.
    pub fn selector(&self, tag: &str) -> Name {
        name(format!("hawk_disp{tag}_k"), NameKind::Selector)
    }

    /// `hawk_disp<tag>`: a dispatch's output.
    pub fn switch(&self, tag: &str) -> Name {
        name(format!("hawk_disp{tag}"), NameKind::Switch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_counters_number_a_mixed_sequence_as_the_frame_did() {
        let mut n = Namer::default();
        let got = vec![
            n.hoist().text,
            n.body_temp().text,
            n.fresh_item().text,
            n.body_tag(),
            n.held_item().text,
            n.body_temp().text,
            n.factor().text,
            n.store_copy().text,
            n.hoist().text,
        ];
        assert_eq!(
            got,
            ["hz0", "tb0", "hawk_la1", "b1", "hawk_it2", "tb2", "hawk_lf3", "hawk_sv4", "hz5"]
        );
    }

    #[test]
    fn positional_names_do_not_move_the_counters() {
        let mut n = Namer::default();
        assert_eq!(n.carry(3, 1, "x").text, "hawk_v3_1_x");
        assert_eq!(n.temp_at(7).text, "t7");
        assert_eq!(n.hoist().text, "hz0");
        assert_eq!(n.next(3, 1).text, "hawk_n3_1");
    }
}
