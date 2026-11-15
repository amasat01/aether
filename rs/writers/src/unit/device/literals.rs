//! The literal table: an option of the device build that reads every `double` literal of the body
//! from one `__constant__`
//! table instead of spelling it as an operand immediate.

/// The name of the table in the text.
const TABLE: &str = "hawk_f64_literals";

/// The literal of a `static_cast<Real>(<literal>)` at the start of `rest`, with the length of the
/// whole cast.
fn literal_cast(rest: &str) -> Option<(&str, usize)> {
    const OPEN: &str = "static_cast<Real>(";
    let after = rest.strip_prefix(OPEN)?;
    let digits = after.strip_prefix('-').unwrap_or(after);
    if !digits.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let length = after.find(|c: char| !(c.is_ascii_digit() || ".eE+-".contains(c)))?;
    if !after[length..].starts_with(')') {
        return None;
    }
    Some((&after[..length], OPEN.len() + length + 1))
}

/// `text` with every literal cast of the part after its last include read from the table, the table
/// declared after that
/// include; the text unchanged when it has no literal. Each distinct literal takes one slot, in
/// order of appearance.
pub fn literal_table(text: &str) -> String {
    let Some(at) = text.rfind("#include") else {
        return text.to_string();
    };
    let (head, tail) = text.split_at(at + "#include".len());
    let (first_line, rest) = tail.split_once('\n').unwrap_or((tail, ""));
    let mut slots: Vec<&str> = vec![];
    let mut replaced = String::new();
    let mut position = 0;
    while position < rest.len() {
        match literal_cast(&rest[position..]) {
            Some((literal, length)) => {
                let slot = slots
                    .iter()
                    .position(|seen| *seen == literal)
                    .unwrap_or_else(|| {
                        slots.push(literal);
                        slots.len() - 1
                    });
                replaced.push_str(&format!("static_cast<Real>({TABLE}[{slot}])"));
                position += length;
            }
            None => {
                let next = rest[position..].chars().next().map_or(1, char::len_utf8);
                replaced.push_str(&rest[position..position + next]);
                position += next;
            }
        }
    }
    if slots.is_empty() {
        return text.to_string();
    }
    format!(
        "{head}{first_line}\nstatic __constant__ double {TABLE}[{}] = {{{}}};\n{replaced}",
        slots.len(),
        slots.join(", ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_move_into_the_table_once_each() {
        let text = "#include <a>\nx = static_cast<Real>(0.5) + static_cast<Real>(-2e-3) \
            + static_cast<Real>(0.5);";
        let want = [
            "#include <a>",
            "static __constant__ double hawk_f64_literals[2] = {0.5, -2e-3};",
            "x = static_cast<Real>(hawk_f64_literals[0]) + static_cast<Real>(hawk_f64_literals[1])",
            " + static_cast<Real>(hawk_f64_literals[0]);",
        ];
        let want = want[..3].join("\n") + want[3];
        assert_eq!(literal_table(text), want);
    }

    #[test]
    fn a_text_without_literals_is_unchanged() {
        assert_eq!(
            literal_table("#include <a>\nint x;"),
            "#include <a>\nint x;"
        );
    }
}
