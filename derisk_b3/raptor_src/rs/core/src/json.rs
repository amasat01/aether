//! A small JSON reader (no dependency): the manifest is read by the build script and by the binding with the same code.
#[derive(Debug, Clone)]
pub enum J { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }
impl J {
    pub fn get(&self, k: &str) -> Option<&J> { if let J::Obj(v) = self { v.iter().find(|x| x.0 == k).map(|x| &x.1) } else { None } }
    pub fn str(&self) -> Option<&str> { if let J::Str(s) = self { Some(s) } else { None } }
    pub fn arr(&self) -> &[J] { if let J::Arr(a) = self { a } else { &[] } }
    pub fn obj(&self) -> &[(String, J)] { if let J::Obj(a) = self { a } else { &[] } }
    pub fn int(&self) -> i64 { if let J::Num(n) = self { *n as i64 } else { 0 } }
    pub fn strs(&self) -> Vec<String> { self.arr().iter().filter_map(|x| x.str().map(|s| s.to_string())).collect() }
}
pub fn parse(s: &str) -> Result<J, String> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let v = val(&c, &mut i)?;
    ws(&c, &mut i);
    if i != c.len() { return Err(format!("json: trailing text at {i}")) }
    Ok(v)
}
fn ws(c: &[char], i: &mut usize) { while *i < c.len() && c[*i].is_whitespace() { *i += 1 } }
fn val(c: &[char], i: &mut usize) -> Result<J, String> {
    ws(c, i);
    match c.get(*i) {
        None => Err("json: end".into()),
        Some('{') => {
            *i += 1; let mut o = vec![];
            loop {
                ws(c, i);
                if c[*i] == '}' { *i += 1; return Ok(J::Obj(o)) }
                let k = match val(c, i)? { J::Str(s) => s, _ => return Err("json: key".into()) };
                ws(c, i); if c[*i] != ':' { return Err("json: ':'".into()) } *i += 1;
                o.push((k, val(c, i)?));
                ws(c, i); if c[*i] == ',' { *i += 1 }
            }
        }
        Some('[') => {
            *i += 1; let mut a = vec![];
            loop {
                ws(c, i);
                if c[*i] == ']' { *i += 1; return Ok(J::Arr(a)) }
                a.push(val(c, i)?);
                ws(c, i); if c[*i] == ',' { *i += 1 }
            }
        }
        Some('"') => {
            *i += 1; let mut s = String::new();
            while c[*i] != '"' {
                if c[*i] == '\\' {
                    *i += 1;
                    match c[*i] { 'n' => s.push('\n'), 't' => s.push('\t'), 'r' => s.push('\r'), '/' => s.push('/'), '\\' => s.push('\\'), '"' => s.push('"'),
                        'u' => { let h: String = c[*i + 1..*i + 5].iter().collect(); s.push(char::from_u32(u32::from_str_radix(&h, 16).map_err(|e| e.to_string())?).unwrap_or('?')); *i += 4 }
                        x => return Err(format!("json: escape {x}")) }
                } else { s.push(c[*i]) }
                *i += 1;
            }
            *i += 1; Ok(J::Str(s))
        }
        Some('t') => { *i += 4; Ok(J::Bool(true)) }
        Some('f') => { *i += 5; Ok(J::Bool(false)) }
        Some('n') => { *i += 4; Ok(J::Null) }
        Some(_) => {
            let st = *i; while *i < c.len() && (c[*i].is_ascii_digit() || "+-.eE".contains(c[*i])) { *i += 1 }
            c[st..*i].iter().collect::<String>().parse().map(J::Num).map_err(|_| "json: number".to_string())
        }
    }
}
