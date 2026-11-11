//! A Python value as the manifest validator sees it (key order, int / float / bool kept apart; a tuple or any other type is its own kind).

use crate::keytext::{float_repr as repr_float, py_str as repr_str};

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Tup(Vec<Json>),
    /// any other Python type: its type name and its repr
    Other(String, String),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Obj(v) => v.iter().find(|(a, _)| a == k).map(|(_, b)| b),
            _ => None,
        }
    }
    pub fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }
    pub fn keys(&self) -> Vec<&str> {
        match self {
            Json::Obj(v) => v.iter().map(|(a, _)| a.as_str()).collect(),
            _ => vec![],
        }
    }
    pub fn type_name(&self) -> String {
        match self {
            Json::Null => "NoneType".into(),
            Json::Bool(_) => "bool".into(),
            Json::Int(_) => "int".into(),
            Json::Float(_) => "float".into(),
            Json::Str(_) => "str".into(),
            Json::Arr(_) => "list".into(),
            Json::Tup(_) => "tuple".into(),
            Json::Other(t, _) => t.clone(),
            Json::Obj(_) => "dict".into(),
        }
    }
    /// Python `repr`.
    pub fn repr(&self) -> String {
        match self {
            Json::Null => "None".into(),
            Json::Bool(true) => "True".into(),
            Json::Bool(false) => "False".into(),
            Json::Int(i) => i.to_string(),
            Json::Float(f) => repr_float(*f),
            Json::Str(s) => repr_str(s),
            Json::Tup(v) => if v.len() == 1 { format!("({},)", v[0].repr()) } else { format!("({})", v.iter().map(Json::repr).collect::<Vec<_>>().join(", ")) },
            Json::Other(_, r) => r.clone(),
            Json::Arr(v) => format!("[{}]", v.iter().map(Json::repr).collect::<Vec<_>>().join(", ")),
            Json::Obj(v) => format!("{{{}}}", v.iter().map(|(k, x)| format!("{}: {}", repr_str(k), x.repr())).collect::<Vec<_>>().join(", ")),
        }
    }
    /// Python `str` (what an f-string `{v}` prints).
    pub fn pystr(&self) -> String {
        match self {
            Json::Str(s) => s.clone(),
            o => o.repr(),
        }
    }
}

