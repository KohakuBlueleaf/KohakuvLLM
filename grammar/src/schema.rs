//! JSON schema -> a flat node table the matcher walks.
//!
//! Supported: `object` (properties in schema order, `required`), `array` (`items`, `minItems`,
//! `maxItems`), `string` (`minLength`, `maxLength`, in characters), `integer`, `number`,
//! `boolean`, `null`, `enum` / `const` of JSON values. Anything else is an error.

use serde_json::Value;

pub const UNBOUNDED: u32 = u32::MAX;
/// Longest key-candidate set and enum set (a `u64` bitmask)
pub const MAX_ALTS: usize = 64;

#[derive(Debug, Clone)]
pub struct Prop {
    /// The key as JSON text, quotes included
    pub key: Vec<u8>,
    pub node: u32,
    pub required: bool,
}

#[derive(Debug, Clone)]
pub enum Node {
    Object { props: Vec<Prop> },
    Array { item: u32, min: u32, max: u32 },
    Str { min: u32, max: u32 },
    Num { integer: bool },
    /// Fixed JSON texts (booleans, null, enum and const values)
    Lit { alts: Vec<Vec<u8>> },
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub nodes: Vec<Node>,
    pub root: u32,
}

fn bound(v: &Value, key: &str, default: u32) -> Result<u32, String> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x
            .as_u64()
            .map(|n| n.min(UNBOUNDED as u64 - 1) as u32)
            .ok_or_else(|| format!("{key} must be a non-negative integer")),
    }
}

fn literals(values: &[Value]) -> Result<Node, String> {
    if values.is_empty() || values.len() > MAX_ALTS {
        return Err(format!("enum needs 1..={MAX_ALTS} values"));
    }
    let alts = values
        .iter()
        .map(|x| serde_json::to_vec(x).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Node::Lit { alts })
}

impl Schema {
    pub fn compile(text: &str) -> Result<Schema, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let mut s = Schema {
            nodes: Vec::new(),
            root: 0,
        };
        s.root = s.add(&v)?;
        Ok(s)
    }

    fn add(&mut self, v: &Value) -> Result<u32, String> {
        let node = if let Some(values) = v.get("enum").and_then(Value::as_array) {
            literals(values)?
        } else if let Some(c) = v.get("const") {
            literals(std::slice::from_ref(c))?
        } else {
            match v.get("type").and_then(Value::as_str) {
                Some("object") => {
                    let required: Vec<&str> = v
                        .get("required")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
                    let mut props = Vec::new();
                    if let Some(map) = v.get("properties").and_then(Value::as_object) {
                        if map.len() > MAX_ALTS {
                            return Err(format!("objects take at most {MAX_ALTS} properties"));
                        }
                        for (name, sub) in map {
                            let node = self.add(sub)?;
                            props.push(Prop {
                                key: serde_json::to_vec(name).unwrap(),
                                node,
                                required: required.contains(&name.as_str()),
                            });
                        }
                    }
                    Node::Object { props }
                }
                Some("array") => {
                    let items = v.get("items").ok_or("array without items")?;
                    let item = self.add(items)?;
                    Node::Array {
                        item,
                        min: bound(v, "minItems", 0)?,
                        max: bound(v, "maxItems", UNBOUNDED)?,
                    }
                }
                Some("string") => Node::Str {
                    min: bound(v, "minLength", 0)?,
                    max: bound(v, "maxLength", UNBOUNDED)?,
                },
                Some("integer") => Node::Num { integer: true },
                Some("number") => Node::Num { integer: false },
                Some("boolean") => Node::Lit {
                    alts: vec![b"true".to_vec(), b"false".to_vec()],
                },
                Some("null") => Node::Lit {
                    alts: vec![b"null".to_vec()],
                },
                other => return Err(format!("unsupported schema type {other:?}")),
            }
        };
        self.nodes.push(node);
        Ok(self.nodes.len() as u32 - 1)
    }
}
