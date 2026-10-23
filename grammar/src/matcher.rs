//! The byte-level JSON matcher: a fixed-size stack of frames over a compiled schema.
//!
//! Counters are stored as what is still needed (`need`: below a minimum) and what is still
//! allowed (`room`: below a maximum), so `State::key` can clamp both to the longest token and
//! get a state that accepts exactly the same next `l` bytes: the memo key for masks.

use std::hash::{Hash, Hasher};

use crate::schema::{Node, Schema, UNBOUNDED};

pub const MAX_DEPTH: usize = 12;
pub const MAX_DIGITS: u32 = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjPh {
    Open,
    First,
    Comma,
    Key,
    AfterKey,
    AfterColon,
    AfterValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArrPh {
    Open,
    First,
    Comma,
    AfterItem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NumPh {
    Start,
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl NumPh {
    fn terminal(self) -> bool {
        matches!(self, NumPh::Zero | NumPh::Int | NumPh::Frac | NumPh::Exp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Frame {
    Empty,
    /// `next`: the first property not yet written; `alts`/`pos`: the key candidates being read
    Obj { node: u32, next: u8, ph: ObjPh, alts: u64, pos: u16 },
    Arr { node: u32, need: u32, room: u32, ph: ArrPh },
    /// `esc`: 1 after a backslash, 2..=5 inside `\uXXXX`; `utf8`: continuation bytes pending
    Str { open: bool, need: u32, room: u32, esc: u8, utf8: u8 },
    Num { integer: bool, ph: NumPh, room: u32 },
    Lit { node: u32, alts: u64, pos: u16 },
}

#[derive(Clone, Copy, Debug)]
pub struct State {
    stack: [Frame; MAX_DEPTH],
    depth: u8,
    ws: u8,
}

impl PartialEq for State {
    fn eq(&self, o: &Self) -> bool {
        self.depth == o.depth && self.ws == o.ws && self.frames() == o.frames()
    }
}

impl Eq for State {}

impl Hash for State {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.depth.hash(h);
        self.ws.hash(h);
        self.frames().hash(h);
    }
}

enum Out {
    Consumed,
    Reject,
    /// The frame is complete without this byte: pop it and give the byte to the parent
    PopRefeed,
    /// Consumed and complete
    PopConsumed,
    /// Open a value of this node and give it the byte
    Push(u32),
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\t' | b'\r')
}

fn dec(x: u32) -> u32 {
    if x == UNBOUNDED {
        x
    } else {
        x.saturating_sub(1)
    }
}

/// Literal alternatives of `alts` that byte `b` extends at `pos`
fn narrow(texts: &[Vec<u8>], alts: u64, pos: usize, b: u8) -> u64 {
    let mut out = 0;
    for (i, t) in texts.iter().enumerate() {
        if alts >> i & 1 == 1 && t.get(pos) == Some(&b) {
            out |= 1 << i;
        }
    }
    out
}

fn complete(texts: &[Vec<u8>], alts: u64, pos: usize) -> (bool, bool) {
    let (mut done, mut longer) = (false, false);
    for (i, t) in texts.iter().enumerate() {
        if alts >> i & 1 == 1 {
            if t.len() == pos {
                done = true;
            } else {
                longer = true;
            }
        }
    }
    (done, longer)
}

impl State {
    pub fn new(schema: &Schema) -> State {
        let mut s = State {
            stack: [Frame::Empty; MAX_DEPTH],
            depth: 0,
            ws: 0,
        };
        s.push(schema, schema.root);
        s
    }

    /// A bare string body with `room` characters left (classifying tokens)
    pub fn string_body(room: u32) -> State {
        let mut s = State {
            stack: [Frame::Empty; MAX_DEPTH],
            depth: 1,
            ws: 0,
        };
        s.stack[0] = Frame::Str {
            open: true,
            need: 0,
            room,
            esc: 0,
            utf8: 0,
        };
        s
    }

    fn frames(&self) -> &[Frame] {
        &self.stack[..self.depth as usize]
    }

    fn push(&mut self, schema: &Schema, node: u32) -> bool {
        if self.depth as usize == MAX_DEPTH {
            return false;
        }
        self.stack[self.depth as usize] = match &schema.nodes[node as usize] {
            Node::Object { .. } => Frame::Obj {
                node,
                next: 0,
                ph: ObjPh::Open,
                alts: 0,
                pos: 0,
            },
            Node::Array { min, max, .. } => Frame::Arr {
                node,
                need: *min,
                room: *max,
                ph: ArrPh::Open,
            },
            Node::Str { min, max } => Frame::Str {
                open: false,
                need: *min,
                room: *max,
                esc: 0,
                utf8: 0,
            },
            Node::Num { integer } => Frame::Num {
                integer: *integer,
                ph: NumPh::Start,
                room: MAX_DIGITS,
            },
            Node::Lit { alts } => Frame::Lit {
                node,
                alts: u64::MAX >> (64 - alts.len()),
                pos: 0,
            },
        };
        self.depth += 1;
        true
    }

    pub fn done(&self) -> bool {
        self.depth == 0
    }

    /// Inside a string body at a character boundary (no escape, no partial character): the
    /// characters still allowed
    pub fn body_room(&self) -> Option<u32> {
        match self.depth {
            0 => None,
            d => match self.stack[d as usize - 1] {
                Frame::Str {
                    open: true,
                    room,
                    esc: 0,
                    utf8: 0,
                    ..
                } => Some(room),
                _ => None,
            },
        }
    }

    /// The reply may end here (end-of-sequence allowed)
    pub fn accepting(&self, schema: &Schema) -> bool {
        match self.depth {
            0 => true,
            1 => match self.stack[0] {
                Frame::Num { ph, .. } => ph.terminal(),
                Frame::Lit { node, alts, pos } => match &schema.nodes[node as usize] {
                    Node::Lit { alts: texts } => complete(texts, alts, pos as usize).0,
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }

    /// The memo key: counters clamped to `l`, the longest token in bytes
    pub fn key(&self, l: u32) -> State {
        let mut k = *self;
        for f in k.stack[..k.depth as usize].iter_mut() {
            match f {
                Frame::Arr { need, room, .. } | Frame::Str { need, room, .. } => {
                    *need = (*need).min(l);
                    *room = (*room).min(l);
                }
                Frame::Num { room, .. } => *room = (*room).min(l),
                _ => {}
            }
        }
        k
    }

    /// Feed one byte; false (state unspecified) if the schema rejects it
    pub fn step(&mut self, schema: &Schema, max_ws: u8, b: u8) -> bool {
        loop {
            if self.depth == 0 {
                return false;
            }
            let top = self.depth as usize - 1;
            let ws_ok = self.ws < max_ws;
            let in_string = matches!(self.stack[top], Frame::Str { open: true, .. });
            let out = Self::frame_step(&mut self.stack[top], schema, ws_ok, b);
            match out {
                Out::Consumed | Out::PopConsumed => {
                    self.ws = if is_ws(b) && !in_string { self.ws + 1 } else { 0 };
                    if matches!(out, Out::PopConsumed) {
                        self.depth -= 1;
                    }
                    return true;
                }
                Out::Reject => return false,
                Out::PopRefeed => self.depth -= 1,
                Out::Push(node) => {
                    if !self.push(schema, node) {
                        return false;
                    }
                }
            }
        }
    }

    fn frame_step(f: &mut Frame, schema: &Schema, ws_ok: bool, b: u8) -> Out {
        match f {
            Frame::Empty => Out::Reject,
            Frame::Obj {
                node,
                next,
                ph,
                alts,
                pos,
            } => {
                let Node::Object { props } = &schema.nodes[*node as usize] else {
                    return Out::Reject;
                };
                let rest = &props[*next as usize..];
                let required_left = rest.iter().any(|p| p.required);
                let gap = matches!(
                    ph,
                    ObjPh::First | ObjPh::Comma | ObjPh::AfterKey | ObjPh::AfterColon | ObjPh::AfterValue
                );
                if gap && is_ws(b) {
                    return if ws_ok { Out::Consumed } else { Out::Reject };
                }
                match ph {
                    ObjPh::Open => {
                        if b != b'{' {
                            return Out::Reject;
                        }
                        *ph = ObjPh::First;
                        Out::Consumed
                    }
                    ObjPh::First | ObjPh::Comma => {
                        if b == b'}' && *ph == ObjPh::First && !required_left {
                            return Out::PopConsumed;
                        }
                        if b != b'"' || rest.is_empty() {
                            return Out::Reject;
                        }
                        // Candidates: every property up to and including the next required one
                        let mut m = 0u64;
                        for (i, p) in rest.iter().enumerate() {
                            m |= 1 << (*next as usize + i);
                            if p.required {
                                break;
                            }
                        }
                        *alts = m;
                        *pos = 1;
                        *ph = ObjPh::Key;
                        Out::Consumed
                    }
                    ObjPh::Key => {
                        let mut m = 0u64;
                        let mut done = None;
                        for (i, p) in props.iter().enumerate() {
                            if *alts >> i & 1 == 1 && p.key.get(*pos as usize) == Some(&b) {
                                m |= 1 << i;
                                if p.key.len() == *pos as usize + 1 {
                                    done = Some(i);
                                }
                            }
                        }
                        if m == 0 {
                            return Out::Reject;
                        }
                        if let Some(i) = done {
                            *next = i as u8 + 1;
                            *ph = ObjPh::AfterKey;
                        } else {
                            *alts = m;
                            *pos += 1;
                        }
                        Out::Consumed
                    }
                    ObjPh::AfterKey => {
                        if b != b':' {
                            return Out::Reject;
                        }
                        *ph = ObjPh::AfterColon;
                        Out::Consumed
                    }
                    ObjPh::AfterColon => {
                        *ph = ObjPh::AfterValue;
                        Out::Push(props[*next as usize - 1].node)
                    }
                    ObjPh::AfterValue => {
                        if b == b',' && !rest.is_empty() {
                            *ph = ObjPh::Comma;
                            Out::Consumed
                        } else if b == b'}' && !required_left {
                            Out::PopConsumed
                        } else {
                            Out::Reject
                        }
                    }
                }
            }
            Frame::Arr {
                node,
                need,
                room,
                ph,
            } => {
                let Node::Array { item, .. } = &schema.nodes[*node as usize] else {
                    return Out::Reject;
                };
                if *ph != ArrPh::Open && is_ws(b) {
                    return if ws_ok { Out::Consumed } else { Out::Reject };
                }
                match ph {
                    ArrPh::Open => {
                        if b != b'[' {
                            return Out::Reject;
                        }
                        *ph = ArrPh::First;
                        Out::Consumed
                    }
                    ArrPh::First | ArrPh::Comma => {
                        if b == b']' && *ph == ArrPh::First && *need == 0 {
                            return Out::PopConsumed;
                        }
                        if *room == 0 {
                            return Out::Reject;
                        }
                        *need = need.saturating_sub(1);
                        *room = dec(*room);
                        *ph = ArrPh::AfterItem;
                        Out::Push(*item)
                    }
                    ArrPh::AfterItem => {
                        if b == b',' && *room > 0 {
                            *ph = ArrPh::Comma;
                            Out::Consumed
                        } else if b == b']' && *need == 0 {
                            Out::PopConsumed
                        } else {
                            Out::Reject
                        }
                    }
                }
            }
            Frame::Str {
                open,
                need,
                room,
                esc,
                utf8,
            } => {
                if !*open {
                    if b != b'"' {
                        return Out::Reject;
                    }
                    *open = true;
                    return Out::Consumed;
                }
                if *utf8 > 0 {
                    if b & 0xC0 != 0x80 {
                        return Out::Reject;
                    }
                    *utf8 -= 1;
                    return Out::Consumed;
                }
                match *esc {
                    1 => {
                        *esc = match b {
                            b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => 0,
                            b'u' => 2,
                            _ => return Out::Reject,
                        };
                        return Out::Consumed;
                    }
                    2..=5 => {
                        if !b.is_ascii_hexdigit() {
                            return Out::Reject;
                        }
                        *esc = if *esc == 5 { 0 } else { *esc + 1 };
                        return Out::Consumed;
                    }
                    _ => {}
                }
                if b == b'"' {
                    return if *need == 0 {
                        Out::PopConsumed
                    } else {
                        Out::Reject
                    };
                }
                let pending = match b {
                    0x00..=0x1F => return Out::Reject,
                    b'\\' => {
                        *esc = 1;
                        0
                    }
                    0x20..=0x7F => 0,
                    0xC2..=0xDF => 1,
                    0xE0..=0xEF => 2,
                    0xF0..=0xF4 => 3,
                    _ => return Out::Reject,
                };
                if *room == 0 {
                    return Out::Reject;
                }
                *room = dec(*room);
                *need = need.saturating_sub(1);
                *utf8 = pending;
                Out::Consumed
            }
            Frame::Num { integer, ph, room } => {
                let digit = b.is_ascii_digit();
                let next = match (*ph, b) {
                    (NumPh::Start, b'-') => Some(NumPh::Minus),
                    (NumPh::Start | NumPh::Minus, b'0') => Some(NumPh::Zero),
                    (NumPh::Start | NumPh::Minus, _) if digit => Some(NumPh::Int),
                    (NumPh::Int, _) if digit => Some(NumPh::Int),
                    (NumPh::Zero | NumPh::Int, b'.') if !*integer => Some(NumPh::Dot),
                    (NumPh::Dot | NumPh::Frac, _) if digit => Some(NumPh::Frac),
                    (NumPh::Zero | NumPh::Int | NumPh::Frac, b'e' | b'E') if !*integer => {
                        Some(NumPh::E)
                    }
                    (NumPh::E, b'+' | b'-') => Some(NumPh::ESign),
                    (NumPh::E | NumPh::ESign | NumPh::Exp, _) if digit => Some(NumPh::Exp),
                    _ => None,
                };
                match next {
                    Some(n) => {
                        if digit {
                            if *room == 0 {
                                return Out::Reject;
                            }
                            *room -= 1;
                        }
                        *ph = n;
                        Out::Consumed
                    }
                    None if ph.terminal() => Out::PopRefeed,
                    None => Out::Reject,
                }
            }
            Frame::Lit { node, alts, pos } => {
                let Node::Lit { alts: texts } = &schema.nodes[*node as usize] else {
                    return Out::Reject;
                };
                let m = narrow(texts, *alts, *pos as usize, b);
                if m == 0 {
                    return if complete(texts, *alts, *pos as usize).0 {
                        Out::PopRefeed
                    } else {
                        Out::Reject
                    };
                }
                *alts = m;
                *pos += 1;
                match complete(texts, m, *pos as usize) {
                    (true, false) => Out::PopConsumed,
                    _ => Out::Consumed,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(schema: &str, text: &str, max_ws: u8) -> bool {
        let s = Schema::compile(schema).unwrap();
        let mut st = State::new(&s);
        text.bytes().all(|b| st.step(&s, max_ws, b)) && st.accepting(&s)
    }

    const ENTITY: &str = r#"{"type":"object","properties":{"present":{"type":"boolean"},
        "caption":{"type":"string","maxLength":5},"features":{"type":"array","maxItems":2,
        "items":{"type":"object","properties":{"name":{"type":"string","minLength":1},
        "multiplicity":{"enum":["single","multiple"]}},"required":["name","multiplicity"]}}},
        "required":["present","caption","features"]}"#;

    #[test]
    fn objects_arrays_strings() {
        let ok = r#"{"present":true,"caption":"ab\"c","features":[{"name":"x","multiplicity":"single"}]}"#;
        assert!(accepts(ENTITY, ok, 0));
        assert!(accepts(ENTITY, r#"{"present":false,"caption":"","features":[]}"#, 0));
        // maxLength in characters: 6 > 5
        assert!(!accepts(ENTITY, r#"{"present":true,"caption":"abcdef","features":[]}"#, 0));
        // a 2-byte character counts once
        assert!(accepts(ENTITY, r#"{"present":true,"caption":"abcdé","features":[]}"#, 0));
        // minLength 1
        let empty = r#"{"present":true,"caption":"","features":[{"name":"","multiplicity":"single"}]}"#;
        assert!(!accepts(ENTITY, empty, 0));
        // maxItems 2
        let three = r#"{"present":true,"caption":"","features":[{"name":"a","multiplicity":"single"},{"name":"a","multiplicity":"single"},{"name":"a","multiplicity":"single"}]}"#;
        assert!(!accepts(ENTITY, three, 0));
        // required order, missing key, enum
        assert!(!accepts(ENTITY, r#"{"caption":"","present":true,"features":[]}"#, 0));
        assert!(!accepts(ENTITY, r#"{"present":true,"features":[]}"#, 0));
        let bad_enum = r#"{"present":true,"caption":"","features":[{"name":"a","multiplicity":"some"}]}"#;
        assert!(!accepts(ENTITY, bad_enum, 0));
        // incomplete is not accepting
        assert!(!accepts(ENTITY, r#"{"present":true"#, 0));
    }

    #[test]
    fn whitespace_knob() {
        let spaced = r#"{ "present": true, "caption": "", "features": [ ] }"#;
        assert!(!accepts(ENTITY, spaced, 0));
        assert!(accepts(ENTITY, spaced, 1));
        assert!(!accepts(ENTITY, r#"{  "present":true,"caption":"","features":[]}"#, 1));
    }

    #[test]
    fn numbers_and_optional() {
        let s = r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"number"},
            "c":{"type":"null"}},"required":["b"]}"#;
        assert!(accepts(s, r#"{"a":-12,"b":3.5e-2}"#, 0));
        assert!(accepts(s, r#"{"b":0,"c":null}"#, 0));
        assert!(!accepts(s, r#"{"a":1.5,"b":0}"#, 0));
        assert!(!accepts(s, r#"{"a":01,"b":0}"#, 0));
        assert!(!accepts(s, r#"{"a":1}"#, 0));
        assert!(accepts(r#"{"type":"integer"}"#, "120", 0));
    }

    #[test]
    fn key_clamps_counters() {
        let s = Schema::compile(r#"{"type":"string","maxLength":1000}"#).unwrap();
        let mut a = State::new(&s);
        let mut b = State::new(&s);
        for c in br#""abc"# {
            a.step(&s, 0, *c);
        }
        for c in br#""abcdefgh"# {
            b.step(&s, 0, *c);
        }
        assert_ne!(a, b);
        assert_eq!(a.key(16), b.key(16));
        assert_ne!(a.key(998), b.key(998));
    }
}
