//! Token masks: a byte trie of the vocabulary, walked once per distinct (grammar, clamped state)
//! and memoised; every mask gets an id into one append-only table (the GPU copy is indexed by it).
//!
//! Inside a string body most of the vocabulary is allowed, so a full walk visits nearly every
//! trie node. "Plain" tokens (they stay in a string body: no quote, no backslash, no control
//! byte, whole characters) are instead answered by `plain_le[min(room, longest)]`, the plain
//! tokens of at most that many characters; only the other tokens' trie is walked there.

use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::matcher::State;
use crate::schema::Schema;

struct Trie {
    /// Per node: children (byte, node), and the tokens ending here
    children: Vec<Vec<(u8, u32)>>,
    ends: Vec<Vec<u32>>,
}

impl Trie {
    fn new(tokens: &[Option<Vec<u8>>], keep: impl Fn(usize) -> bool) -> Trie {
        let mut t = Trie {
            children: vec![Vec::new()],
            ends: vec![Vec::new()],
        };
        for (id, bytes) in tokens.iter().enumerate() {
            let Some(bytes) = bytes else { continue };
            if bytes.is_empty() || !keep(id) {
                continue;
            }
            let mut n = 0usize;
            for &b in bytes {
                n = match t.children[n].iter().find(|(c, _)| *c == b) {
                    Some((_, child)) => *child as usize,
                    None => {
                        t.children.push(Vec::new());
                        t.ends.push(Vec::new());
                        let child = t.children.len() - 1;
                        t.children[n].push((b, child as u32));
                        child
                    }
                };
            }
            t.ends[n].push(id as u32);
        }
        t
    }
}

pub struct Matcher {
    grammar: u32,
    state: State,
    dead: bool,
}

#[derive(Default, Clone, Copy)]
pub struct Stats {
    pub masks: u64,
    pub mask_s: f64,
    pub steps: u64,
    pub step_s: f64,
    pub tokens: u64,
    pub rejected: u64,
}

pub struct Store {
    tokens: Vec<Option<Vec<u8>>>,
    eos: Vec<u32>,
    trie: Trie,
    /// The trie of the tokens that are not plain
    other: Trie,
    /// `plain_le[c]`: the plain tokens of at most `c` characters, `c` in `0..=longest`
    plain_le: Vec<u32>,
    pub words: usize,
    max_ws: u8,
    /// The longest token in bytes: the clamp of `State::key`
    longest: u32,
    grammars: Vec<Schema>,
    memo: FxHashMap<(u32, State), u32>,
    masks: Vec<u32>,
    taken: usize,
    matchers: Vec<Option<Matcher>>,
    free: Vec<u32>,
    pub stats: Stats,
}

/// Mask id 0: end-of-sequence only (finished or rejected replies)
pub const EOS_ONLY: u32 = 0;

impl Store {
    pub fn new(tokens: Vec<Option<Vec<u8>>>, eos: Vec<u32>, vocab: usize, max_ws: u8) -> Store {
        let words = vocab.div_ceil(32);
        let longest = tokens.iter().flatten().map(Vec::len).max().unwrap_or(1) as u32;
        let chars = plain_chars(&tokens);
        let mut plain_le = vec![0u32; (longest as usize + 1) * words];
        for (t, c) in chars.iter().enumerate() {
            if let Some(c) = *c {
                for limit in c as usize..=longest as usize {
                    plain_le[limit * words + t / 32] |= 1 << (t % 32);
                }
            }
        }
        let mut s = Store {
            trie: Trie::new(&tokens, |_| true),
            other: Trie::new(&tokens, |t| chars[t].is_none()),
            plain_le,
            tokens,
            eos,
            words,
            max_ws,
            longest,
            grammars: Vec::new(),
            memo: FxHashMap::default(),
            masks: Vec::new(),
            taken: 0,
            matchers: Vec::new(),
            free: Vec::new(),
            stats: Stats::default(),
        };
        let mut m = vec![0u32; words];
        s.set_eos(&mut m);
        s.masks.extend_from_slice(&m);
        s
    }

    fn set_eos(&self, m: &mut [u32]) {
        for &e in &self.eos {
            m[e as usize / 32] |= 1 << (e % 32);
        }
    }

    pub fn compile(&mut self, schema: &str) -> Result<u32, String> {
        self.grammars.push(Schema::compile(schema)?);
        Ok(self.grammars.len() as u32 - 1)
    }

    pub fn matcher(&mut self, grammar: u32) -> u32 {
        let m = Matcher {
            grammar,
            state: State::new(&self.grammars[grammar as usize]),
            dead: false,
        };
        match self.free.pop() {
            Some(h) => {
                self.matchers[h as usize] = Some(m);
                h
            }
            None => {
                self.matchers.push(Some(m));
                self.matchers.len() as u32 - 1
            }
        }
    }

    pub fn release(&mut self, h: u32) {
        if self.matchers[h as usize].take().is_some() {
            self.free.push(h);
        }
    }

    /// Feed each matcher its new tokens; the mask id each one's next token must satisfy
    pub fn step(&mut self, handles: &[u32], tokens: &[Vec<u32>]) -> Vec<u32> {
        let t0 = Instant::now();
        let mut out = Vec::with_capacity(handles.len());
        for (&h, new) in handles.iter().zip(tokens) {
            let mut m = self.matchers[h as usize].take().expect("released matcher");
            let g = &self.grammars[m.grammar as usize];
            for &t in new {
                self.stats.tokens += 1;
                if m.dead || self.eos.contains(&t) {
                    continue;
                }
                let ok = match self.tokens.get(t as usize) {
                    Some(Some(bytes)) => bytes.iter().all(|&b| m.state.step(g, self.max_ws, b)),
                    _ => false,
                };
                if !ok {
                    m.dead = true;
                    self.stats.rejected += 1;
                }
            }
            let id = if m.dead {
                EOS_ONLY
            } else {
                self.mask_id(m.grammar, &m.state)
            };
            self.matchers[h as usize] = Some(m);
            out.push(id);
        }
        self.stats.steps += 1;
        self.stats.step_s += t0.elapsed().as_secs_f64();
        out
    }

    fn mask_id(&mut self, grammar: u32, state: &State) -> u32 {
        let key = (grammar, state.key(self.longest));
        if let Some(&id) = self.memo.get(&key) {
            return id;
        }
        let t0 = Instant::now();
        let g = &self.grammars[grammar as usize];
        let mut m = vec![0u32; self.words];
        if state.accepting(g) {
            self.set_eos(&mut m);
        }
        match key.1.body_room() {
            Some(room) => {
                let c = room.min(self.longest) as usize;
                for (w, p) in m.iter_mut().zip(&self.plain_le[c * self.words..]) {
                    *w |= p;
                }
                walk(&self.other, 0, &key.1, g, self.max_ws, &mut m);
            }
            None => walk(&self.trie, 0, &key.1, g, self.max_ws, &mut m),
        }
        let id = (self.masks.len() / self.words) as u32;
        self.masks.extend_from_slice(&m);
        self.memo.insert(key, id);
        self.stats.masks += 1;
        self.stats.mask_s += t0.elapsed().as_secs_f64();
        id
    }

    /// Masks added since the last call, as their words
    pub fn take_new(&mut self) -> &[u32] {
        let start = self.taken;
        self.taken = self.masks.len();
        &self.masks[start..]
    }

    pub fn mask(&self, id: u32) -> &[u32] {
        &self.masks[id as usize * self.words..(id as usize + 1) * self.words]
    }

    pub fn count(&self) -> usize {
        self.masks.len() / self.words
    }
}

/// Each token's characters if it is plain (stays in a string body, per the matcher itself)
fn plain_chars(tokens: &[Option<Vec<u8>>]) -> Vec<Option<u32>> {
    let none = Schema {
        nodes: Vec::new(),
        root: 0,
    };
    let start = u32::MAX - 1;
    tokens
        .iter()
        .map(|t| {
            let bytes = t.as_ref()?;
            let mut s = State::string_body(start);
            if bytes.is_empty() || !bytes.iter().all(|&b| s.step(&none, 0, b)) {
                return None;
            }
            s.body_room().map(|room| start - room)
        })
        .collect()
}

fn walk(trie: &Trie, node: usize, state: &State, g: &Schema, max_ws: u8, m: &mut [u32]) {
    for &(b, child) in &trie.children[node] {
        let mut s = *state;
        if s.step(g, max_ws, b) {
            for &t in &trie.ends[child as usize] {
                m[t as usize / 32] |= 1 << (t % 32);
            }
            walk(trie, child as usize, &s, g, max_ws, m);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn allowed(store: &Store, id: u32) -> Vec<u32> {
        let m = store.mask(id);
        (0..store.words as u32 * 32)
            .filter(|t| m[*t as usize / 32] >> (t % 32) & 1 == 1)
            .collect()
    }

    #[test]
    fn masks_follow_the_schema() {
        let vocab: Vec<Option<Vec<u8>>> = ["{", "\"", "a", "\"a\"", "}", "\":", "ab", "true", "\"}", "x"]
            .iter()
            .map(|s| Some(s.as_bytes().to_vec()))
            .chain([None])
            .collect();
        let eos = 10;
        let mut st = Store::new(vocab, vec![eos], 11, 0);
        let g = st
            .compile(r#"{"type":"object","properties":{"a":{"type":"string","maxLength":2}},"required":["a"]}"#)
            .unwrap();
        let h = st.matcher(g);
        let id = st.step(&[h], &[vec![]])[0];
        assert_eq!(allowed(&st, id), vec![0]);
        // {"a  -> the key closes: `"` or `":`
        let id = st.step(&[h], &[vec![0, 1, 2]])[0];
        assert_eq!(allowed(&st, id), vec![1, 5]);
        // {"a":"ab -> room 0: only the closing quote (alone or with `}`)
        let id = st.step(&[h], &[vec![5, 1, 6]])[0];
        assert_eq!(allowed(&st, id), vec![1, 8]);
        let id = st.step(&[h], &[vec![8]])[0];
        assert_eq!(allowed(&st, id), vec![eos]);
        // a rejected token: end-of-sequence only from then on
        let h2 = st.matcher(g);
        assert_eq!(st.step(&[h2], &[vec![9]])[0], EOS_ONLY);
        assert_eq!(st.stats.rejected, 1);
    }

    #[test]
    fn body_fast_path_equals_full_walk() {
        let words = ["\"", "a", "ab", "a\"", "\"}", "\\", "\\n", "é", "\u{e9}\"", " ", "x y"];
        let mut vocab: Vec<Option<Vec<u8>>> =
            words.iter().map(|s| Some(s.as_bytes().to_vec())).collect();
        vocab.push(Some(vec![0xC3]));
        vocab.push(Some(vec![0xA9, b'"']));
        let n = vocab.len();
        let schema = r#"{"type":"object","properties":{"s":{"type":"string","minLength":2,"maxLength":4}},"required":["s"]}"#;
        let mut st = Store::new(vocab.clone(), vec![], n, 1);
        let g = st.compile(schema).unwrap();
        let open = vec!["{\"s\":\"".to_string()];
        for text in ["", "a", "ab", "abc", "abcd", "é"] {
            let h = st.matcher(g);
            let mut s = st.matchers[h as usize].as_ref().unwrap().state;
            let gr = &st.grammars[g as usize];
            for b in open[0].bytes().chain(text.bytes()) {
                assert!(s.step(gr, 1, b));
            }
            assert!(s.body_room().is_some());
            let mut full = vec![0u32; st.words];
            walk(&st.trie, 0, &s.key(st.longest), gr, 1, &mut full);
            st.matchers[h as usize].as_mut().unwrap().state = s;
            let id = st.step(&[h], &[vec![]])[0];
            assert_eq!(st.mask(id), &full[..], "after {text:?}");
        }
    }

    #[test]
    fn memo_shares_masks_across_lengths() {
        let vocab: Vec<Option<Vec<u8>>> = ["\"", "a", "b"]
            .iter()
            .map(|s| Some(s.as_bytes().to_vec()))
            .collect();
        let mut st = Store::new(vocab, vec![], 3, 0);
        let g = st.compile(r#"{"type":"string","maxLength":100}"#).unwrap();
        let h = st.matcher(g);
        let first = st.step(&[h], &[vec![0, 1]])[0];
        let later = st.step(&[h], &[vec![2, 1, 2]])[0];
        assert_eq!(first, later);
    }
}
