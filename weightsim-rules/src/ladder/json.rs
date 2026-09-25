//! A minimal JSON reader, just enough for the fixture manifest, the key metrics and the mutant expectations.
//!
//! Why not `serde_json`: this crate keeps its dependency set to the two path crates plus `chrono`, so that consuming it
//! from the Engine cannot introduce a second copy of anything. The files read here are machine-written by
//! `export_key.py` (`json.dumps(..., indent=1, sort_keys=True)`): objects, arrays, strings, numbers, booleans, null.
//! Everything else is a parse error; there is no repair.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// Insertion-ordered key/value pairs (duplicate keys are a parse error).
    Obj(Vec<(String, Json)>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError(pub String);

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "json: {}", self.0)
    }
}

impl std::error::Error for JsonError {}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(kv) => kv.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    /// Follow a chain of object keys.
    pub fn path(&self, keys: &[&str]) -> Option<&Json> {
        let mut cur = self;
        for k in keys {
            cur = cur.get(k)?;
        }
        Some(cur)
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(a) => Some(a),
            _ => None,
        }
    }
    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(o) => Some(o),
            _ => None,
        }
    }
    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }
}

/// Parse a complete JSON document (trailing non-whitespace is an error).
pub fn parse(text: &str) -> Result<Json, JsonError> {
    let mut p = Parser { b: text.as_bytes(), i: 0, depth: 0 };
    p.ws();
    let v = p.value()?;
    p.ws();
    if p.i != p.b.len() {
        return Err(p.err("trailing characters after the document"));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
    depth: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> JsonError {
        JsonError(format!("{msg} (byte {})", self.i))
    }
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }
    fn eat(&mut self, c: u8) -> Result<(), JsonError> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected `{}`", c as char)))
        }
    }
    fn lit(&mut self, word: &str, v: Json) -> Result<Json, JsonError> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("bad literal"))
        }
    }
    fn value(&mut self) -> Result<Json, JsonError> {
        self.depth += 1;
        if self.depth > 64 {
            return Err(self.err("nesting too deep"));
        }
        let out = match self.b.get(self.i) {
            None => Err(self.err("unexpected end of input")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(Json::Str),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => Err(self.err("unexpected character")),
        };
        self.depth -= 1;
        out
    }
    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.i;
        while self.i < self.b.len() && matches!(self.b[self.i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
            self.i += 1;
        }
        let tok = std::str::from_utf8(&self.b[start..self.i]).expect("ascii");
        tok.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .map(Json::Num)
            .ok_or_else(|| JsonError(format!("bad number `{tok}` (byte {start})")))
    }
    fn string(&mut self) -> Result<String, JsonError> {
        self.eat(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = *self.b.get(self.i).ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or_else(|| self.err("unterminated escape"))?;
                    self.i += 1;
                    match e {
                        b'"' | b'\\' | b'/' => out.push(e),
                        b'n' => out.push(b'\n'),
                        b't' => out.push(b'\t'),
                        b'r' => out.push(b'\r'),
                        b'u' => {
                            let hex = self.b.get(self.i..self.i + 4).ok_or_else(|| self.err("short \\u escape"))?;
                            let hex = std::str::from_utf8(hex).map_err(|_| self.err("bad \\u escape"))?;
                            let cp = u32::from_str_radix(hex, 16).map_err(|_| self.err("bad \\u escape"))?;
                            let ch = char::from_u32(cp).ok_or_else(|| self.err("unsupported \\u escape"))?;
                            let mut buf = [0u8; 4];
                            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            self.i += 4;
                        }
                        _ => return Err(self.err("unsupported escape")),
                    }
                }
                c if c < 0x20 => return Err(self.err("control character in string")),
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("string is not valid UTF-8"))
    }
    fn array(&mut self) -> Result<Json, JsonError> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(self.err("expected `,` or `]`")),
            }
        }
    }
    fn object(&mut self) -> Result<Json, JsonError> {
        self.eat(b'{')?;
        let mut kv: Vec<(String, Json)> = Vec::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(kv));
        }
        loop {
            self.ws();
            let k = self.string()?;
            if kv.iter().any(|(e, _)| *e == k) {
                return Err(self.err(&format!("duplicate key `{k}`")));
            }
            self.ws();
            self.eat(b':')?;
            self.ws();
            let v = self.value()?;
            kv.push((k, v));
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(kv));
                }
                _ => return Err(self.err("expected `,` or `}`")),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_shapes_the_fixtures_use() {
        let j = parse(
            "{\n \"a\": {\"b\": [1, -2.5e-3, true, null, \"x\\\"y\"]},\n \"c\": 1e-09, \"d\": \"\\u00e9\", \"e\": {}\n}",
        )
        .unwrap();
        let arr = j.path(&["a", "b"]).unwrap().as_array().unwrap();
        assert_eq!(arr[0].as_f64(), Some(1.0));
        assert_eq!(arr[1].as_f64(), Some(-2.5e-3));
        assert_eq!(arr[2].as_bool(), Some(true));
        assert!(arr[3].is_null());
        assert_eq!(arr[4].as_str(), Some("x\"y"));
        assert_eq!(j.get("c").unwrap().as_f64(), Some(1e-9));
        assert_eq!(j.get("d").unwrap().as_str(), Some("é"));
        assert_eq!(j.get("e").unwrap().as_object().unwrap().len(), 0);
        assert!(j.get("missing").is_none());
        assert!(j.path(&["a", "zzz"]).is_none());
    }

    #[test]
    fn rejects_malformed_documents() {
        for bad in [
            "",
            "{",
            "{\"a\":}",
            "{\"a\":1,}",
            "[1,]",
            "[1 2]",
            "{\"a\":1} x",
            "{\"a\":1,\"a\":2}",
            "\"abc",
            "tru",
            "1.2.3",
            "{a:1}",
            "NaN",
            "\"\\q\"",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn nesting_depth_is_bounded() {
        let deep = "[".repeat(100) + &"]".repeat(100);
        assert!(parse(&deep).is_err());
        let ok = "[".repeat(30) + &"]".repeat(30);
        assert!(parse(&ok).is_ok());
    }
}
