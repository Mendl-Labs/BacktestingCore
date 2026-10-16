//! Regenerates the generated section of `POWER_TABLE.md`.
//!
//! ```text
//! cargo run --release --bin power_table                 # print the generated block to stdout
//! cargo run --release --bin power_table -- --write      # rewrite the block inside ../POWER_TABLE.md in place
//! cargo run --release --bin power_table -- --threads 4  # worker threads (the result does not depend on this)
//! ```
//!
//! `tests/power_table.rs` pins the committed block (an always-on digest check plus an `--ignored` byte-for-byte
//! regeneration that CI runs in release mode).

use portfolio_eval::power::{generated_block, BEGIN_MARKER, END_MARKER};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write = args.iter().any(|a| a == "--write");
    let threads = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);
    let started = std::time::Instant::now();
    let block = match generated_block(threads) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("power table generation failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("generated in {:.1}s on {} thread(s)", started.elapsed().as_secs_f64(), threads);
    if !write {
        print!("{block}");
        return;
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/POWER_TABLE.md");
    let old = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(1);
    });
    let (Some(b), Some(e)) = (old.find(BEGIN_MARKER), old.find(END_MARKER)) else {
        eprintln!("markers not found in {path}");
        std::process::exit(1);
    };
    let end = e + END_MARKER.len() + usize::from(old[e + END_MARKER.len()..].starts_with('\n'));
    let new = format!("{}{}{}", &old[..b], block, &old[end..]);
    std::fs::write(path, new).unwrap_or_else(|e| {
        eprintln!("cannot write {path}: {e}");
        std::process::exit(1);
    });
    eprintln!("rewrote the generated block of {path}");
}
