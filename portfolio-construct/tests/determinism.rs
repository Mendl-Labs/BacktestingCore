//! Determinism: the same inputs give bit-identical outputs, in one process and across platforms.
//!
//! The crate uses only IEEE-exact operations (`+ - * /`, `sqrt`, `floor`, `ceil`, comparisons, `total_cmp`), a table for
//! powers of ten and no `powi`/`exp`/`ln`, no threads, no clock, no randomness, no hash-map iteration. The pinned digest
//! below was produced on aarch64 Linux and must be reproduced by the x86_64 CI runner; a difference means a platform
//! dependence crept in.

mod common;

use common::*;
use portfolio_construct::*;

fn book_digest(cases: std::ops::Range<u64>) -> u64 {
    let mut h = Fnv::new();
    for seed in cases {
        let g = gen(seed);
        h.u64(seed);
        match g.run() {
            Ok(o) => {
                h.u64(1);
                digest_output(&mut h, &o);
            }
            Err(e) => {
                h.u64(2);
                h.str(&format!("{e:?}"));
            }
        }
    }
    h.0
}

fn ladder_digest() -> u64 {
    let ladder = Ladder::new(
        0.03,
        vec![
            Rung { at: 0.05, action: RungAction::Shrink { scale: 0.75 } },
            Rung { at: 0.10, action: RungAction::Shrink { scale: 0.5 } },
            Rung { at: 0.20, action: RungAction::HaltFlatten },
        ],
        0.5,
    )
    .unwrap();
    let mut h = Fnv::new();
    let mut rng = SplitMix64(2026);
    for _ in 0..300 {
        let mut st = LadderState::new();
        let mut equity = 100_000.0 * (1.0 + rng.unit());
        let mut day_start = equity;
        for step in 0..60 {
            if step % 5 == 0 {
                day_start = equity;
            }
            equity *= 1.0 + (rng.unit() - 0.52) * 0.04;
            let d = ladder.step(&mut st, equity, day_start);
            h.f64(d.scale);
            h.u64(d.code as u64);
            h.f64(d.drawdown);
            h.f64(d.daily_loss);
            if st.is_halted() {
                break;
            }
        }
    }
    h.0
}

fn allocator_digest() -> u64 {
    let mut h = Fnv::new();
    let mut rng = SplitMix64(77);
    for _ in 0..200 {
        let n = 2 + (rng.range(0, 2) as usize);
        let rets: Vec<Vec<f64>> = (0..n).map(|k| (0..120).map(|_| (rng.unit() - 0.5) * 0.02 * (1.0 + k as f64)).collect()).collect();
        let refs: Vec<&[f64]> = rets.iter().map(|v| v.as_slice()).collect();
        let mut a = AllocatorState::new(
            AllocatorSpec::InverseVol { lookback_bars: 60, floor: 0.0, freeze: FreezeRule::AtReviewDates },
            n,
            1.0,
        )
        .unwrap();
        a.review(&refs, &vec![100; n]);
        for s in a.shares() {
            h.f64(*s);
        }
    }
    h.0
}

#[test]
fn repeated_runs_are_bit_identical() {
    let a = book_digest(0..400);
    let b = book_digest(0..400);
    assert_eq!(a, b);
    assert_eq!(ladder_digest(), ladder_digest());
    assert_eq!(allocator_digest(), allocator_digest());
    // A fresh output compared field by field, not through a hash.
    let g = gen(17);
    let (x, y) = (g.run(), g.run());
    assert_eq!(x, y);
}

#[test]
fn a_clone_of_the_inputs_gives_the_same_bits() {
    for seed in 0..300 {
        let g = gen(seed);
        let h = g.clone();
        match (g.run(), h.run()) {
            (Ok(a), Ok(b)) => {
                let (mut ha, mut hb) = (Fnv::new(), Fnv::new());
                digest_output(&mut ha, &a);
                digest_output(&mut hb, &b);
                assert_eq!(ha.0, hb.0, "seed {seed}");
            }
            (Err(a), Err(b)) => assert_eq!(a, b),
            _ => panic!("seed {seed}"),
        }
    }
}

/// Pinned digests, produced on aarch64 Linux (WSL2). The x86_64 CI runner must reproduce them.
#[test]
fn digests_match_the_pinned_goldens() {
    let books = book_digest(0..600);
    let ladder = ladder_digest();
    let alloc = allocator_digest();
    println!("book digest      = {books:#018x}");
    println!("ladder digest    = {ladder:#018x}");
    println!("allocator digest = {alloc:#018x}");
    assert_eq!(books, PINNED_BOOKS, "construct digest changed");
    assert_eq!(ladder, PINNED_LADDER, "ladder digest changed");
    assert_eq!(alloc, PINNED_ALLOC, "allocator digest changed");
}

const PINNED_BOOKS: u64 = 0x0a77_1686_f9c9_f021;
const PINNED_LADDER: u64 = 0xfb10_5b52_a86b_8ff7;
const PINNED_ALLOC: u64 = 0xba86_e011_18c6_2341;
