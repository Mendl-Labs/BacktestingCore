//! Shared fixtures for the portfolio-construct tests: a seeded PRNG, the planner-test universe, a case builder.
//! Everything is synthetic; no vendor data and no account data appear anywhere in this crate.
#![allow(dead_code)]

use portfolio_construct::*;

/// SplitMix64: a tiny deterministic PRNG (no `rand` dependency), the generator SignalEngine's planner tests use.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform integer in `lo..=hi`.
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next_u64() % (hi - lo + 1)
    }
    pub fn chance(&mut self, percent: u64) -> bool {
        self.range(0, 99) < percent
    }
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.range(0, items.len() as u64 - 1) as usize]
    }
    /// Uniform f64 in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.range(0, i as u64) as usize;
            items.swap(i, j);
        }
    }
}

pub const SPY: usize = 0;
pub const EFA: usize = 1;
pub const IEF: usize = 2;
pub const DBC: usize = 3;
pub const VNQ: usize = 4;
pub const BTC: usize = 5;
pub const ETH: usize = 6;

/// The seven-instrument universe of the planner's tests (synthetic prices): five ETFs on "alpaca", two coins on "kraken".
pub fn universe() -> Vec<InstrumentFacts> {
    vec![
        InstrumentFacts::new("SPY", "alpaca", "us_etf", 500.0),
        InstrumentFacts::new("EFA", "alpaca", "us_etf", 80.0),
        InstrumentFacts::new("IEF", "alpaca", "us_etf", 95.0),
        InstrumentFacts::new("DBC", "alpaca", "us_etf", 25.0),
        InstrumentFacts::new("VNQ", "alpaca", "us_etf", 90.0),
        InstrumentFacts::new("BTC/USD", "kraken", "crypto_spot", 60000.0),
        InstrumentFacts::new("ETH/USD", "kraken", "crypto_spot", 3000.0),
    ]
}

/// A lot table with the SHAPE of the venue adapters' rules (fractional ETFs to 9 decimals with a minimum order value of
/// 1; coins to 8 decimals with a minimum quantity and value). The numbers are test inputs, not claims about any venue.
pub fn planner_rounder() -> LotRounder {
    let etf = LotRule::new(9).with_min_notional(1.0);
    let coin = |min_q: f64| LotRule::new(8).with_min_quantity(min_q).with_min_notional(0.5);
    LotRounder::new()
        .with("SPY", etf)
        .with("EFA", etf)
        .with("IEF", etf)
        .with("DBC", etf)
        .with("VNQ", etf)
        .with("QQQ", etf)
        .with("BTC/USD", coin(0.0001))
        .with("ETH/USD", coin(0.001))
}

/// The ETF sleeve of the planner tests: id "etf", weights on the five ETFs.
pub fn etf(share: f64, w: [f64; 5]) -> SleeveTargets {
    SleeveTargets::long_only("etf", share, (0..5).map(|j| (j, w[j])).collect())
}

/// The crypto sleeve: id "crypto", weights on BTC and ETH.
pub fn crypto(share: f64, btc: f64, eth: f64) -> SleeveTargets {
    SleeveTargets::long_only("crypto", share, vec![(BTC, btc), (ETH, eth)])
}

pub fn both_sleeves() -> Vec<SleeveTargets> {
    vec![etf(0.5, [0.2; 5]), crypto(0.5, 0.5, 0.5)]
}

/// An owned bundle of everything `construct` needs, with the planner's defaults: equity 5000 all cash, allocation 5000,
/// reserve 5%, fee 0.25%, filter 10 / 2%, the planner-shaped lot table, targets rounded to 8 decimals, no limits.
pub struct Case {
    pub equity: f64,
    pub allocated: Option<f64>,
    pub sleeves: Vec<SleeveTargets>,
    pub rs: RiskScale,
    pub limits: Limits,
    pub instruments: Vec<InstrumentFacts>,
    pub margin: Box<dyn MarginModel>,
    pub filter: TradeFilter,
    pub rounder: Option<LotRounder>,
    pub funding: Funding,
    pub target_dp: Option<u32>,
    pub unmanaged: f64,
}

impl Case {
    pub fn new(instruments: Vec<InstrumentFacts>, sleeves: Vec<SleeveTargets>) -> Case {
        Case {
            equity: 5000.0,
            allocated: Some(5000.0),
            sleeves,
            rs: RiskScale::ONE,
            limits: Limits::unlimited(),
            instruments,
            margin: Box::new(NoMargin),
            filter: TradeFilter::PLANNER_DEFAULT,
            rounder: Some(planner_rounder()),
            funding: Funding::Cash {
                cash: 5000.0,
                reserve_fraction: 0.05,
                fee_rate: 0.0025,
                credit_sell_proceeds: true,
            },
            target_dp: Some(8),
            unmanaged: 0.0,
        }
    }

    /// The seven-instrument planner universe.
    pub fn planner(sleeves: Vec<SleeveTargets>) -> Case {
        Case::new(universe(), sleeves)
    }

    pub fn with_cash(mut self, cash: f64) -> Case {
        if let Funding::Cash { reserve_fraction, fee_rate, credit_sell_proceeds, .. } = self.funding {
            self.funding = Funding::Cash { cash, reserve_fraction, fee_rate, credit_sell_proceeds };
        }
        self
    }

    pub fn with_credit(mut self, credit: bool) -> Case {
        if let Funding::Cash { cash, reserve_fraction, fee_rate, .. } = self.funding {
            self.funding = Funding::Cash { cash, reserve_fraction, fee_rate, credit_sell_proceeds: credit };
        }
        self
    }

    pub fn held(mut self, idx: usize, units: f64) -> Case {
        self.instruments[idx].held_units = units;
        self
    }

    pub fn run(&self) -> Result<ConstructOutput, ConstructRefusal> {
        construct(&ConstructInputs {
            equity: self.equity,
            allocated_capital: self.allocated,
            sleeves: &self.sleeves,
            risk_scale: self.rs,
            limits: &self.limits,
            instruments: &self.instruments,
            margin: self.margin.as_ref(),
            trade_filter: self.filter,
            rounding: self.rounder.as_ref().map(|r| r as &dyn QuantityRounder),
            funding: self.funding,
            target_dp: self.target_dp,
            unmanaged_gross: self.unmanaged,
        })
    }

    pub fn ok(&self) -> ConstructOutput {
        match self.run() {
            Ok(o) => o,
            Err(e) => panic!("construct refused: {e}"),
        }
    }
}

/// The trade for `symbol` (panics with the whole output when there is none).
pub fn trade<'a>(out: &'a ConstructOutput, symbol: &str) -> &'a TradeIntent {
    out.trades.iter().find(|t| t.symbol == symbol).unwrap_or_else(|| panic!("no trade for {symbol}: {out:#?}"))
}

pub fn skip_reason<'a>(out: &'a ConstructOutput, symbol: &str) -> &'a SkipReason {
    &out.skipped
        .iter()
        .find(|s| s.symbol == symbol)
        .unwrap_or_else(|| panic!("{symbol} not skipped: {:?}", out.skipped))
        .reason
}

pub fn line<'a>(out: &'a ConstructOutput, symbol: &str) -> &'a Line {
    out.lines.iter().find(|l| l.symbol == symbol).unwrap_or_else(|| panic!("no line for {symbol}"))
}

pub fn brief(out: &ConstructOutput) -> Vec<(String, Side, f64)> {
    out.trades.iter().map(|t| (t.symbol.clone(), t.side, t.quantity)).collect()
}

pub fn assert_close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (diff {}, tol {tol})", (a - b).abs());
}

/// FNV-1a over the little-endian bytes of the given f64 bit patterns (a digest for bit-identity checks; no dependency).
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub fn u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    pub fn str(&mut self, s: &str) {
        for b in s.bytes() {
            self.u64(u64::from(b));
        }
        self.u64(0xff);
    }
}

/// Hash every observable part of an output.
pub fn digest_output(h: &mut Fnv, o: &ConstructOutput) {
    h.f64(o.capital_base);
    h.f64(o.risk_scale_applied);
    for t in &o.target_notional {
        h.f64(*t);
    }
    for l in &o.lines {
        h.str(&l.symbol);
        h.f64(l.held_units);
        h.f64(l.current_notional);
        h.f64(l.target_notional);
    }
    for t in &o.trades {
        h.str(&t.symbol);
        h.u64(if t.side == Side::Buy { 1 } else { 2 });
        h.f64(t.quantity);
        h.f64(t.price);
        h.f64(t.notional);
        h.f64(t.est_fee);
        h.u64(u64::from(t.reducing));
    }
    h.u64(o.skipped.len() as u64);
    h.f64(o.gross);
    h.f64(o.net);
    h.f64(o.margin_used);
    h.f64(o.projected_gross);
    h.f64(o.funding_left.unwrap_or(-1.0));
}

// ---------------------------------------------------------------------------------------------------------------
// A cloneable random-book generator (seeded, reproducible from the seed printed on failure).
// ---------------------------------------------------------------------------------------------------------------

/// Plain-data description of one construction step, cloneable so a test can perturb it (permute, scale, tighten).
#[derive(Clone)]
pub struct G {
    pub equity: f64,
    pub allocated: Option<f64>,
    pub sleeves: Vec<SleeveTargets>,
    pub rs: RiskScale,
    pub limits: Limits,
    pub instruments: Vec<InstrumentFacts>,
    /// 0 = NoMargin, 1 = OANDA (R3), 2 = Alpaca Reg T (R4)
    pub margin: u8,
    pub filter: TradeFilter,
    pub rounder: Option<LotRounder>,
    pub funding: Funding,
    pub target_dp: Option<u32>,
    pub unmanaged: f64,
}

impl G {
    pub fn run(&self) -> Result<ConstructOutput, ConstructRefusal> {
        let no = NoMargin;
        let oa = OandaMargin::r3_default();
        let al = AlpacaRegT::r4_default();
        let margin: &dyn MarginModel = match self.margin {
            0 => &no,
            1 => &oa,
            _ => &al,
        };
        construct(&ConstructInputs {
            equity: self.equity,
            allocated_capital: self.allocated,
            sleeves: &self.sleeves,
            risk_scale: self.rs,
            limits: &self.limits,
            instruments: &self.instruments,
            margin,
            trade_filter: self.filter,
            rounding: self.rounder.as_ref().map(|r| r as &dyn QuantityRounder),
            funding: self.funding,
            target_dp: self.target_dp,
            unmanaged_gross: self.unmanaged,
        })
    }

    /// The same book with the instruments listed in the order `perm` (new position p holds old instrument perm[p]),
    /// every sleeve's indices remapped, and sleeves and weights shuffled by `rng`.
    pub fn permuted(&self, perm: &[usize], rng: &mut SplitMix64) -> G {
        let n = self.instruments.len();
        let mut new_of_old = vec![0usize; n];
        for (p, &o) in perm.iter().enumerate() {
            new_of_old[o] = p;
        }
        let mut g = self.clone();
        g.instruments = perm.iter().map(|&o| self.instruments[o].clone()).collect();
        for s in &mut g.sleeves {
            for w in &mut s.weights {
                w.0 = new_of_old[w.0];
            }
            rng.shuffle(&mut s.weights);
        }
        rng.shuffle(&mut g.sleeves);
        g
    }
}

pub fn gen(seed: u64) -> G {
    let mut r = SplitMix64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x00AB_CDEF);
    let n = r.range(3, 8) as usize;
    let venues = ["alpaca", "kraken"];
    let classes = ["a", "b", "c"];
    let mut instruments: Vec<InstrumentFacts> = (0..n)
        .map(|k| {
            let price = 5.0 + r.range(0, 59_500) as f64 / 100.0;
            let (venue, class): (&str, &str) = (venues[r.range(0, 1) as usize], classes[r.range(0, 2) as usize]);
            let mut i = InstrumentFacts::new(&format!("S{k}"), venue, class, price)
                .with_margin_rate(*r.pick(&[0.02, 0.05, 0.25, 0.75]));
            if r.chance(50) {
                let mut units = r.range(1, 300) as f64 / 10.0;
                if r.chance(15) {
                    units = -units;
                }
                i.held_units = units;
            }
            if r.chance(5) {
                i.marginable = false;
            }
            i
        })
        .collect();
    // sleeves: shares on a 0.05 grid summing to at most 1
    let ns = r.range(1, 3) as usize;
    let mut remaining = 20u64;
    let mut sleeves = Vec::new();
    for k in 0..ns {
        let max_s = (remaining - (ns - 1 - k) as u64).min(20);
        let s = r.range(1, max_s);
        remaining -= s;
        let share = s as f64 * 0.05;
        let signed = r.chance(55);
        let mut weights: Vec<(usize, f64)> = Vec::new();
        let mut idx: Vec<usize> = (0..n).collect();
        r.shuffle(&mut idx);
        let take = r.range(1, n as u64) as usize;
        if signed {
            for &j in idx.iter().take(take) {
                weights.push((j, (r.range(0, 300) as f64 - 150.0) / 100.0));
            }
            sleeves.push(SleeveTargets::signed(&format!("sl{k}"), share, 2.0, weights));
        } else {
            let mut units = 20u64;
            for &j in idx.iter().take(take) {
                let u = r.range(0, units.min(8));
                units -= u;
                weights.push((j, u as f64 * 0.05));
            }
            sleeves.push(SleeveTargets::long_only(&format!("sl{k}"), share, weights));
        }
    }
    let equity = r.range(1000, 60_000) as f64;
    let allocated = if r.chance(50) { None } else { Some(r.range(500, 80_000) as f64) };
    let rs = RiskScale::new(*r.pick(&[1.0, 0.8, 0.6122448979591837, 0.5]), *r.pick(&[1.0, 1.0, 1.0, 0.75, 0.5, 0.25]));
    let limits = if r.chance(50) {
        Limits::unlimited()
    } else {
        let mut l = Limits::unlimited()
            .with_max_gross(*r.pick(&[0.5, 1.0, 1.5, 3.0]))
            .with_max_net(*r.pick(&[f64::INFINITY, 1.0, 2.0]))
            .with_max_position(*r.pick(&[f64::INFINITY, 0.3, 1.0]))
            .with_shorting(r.chance(60));
        l.leverage_max_gross = *r.pick(&[f64::INFINITY, 2.0]);
        if r.chance(50) {
            l = l.with_class_cap("a", *r.pick(&[0.4, 1.0]));
        }
        if r.chance(15) {
            l = l.with_policy(LimitPolicy::PlannerFaithful);
        }
        l
    };
    let funding = match r.range(0, 2) {
        0 => Funding::Unconstrained,
        1 => Funding::Cash {
            cash: r.range(0, 100_000) as f64,
            reserve_fraction: *r.pick(&[0.0, 0.05]),
            fee_rate: *r.pick(&[0.0, 0.0025]),
            credit_sell_proceeds: r.chance(70),
        },
        _ => Funding::BuyingPower {
            buying_power: r.range(0, 300_000) as f64,
            reserve_fraction: *r.pick(&[0.0, 0.05]),
            fee_rate: *r.pick(&[0.0, 0.0025]),
        },
    };
    let filter = *r.pick(&[
        TradeFilter::NONE,
        TradeFilter::PLANNER_DEFAULT,
        TradeFilter::new(5.0, 0.05),
        TradeFilter::new(50.0, 0.0),
    ]);
    let rounder = if r.chance(50) {
        let mut t = LotRounder::new();
        for i in &instruments {
            let mut rule = LotRule::new(*r.pick(&[0u32, 2, 4, 9]));
            let mn = *r.pick(&[0.0, 1.0, 10.0]);
            rule = rule.with_min_notional(mn);
            if r.chance(20) {
                rule = rule.with_min_quantity(1.0);
            }
            t = t.with(&i.symbol, rule);
        }
        Some(t)
    } else {
        None
    };
    let target_dp = *r.pick(&[Some(8u32), None]);
    let unmanaged = if r.chance(30) { r.range(0, 20_000) as f64 } else { 0.0 };
    let margin = r.range(0, 2) as u8;
    // a price gap: occasionally an instrument has no price
    if r.chance(10) {
        let k = r.range(0, n as u64 - 1) as usize;
        instruments[k].price = None;
    }
    G { equity, allocated, sleeves, rs, limits, instruments, margin, filter, rounder, funding, target_dp, unmanaged }
}
