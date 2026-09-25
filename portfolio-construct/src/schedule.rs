//! Cadence: which sleeve is due when (design 5.2 level 1, `portfolio-construct::schedule`).
//!
//! Four questions, answered once so that the backtester and the live driver can call the same function:
//!
//! * [`Cadence::Daily`]: every date (`SleeveKind::CryptoTrend` in the driver, `DecisionSchedule::Daily` in `weightsim`).
//! * [`Cadence::DecisionPending`]: THE LIVE MONTHLY (ETF, later FX) CADENCE (council Ruling 1, `product-mandate/
//!   COUNCIL_ETF_TIMING_AND_CADENCE.md`). The sleeve is EVALUATED on every driver run and PLANNED iff the newest decision
//!   its data can support, `D_computable`, is later than the newest decision already acted on, `D_acted` (or nothing was acted
//!   on yet: the entry). `D_computable` is the last bar of the newest month that the data itself proves over, i.e. the last
//!   bar of the month before the month of the newest bar (`MonthEndMode::NextMonthBar`: a later-month bar must exist). It
//!   needs the panel and the ledger, so it is not a function of the date alone: see [`DueInputs`], [`due_on`],
//!   [`computable_decision_date`], [`decision_pending`] and [`advance_acted`].
//! * [`Cadence::LastBarOfMonth`]: the last BAR of each calendar month present in a data calendar (`weightsim`'s
//!   `DecisionSchedule::LastBarOfMonth`; the last bar of the calendar counts, even when its month is incomplete). This is the
//!   backtester's decision calendar (the certified key), NOT a live trigger: a live run cannot know it is looking at the last
//!   bar of the month before a later bar exists.
//! * [`Cadence::CalendarMonthEnd`]: the last CALENDAR day of a month, a wall-clock question a scheduler asks before any data
//!   exists (the driver's old `sleeve_due_on(EtfTrend)` = `reference_rules::is_calendar_month_end`). **It is the defect of
//!   finding U3 and must NOT be used for the live ETF cadence**: it fires on a day whose data does not yet contain the new
//!   month, so the rule silently decides the PREVIOUS month (a month behind), and on weekend month-ends it does not fire at
//!   all when the last session was a Friday. It is kept only so that the cadence parity test can name "the documented
//!   wall-clock case" and so that the U3 characterisation can be reproduced.
//!
//! The two month-end cadences "usually agree" and differ around weekends, holidays and late data; that difference is the
//! documented wall-clock versus data-calendar case of the cadence parity test (design 5.4 test 3), not a bug of the data
//! calendar. What the LIVE driver must use is `DecisionPending`, which agrees with `LastBarOfMonth` on WHICH decision is
//! acted on (the last bar of a month) and acts exactly once per month, on the run after the first bar of the new month
//! exists: with runs whose panel holds only bars dated strictly before the run date, that is the day after the first
//! new-month session (Ruling 1, acceptance 1-3, 5 and 10).
//!
//! [`BookCadence`] then says which sleeves a run re-targets: `PerSleeve` (T1 semantics, only the due sleeves; the
//! certification mode and the council's live default, Ruling 3) or `AllSleevesOnAnyDue` (the live driver as read in finding F1:
//! when any sleeve is due EVERY sleeve is planned; kept as a labelled legacy-emulation mode).

use std::fmt;

/// A calendar date (proleptic Gregorian, no time zone, no clock).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CivilDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// An impossible date.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DateError(pub String);

impl fmt::Display for DateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid date: {}", self.0)
    }
}

impl std::error::Error for DateError {}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u8) -> u8 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

impl CivilDate {
    pub fn new(year: i32, month: u8, day: u8) -> Result<CivilDate, DateError> {
        if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
            return Err(DateError(format!("{year:04}-{month:02}-{day:02}")));
        }
        Ok(CivilDate { year, month, day })
    }

    /// The next calendar day.
    pub fn succ(self) -> CivilDate {
        if self.day < days_in_month(self.year, self.month) {
            CivilDate { day: self.day + 1, ..self }
        } else if self.month < 12 {
            CivilDate { year: self.year, month: self.month + 1, day: 1 }
        } else {
            CivilDate { year: self.year + 1, month: 1, day: 1 }
        }
    }

    pub fn same_month(self, other: CivilDate) -> bool {
        self.year == other.year && self.month == other.month
    }

    /// Is this the last calendar day of its month? Calendar-free (no holiday table).
    pub fn is_calendar_month_end(self) -> bool {
        !self.same_month(self.succ())
    }
}

/// When a sleeve's decision is due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cadence {
    /// Every date.
    Daily,
    /// The last CALENDAR day of a month: a wall-clock predicate of a scheduler that has no data yet.
    ///
    /// **Do NOT use this for the live ETF cadence.** It is the U3 defect (see the module docs), retained only as the
    /// documented wall-clock case of the parity test. The live monthly cadence is [`Cadence::DecisionPending`].
    CalendarMonthEnd,
    /// The last BAR of each calendar month present in a data calendar (the backtester's decision calendar).
    LastBarOfMonth,
    /// The live monthly cadence (Ruling 1): evaluate every run, act iff `D_computable > D_acted` (or nothing acted yet).
    /// Needs the panel and the ledger; answer it with [`due_on`] (a stateless [`due`] call has neither and says "not due").
    DecisionPending,
}

/// Everything a cadence may look at on one run. `date` and `next_bar` serve the calendar cadences; `computable_decision` and
/// `last_acted` serve [`Cadence::DecisionPending`] (compute the former with [`computable_decision_date`] from the bars the
/// run can see, take the latter from the decision ledger).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DueInputs {
    /// The run date.
    pub date: CivilDate,
    /// The date of the next bar in the data calendar (`None` for the last bar), read only by `LastBarOfMonth`.
    pub next_bar: Option<CivilDate>,
    /// `D_computable`: the newest decision the visible data can support (`None`: no completed month yet).
    pub computable_decision: Option<CivilDate>,
    /// `D_acted`: the newest decision already acted on for this account and sleeve (`None`: nothing acted yet).
    pub last_acted: Option<CivilDate>,
}

impl DueInputs {
    /// The inputs of a stateless question: only the date (and the next bar); nothing computable, nothing acted.
    pub fn stateless(date: CivilDate, next_bar: Option<CivilDate>) -> DueInputs {
        DueInputs { date, next_bar, computable_decision: None, last_acted: None }
    }
}

/// Is a sleeve with `cadence` due on this run? Total over every cadence.
pub fn due_on(cadence: Cadence, i: &DueInputs) -> bool {
    match cadence {
        Cadence::Daily => true,
        Cadence::CalendarMonthEnd => i.date.is_calendar_month_end(),
        Cadence::LastBarOfMonth => i.next_bar.is_none_or(|n| !i.date.same_month(n)),
        Cadence::DecisionPending => decision_pending(i.computable_decision, i.last_acted),
    }
}

/// Is a sleeve with `cadence` due on `date`? `next_bar` is the date of the next bar in the data calendar (`None` for the
/// last bar of the calendar) and is only read by `LastBarOfMonth`. A stateless call has no panel and no ledger, so
/// `DecisionPending` is never pending here (nothing is computable): use [`due_on`] for it.
pub fn due(cadence: Cadence, date: CivilDate, next_bar: Option<CivilDate>) -> bool {
    due_on(cadence, &DueInputs::stateless(date, next_bar))
}

/// Bars the panel is not strictly ascending in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BarsError {
    /// `bars[index]` is not later than `bars[index - 1]` (duplicates and unsorted input are refused, never repaired).
    NotStrictlyAscending { index: usize },
}

impl fmt::Display for BarsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BarsError::NotStrictlyAscending { index } => {
                write!(f, "bar dates must be strictly ascending (bar {index} is not later than its predecessor)")
            }
        }
    }
}

impl std::error::Error for BarsError {}

/// The bars a run on `as_of` may see when the vendor delivers only CLOSED bars: those dated strictly BEFORE `as_of` (the
/// driver's `ClosedBarsOnly`). `bars` must be ascending (checked by [`computable_decision_date`], not here).
pub fn closed_bars(bars: &[CivilDate], as_of: CivilDate) -> &[CivilDate] {
    &bars[..bars.partition_point(|b| *b < as_of)]
}

/// `D_computable`: the newest decision the visible bars support, calendar-free: the LAST BAR OF THE NEWEST MONTH PROVEN
/// COMPLETE, i.e. the last bar dated in a month earlier than the month of the newest bar (`NextMonthBar`: the month is over
/// only once a bar of a later month exists). `Ok(None)` when no bar precedes the newest bar's month (the first month of data,
/// or no data). Errors on non-ascending input.
///
/// Monotone in the data: appending a bar never lowers the result.
pub fn computable_decision_date(bars: &[CivilDate]) -> Result<Option<CivilDate>, BarsError> {
    for i in 1..bars.len() {
        if bars[i] <= bars[i - 1] {
            return Err(BarsError::NotStrictlyAscending { index: i });
        }
    }
    let Some(&newest) = bars.last() else {
        return Ok(None);
    };
    Ok(bars.iter().rev().find(|b| !b.same_month(newest)).copied())
}

/// The pending-decision predicate (Ruling 1): planned iff a decision is computable and it is later than the newest one
/// acted on, or nothing was acted on yet (the entry, Ruling 9). Nothing computable is never pending.
pub fn decision_pending(computable: Option<CivilDate>, last_acted: Option<CivilDate>) -> bool {
    match computable {
        None => false,
        Some(c) => last_acted.is_none_or(|a| c > a),
    }
}

/// The ledger update after acting on `acted_on`: `D_acted` is MONOTONE, so it never moves backwards (a late or replayed
/// record of an older decision leaves it where it is) and acting twice on the same decision is a no-op.
pub fn advance_acted(last_acted: Option<CivilDate>, acted_on: CivilDate) -> CivilDate {
    match last_acted {
        Some(a) if a >= acted_on => a,
        _ => acted_on,
    }
}

/// One run's evaluation of a `DecisionPending` sleeve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecisionEvaluation {
    pub computable: Option<CivilDate>,
    pub pending: bool,
}

/// `D_computable` from the bars a run sees and whether it is pending against `last_acted`.
pub fn evaluate_decision(
    visible_bars: &[CivilDate],
    last_acted: Option<CivilDate>,
) -> Result<DecisionEvaluation, BarsError> {
    let computable = computable_decision_date(visible_bars)?;
    Ok(DecisionEvaluation { computable, pending: decision_pending(computable, last_acted) })
}

/// Which sleeves a run re-targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BookCadence {
    /// Each sleeve on its own schedule: only the due sleeves are planned (T1 `PerSleeve`).
    PerSleeve,
    /// The live driver's behaviour (finding F1): when ANY sleeve is due a run happens and EVERY sleeve is planned.
    AllSleevesOnAnyDue,
}

/// The sleeves a run plans. `due[s]` says the sleeve's own decision is due, `tradable[s]` that its market can trade at
/// all today (an ETF sleeve on a weekend cannot). `PerSleeve` plans exactly the due sleeves; `AllSleevesOnAnyDue` plans
/// every tradable sleeve when at least one is due and nothing otherwise.
pub fn plan_flags(mode: BookCadence, due: &[bool], tradable: &[bool]) -> Vec<bool> {
    let any = due.iter().any(|d| *d);
    match mode {
        BookCadence::PerSleeve => due.iter().zip(tradable).map(|(d, t)| *d && *t).collect(),
        BookCadence::AllSleevesOnAnyDue => tradable.iter().map(|t| any && *t).collect(),
    }
}

/// `plan_flags` over cadences: sleeve `s` is due iff `due_on(cadences[s], &inputs[s])`, then the mode decides who is planned.
/// `PerSleeve` is the council's live default (only pending ETF sleeves reach the planner, crypto daily); `AllSleevesOnAnyDue`
/// is the legacy emulation of the old driver (a run re-plans every tradable sleeve).
pub fn plan_flags_for(mode: BookCadence, cadences: &[Cadence], inputs: &[DueInputs], tradable: &[bool]) -> Vec<bool> {
    assert_eq!(cadences.len(), inputs.len());
    assert_eq!(cadences.len(), tradable.len());
    let dues: Vec<bool> = cadences.iter().zip(inputs).map(|(c, i)| due_on(*c, i)).collect();
    plan_flags(mode, &dues, tradable)
}
