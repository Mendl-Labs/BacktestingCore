//! Cadence: which sleeve is due when (design 5.2 level 1, `portfolio-construct::schedule`).
//!
//! Three questions, answered once so that the backtester and the live driver can call the same function:
//!
//! * [`Cadence::Daily`]: every date (`SleeveKind::CryptoTrend` in the driver, `DecisionSchedule::Daily` in `weightsim`).
//! * [`Cadence::CalendarMonthEnd`]: the last CALENDAR day of a month, a wall-clock question a scheduler asks before any
//!   data exists (the driver's `sleeve_due_on(EtfTrend)` = `reference_rules::is_calendar_month_end`).
//! * [`Cadence::LastBarOfMonth`]: the last BAR of each calendar month present in a data calendar (`weightsim`'s
//!   `DecisionSchedule::LastBarOfMonth`; the last bar of the calendar counts, even when its month is incomplete).
//!
//! The two month-end cadences "usually agree" and differ around holidays and late data; that difference is the documented
//! wall-clock versus data-calendar case of the cadence parity test (design 5.4 test 3), not a bug of either.
//!
//! [`BookCadence`] then says which sleeves a run re-targets: `PerSleeve` (T1 semantics, only the due sleeves; the
//! certification mode) or `AllSleevesOnAnyDue` (the live driver as read in finding F1: when any sleeve is due EVERY
//! sleeve is planned).

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
    Daily,
    CalendarMonthEnd,
    LastBarOfMonth,
}

/// Is a sleeve with `cadence` due on `date`? `next_bar` is the date of the next bar in the data calendar (`None` for the
/// last bar of the calendar) and is only read by `LastBarOfMonth`.
pub fn due(cadence: Cadence, date: CivilDate, next_bar: Option<CivilDate>) -> bool {
    match cadence {
        Cadence::Daily => true,
        Cadence::CalendarMonthEnd => date.is_calendar_month_end(),
        Cadence::LastBarOfMonth => next_bar.is_none_or(|n| !date.same_month(n)),
    }
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
