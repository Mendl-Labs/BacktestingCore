//! [`BarTime`]: the instant a bar CLOSES, in milliseconds since the Unix epoch (UTC). It is the decision instant of the
//! book simulator (design 3.3): never synthetic, never an index. Daily fixtures identify a bar by the UTC calendar
//! date of the vendor bar (T1 semantics S-1), which maps to midnight UTC of that date ([`BarTime::from_date`]).

use crate::date::Date;
use std::fmt;

/// Bar-close time, milliseconds since 1970-01-01T00:00:00Z. Ordering is chronological.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BarTime(i64);

impl BarTime {
    pub const MS_PER_DAY: i64 = 86_400_000;

    pub fn from_ms(ms: i64) -> BarTime {
        BarTime(ms)
    }

    /// Milliseconds since the epoch.
    pub fn ms(self) -> i64 {
        self.0
    }

    /// Midnight UTC of `d`: how a daily bar identified by its UTC calendar date is placed on the account clock.
    pub fn from_date(d: Date) -> BarTime {
        BarTime(d.days_since_epoch() * BarTime::MS_PER_DAY)
    }

    /// The UTC calendar date this instant falls on (floor division, so pre-1970 instants are correct too).
    pub fn date(self) -> Date {
        Date::from_days_since_epoch(self.0.div_euclid(BarTime::MS_PER_DAY))
    }

    /// Milliseconds since midnight UTC (0 for a daily bar).
    pub fn ms_of_day(self) -> i64 {
        self.0.rem_euclid(BarTime::MS_PER_DAY)
    }
}

impl fmt::Display for BarTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m = self.ms_of_day();
        if m == 0 {
            write!(f, "{}", self.date())
        } else {
            write!(
                f,
                "{}T{:02}:{:02}:{:02}.{:03}Z",
                self.date(),
                m / 3_600_000,
                (m / 60_000) % 60,
                (m / 1000) % 60,
                m % 1000
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_roundtrip_and_display() {
        for s in ["1969-12-31", "1970-01-01", "2020-02-29", "2024-12-31"] {
            let d = Date::parse(s).unwrap();
            let t = BarTime::from_date(d);
            assert_eq!(t.date(), d);
            assert_eq!(t.ms_of_day(), 0);
            assert_eq!(t.to_string(), s);
        }
        assert_eq!(BarTime::from_date(Date::parse("1970-01-02").unwrap()).ms(), 86_400_000);
        assert_eq!(BarTime::from_date(Date::parse("1969-12-31").unwrap()).ms(), -86_400_000);
    }

    #[test]
    fn intraday_instants_floor_to_their_utc_date() {
        let d = Date::parse("2021-03-05").unwrap();
        let t = BarTime::from_ms(BarTime::from_date(d).ms() + 23 * 3_600_000 + 59 * 60_000 + 59_999);
        assert_eq!(t.date(), d);
        assert_eq!(t.to_string(), "2021-03-05T23:59:59.999Z");
        let before = BarTime::from_ms(BarTime::from_date(d).ms() - 1);
        assert_eq!(before.date(), d.add_days(-1));
        assert!(before < BarTime::from_date(d));
    }
}
