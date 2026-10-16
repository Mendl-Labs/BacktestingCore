//! Minimal proleptic-Gregorian calendar date (no time zone, no clock, no I/O).
//!
//! A bar is identified by the UTC calendar date of the vendor's daily bar (design S-1). Only what the simulator needs:
//! parsing `YYYY-MM-DD`, ordering, day differences (for `years = days / 365.25` and financing accrual), weekday
//! (for building test calendars) and "same calendar month" (for the `LastBarOfMonth` schedule).

use std::fmt;

/// A calendar date. Field order makes the derived `Ord` chronological.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date {
    year: i32,
    month: u8,
    day: u8,
}

/// Error for malformed or impossible dates.
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

impl Date {
    /// Build a date, rejecting impossible month/day combinations.
    pub fn new(year: i32, month: u8, day: u8) -> Result<Date, DateError> {
        if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
            return Err(DateError(format!("{year:04}-{month:02}-{day:02}")));
        }
        Ok(Date { year, month, day })
    }

    /// Parse exactly `YYYY-MM-DD` (ten ASCII bytes). Anything else is an error: no silent repair.
    pub fn parse(s: &str) -> Result<Date, DateError> {
        let b = s.as_bytes();
        if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
            return Err(DateError(s.to_string()));
        }
        let digits = |r: std::ops::Range<usize>| -> Result<i32, DateError> {
            let mut v = 0i32;
            for &c in &b[r] {
                if !c.is_ascii_digit() {
                    return Err(DateError(s.to_string()));
                }
                v = v * 10 + i32::from(c - b'0');
            }
            Ok(v)
        };
        let y = digits(0..4)?;
        let m = digits(5..7)?;
        let d = digits(8..10)?;
        Date::new(y, m as u8, d as u8).map_err(|_| DateError(s.to_string()))
    }

    pub fn year(self) -> i32 {
        self.year
    }
    pub fn month(self) -> u8 {
        self.month
    }
    pub fn day(self) -> u8 {
        self.day
    }

    /// Days since 1970-01-01 (Howard Hinnant's `days_from_civil`).
    pub fn days_since_epoch(self) -> i64 {
        let y = i64::from(self.year) - i64::from(self.month <= 2);
        let m = i64::from(self.month);
        let d = i64::from(self.day);
        let era = y.div_euclid(400);
        let yoe = y - era * 400;
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }

    /// Inverse of [`Date::days_since_epoch`].
    pub fn from_days_since_epoch(z: i64) -> Date {
        let z = z + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = if m <= 2 { y + 1 } else { y };
        Date { year: y as i32, month: m as u8, day: d as u8 }
    }

    /// `later - self` in whole calendar days (negative if `later` is earlier).
    pub fn days_until(self, later: Date) -> i64 {
        later.days_since_epoch() - self.days_since_epoch()
    }

    pub fn add_days(self, n: i64) -> Date {
        Date::from_days_since_epoch(self.days_since_epoch() + n)
    }

    /// Monday = 0 ... Sunday = 6.
    pub fn weekday(self) -> u8 {
        // 1970-01-01 was a Thursday (= 3).
        (self.days_since_epoch() + 3).rem_euclid(7) as u8
    }

    /// True when both dates fall in the same calendar (year, month).
    pub fn same_month(self, other: Date) -> bool {
        self.year == other.year && self.month == other.month
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_display_roundtrip() {
        let d = Date::parse("2020-02-29").unwrap();
        assert_eq!((d.year(), d.month(), d.day()), (2020, 2, 29));
        assert_eq!(d.to_string(), "2020-02-29");
    }

    #[test]
    fn rejects_malformed_and_impossible_dates() {
        for bad in [
            "2019-02-29",
            "2020-13-01",
            "2020-00-10",
            "2020-01-32",
            "2020/01/01",
            "20200101",
            "2020-1-01",
            " 2020-01-01",
            "2020-01-0a",
            "",
        ] {
            assert!(Date::parse(bad).is_err(), "{bad} should be rejected");
        }
    }

    #[test]
    fn epoch_and_known_day_counts() {
        assert_eq!(Date::parse("1970-01-01").unwrap().days_since_epoch(), 0);
        assert_eq!(Date::parse("2000-03-01").unwrap().days_since_epoch(), 11_017);
        // 2020 is a leap year: Jan 1 -> next Jan 1 is 366 days.
        let a = Date::parse("2020-01-01").unwrap();
        let b = Date::parse("2021-01-01").unwrap();
        assert_eq!(a.days_until(b), 366);
        assert_eq!(b.days_until(a), -366);
    }

    #[test]
    fn civil_roundtrip_over_a_wide_range() {
        // Years ~327 .. ~4160: all four-digit, so the textual roundtrip is well defined too.
        for z in (-600_000i64..800_000).step_by(37) {
            let d = Date::from_days_since_epoch(z);
            assert_eq!(d.days_since_epoch(), z, "{d}");
            assert_eq!(Date::parse(&d.to_string()).unwrap(), d);
        }
    }

    #[test]
    fn weekday_matches_known_dates() {
        // 2020-02-29 was a Saturday, 2019-12-31 a Tuesday, 1970-01-01 a Thursday.
        assert_eq!(Date::parse("2020-02-29").unwrap().weekday(), 5);
        assert_eq!(Date::parse("2019-12-31").unwrap().weekday(), 1);
        assert_eq!(Date::parse("1970-01-01").unwrap().weekday(), 3);
    }

    #[test]
    fn ordering_is_chronological_and_month_test() {
        let a = Date::parse("2019-12-31").unwrap();
        let b = Date::parse("2020-01-01").unwrap();
        assert!(a < b);
        assert!(!a.same_month(b));
        assert!(a.same_month(Date::parse("2019-12-01").unwrap()));
        assert!(!Date::parse("2019-01-31").unwrap().same_month(Date::parse("2020-01-31").unwrap()));
    }
}
