//! Windowing logic for walk-forward analysis

use chrono::{NaiveDate, Duration};

#[derive(Debug, Clone)]
pub struct WalkForwardWindow {
    pub train_start: NaiveDate,
    pub train_end: NaiveDate,
    pub test_start: NaiveDate,
    pub test_end: NaiveDate,
}

/// Generate rolling (or anchored) train/test windows with optional embargo gap
pub fn generate_windows(
    start: NaiveDate,
    end: NaiveDate,
    train_size: Duration,
    test_size: Duration,
    step_size: Duration,
    embargo_days: i64,
    anchored: bool,
) -> Vec<WalkForwardWindow> {
    let mut windows = Vec::new();
    let mut train_start = start;
    let embargo = Duration::days(embargo_days);
    loop {
        let train_end = if anchored {
            // Anchored mode: training always starts from `start`, and
            // the window expands as we step forward
            start + train_size + (Duration::days((windows.len() as i64) * step_size.num_days()))
        } else {
            train_start + train_size
        };
        let test_start = train_end + embargo;
        let test_end = test_start + test_size;
        if test_end > end {
            break;
        }
        windows.push(WalkForwardWindow {
            train_start: if anchored { start } else { train_start },
            train_end,
            test_start,
            test_end,
        });
        train_start = train_start + step_size;
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    #[test]
    fn generate_windows_basic() {
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let windows = generate_windows(
            start, end,
            Duration::days(180), Duration::days(60), Duration::days(30),
            0, false,
        );
        assert!(!windows.is_empty());
        for w in &windows {
            assert!(w.train_start < w.train_end);
            assert!(w.test_start <= w.test_end);
            assert!(w.test_end <= end);
        }
    }

    #[test]
    fn generate_windows_with_embargo() {
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let windows = generate_windows(
            start, end,
            Duration::days(180), Duration::days(60), Duration::days(30),
            5, false,
        );
        for w in &windows {
            let gap = w.test_start - w.train_end;
            assert!(gap.num_days() >= 5, "embargo not respected");
        }
    }

    #[test]
    fn generate_windows_anchored() {
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2024, 6, 1).unwrap();
        let windows = generate_windows(
            start, end,
            Duration::days(180), Duration::days(60), Duration::days(30),
            0, true,
        );
        for w in &windows {
            assert_eq!(w.train_start, start, "anchored training should start from beginning");
        }
    }

    #[test]
    fn generate_windows_too_short_returns_empty() {
        let start = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2023, 2, 1).unwrap();
        let windows = generate_windows(
            start, end,
            Duration::days(180), Duration::days(60), Duration::days(30),
            0, false,
        );
        assert!(windows.is_empty());
    }
}
