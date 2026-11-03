//! UTC timestamps for one log line.
//!
//! The clock is microseconds since the Unix epoch. A time before the epoch
//! is marked, because a log line still has to be one field.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub(crate) fn format_system_time(time: SystemTime) -> String {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => format_timestamp(duration),
        Err(_) => "unknown-time".to_owned(),
    }
}

pub(crate) fn format_timestamp(since_epoch: Duration) -> String {
    let secs = since_epoch.as_secs();
    let micros = since_epoch.subsec_micros();
    let days = secs / 86_400;
    let tod = secs % 86_400;
    let Some(days) = i64::try_from(days).ok() else {
        return "unknown-time".to_owned();
    };
    let Some((year, month, day)) = civil_from_days(days) else {
        return "unknown-time".to_owned();
    };
    let hour = tod / 3_600;
    let minute = (tod % 3_600) / 60;
    let second = tod % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{micros:06}Z")
}

/// Howard Hinnant's civil-from-days conversion. Days are counted from the Unix epoch.
fn civil_from_days(days: i64) -> Option<(i64, u32, u32)> {
    let z = days.checked_add(719_468)?;
    let era = if z >= 0 {
        z.checked_div(146_097)?
    } else {
        z.checked_sub(146_096)?.checked_div(146_097)?
    };
    let doe = u64::try_from(z.checked_sub(era.checked_mul(146_097)?)?).ok()?;
    let yoe = doe
        .checked_sub(doe / 1_460)?
        .checked_add(doe / 36_524)?
        .checked_sub(doe / 146_096)?
        / 365;
    let year = i64::try_from(yoe)
        .ok()?
        .checked_add(era.checked_mul(400)?)?;
    let doy = doe.checked_sub(365 * yoe + yoe / 4 - yoe / 100)?;
    let mp = (5 * doy + 2) / 153;
    let day = doy.checked_sub((153 * mp + 2) / 5)?.checked_add(1)?;
    let month = if mp < 10 {
        mp.checked_add(3)?
    } else {
        mp.checked_sub(9)?
    };
    let year = if month <= 2 {
        year.checked_add(1)?
    } else {
        year
    };
    Some((year, u32::try_from(month).ok()?, u32::try_from(day).ok()?))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::{format_system_time, format_timestamp};

    #[test]
    fn timestamps_match_utc() {
        let cases = [
            (Duration::from_secs(0), "1970-01-01T00:00:00.000000Z"),
            (Duration::from_secs(1), "1970-01-01T00:00:01.000000Z"),
            (Duration::from_secs(86_400), "1970-01-02T00:00:00.000000Z"),
            (Duration::from_secs(86_401), "1970-01-02T00:00:01.000000Z"),
            (
                Duration::from_secs(951_782_400),
                "2000-02-29T00:00:00.000000Z",
            ),
            (
                Duration::from_secs(1_700_000_000),
                "2023-11-14T22:13:20.000000Z",
            ),
        ];
        for (duration, expected) in cases {
            assert_eq!(format_timestamp(duration), expected);
        }
        let micros = Duration::new(1_700_000_000, 123_456_789);
        assert_eq!(format_timestamp(micros), "2023-11-14T22:13:20.123456Z");
    }

    #[test]
    fn a_time_before_the_epoch_is_marked() {
        let earlier = UNIX_EPOCH.checked_sub(Duration::from_secs(1)).unwrap();
        assert_eq!(format_system_time(earlier), "unknown-time");
        assert_ne!(format_system_time(SystemTime::now()), "unknown-time");
    }
}
