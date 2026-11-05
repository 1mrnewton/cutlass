//! `Http`: the [`Transport`] over `ureq`.

use std::io::Read;
use std::time::Duration;

use crate::{Head, Transport};

/// HTTP and HTTPS over `ureq`; rustls checks certificates against the
/// Mozilla roots `ureq` carries.
pub struct Http {
    agent: ureq::Agent,
}

impl Http {
    /// A client that gives up on a silent server after 30 seconds.
    #[must_use]
    pub fn new() -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_recv_body(Some(Duration::from_secs(30)))
            .timeout_recv_response(Some(Duration::from_secs(30)))
            .http_status_as_error(false)
            .build();
        Self {
            agent: config.into(),
        }
    }
}

impl Default for Http {
    fn default() -> Self {
        Self::new()
    }
}

/// `bytes 0-0/1234` → 1234.
fn total_from_content_range(value: &str) -> Option<u64> {
    value.rsplit_once('/')?.1.trim().parse().ok()
}

/// `bytes 100-199/1234` → 100.
fn start_from_content_range(value: &str) -> Option<u64> {
    let span = value.trim().strip_prefix("bytes")?.trim();
    span.split_once('-')?.0.trim().parse().ok()
}

impl Transport for Http {
    fn head(&self, url: &str) -> Result<Head, String> {
        // A one-byte range rather than HEAD: some hosts answer HEAD
        // differently from GET, and the range answer shows both things.
        let response = self
            .agent
            .get(url)
            .header("Range", "bytes=0-0")
            .call()
            .map_err(|error| format!("{url}: {error}"))?;
        let status = response.status().as_u16();
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        };
        match status {
            206 => Ok(Head {
                length: header("content-range")
                    .as_deref()
                    .and_then(total_from_content_range),
                ranges: true,
            }),
            200 => Ok(Head {
                length: header("content-length").and_then(|value| value.trim().parse().ok()),
                ranges: false,
            }),
            _ => Err(format!("{url}: the server answered {status}")),
        }
    }

    fn get(
        &self,
        url: &str,
        range: Option<(u64, u64)>,
        sink: &mut dyn FnMut(&[u8]) -> bool,
    ) -> Result<(), String> {
        let mut request = self.agent.get(url);
        if let Some((start, end)) = range {
            if end <= start {
                return Ok(());
            }
            request = request.header("Range", format!("bytes={start}-{}", end - 1));
        }
        let mut response = request.call().map_err(|error| format!("{url}: {error}"))?;
        let status = response.status().as_u16();
        let expected = if range.is_some() { 206 } else { 200 };
        if status != expected {
            return Err(format!("{url}: the server answered {status}"));
        }
        if let Some((start, _)) = range {
            let served = response
                .headers()
                .get("content-range")
                .and_then(|value| value.to_str().ok())
                .and_then(start_from_content_range);
            if served != Some(start) {
                return Err(format!("{url}: the server sent a different range"));
            }
        }
        let mut reader = response.body_mut().with_config().limit(u64::MAX).reader();
        let mut buffer = vec![0u8; 64 * 1024];
        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|error| format!("{url}: {error}"))?;
            if read == 0 {
                return Ok(());
            }
            if !sink(&buffer[..read]) {
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{start_from_content_range, total_from_content_range};

    #[test]
    fn content_range_gives_the_start_and_the_total() {
        assert_eq!(total_from_content_range("bytes 0-0/1234"), Some(1234));
        assert_eq!(total_from_content_range("bytes 0-0/*"), None);
        assert_eq!(start_from_content_range("bytes 100-199/1234"), Some(100));
        assert_eq!(start_from_content_range("items 1-2/3"), None);
    }
}
