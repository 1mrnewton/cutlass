//! Cloud transcription via an OpenAI-compatible `/audio/transcriptions`
//! endpoint — the hardware-independent counterpart to the local whisper.cpp
//! backend.
//!
//! "Local-first, never local-only": this backend is **pure HTTP** (no C/C++ or
//! GPU toolchain), so it compiles into the default build and runs anywhere the
//! editor does — including machines where the local runtime can't (e.g.
//! whisper.cpp doesn't yet support the newest Apple Silicon, ggml-org/
//! whisper.cpp#3722). Selected by `[ml] transcribe_provider = "cloud"`; the
//! analysis audio (mono `[-1, 1]`) is muxed to a 16-bit PCM WAV, uploaded as
//! multipart form data with `response_format=verbose_json`, and the reply is
//! parsed into our word-timed [`Transcript`]. Same backends OpenAI, Groq,
//! and llama.cpp-server speak, so "cloud providers later" stays config, not
//! code — mirroring `cutlass-ai`'s `OpenAiCompatProvider`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::config::MlSection;
use crate::transcribe::{
    Segment, Transcribe, TranscribeError, TranscribeOptions, Transcript, Word,
};

/// OpenAI-compatible cloud transcriber. Cheap to construct (no network); the
/// request happens in [`Transcribe::transcribe`].
pub struct CloudTranscriber {
    base_url: String,
    model: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl CloudTranscriber {
    pub fn new(base_url: &str, model: &str, api_key: Option<String>) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key,
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .build(),
        }
    }

    /// Build from the `[ml]` config: requires `base_url` and `cloud_model`, and
    /// resolves `api_key_env`. Returns [`TranscribeError::NotConfigured`]
    /// naming what's missing, so the worker can surface it.
    pub fn from_config(cfg: &MlSection) -> Result<Self, TranscribeError> {
        let base_url = cfg
            .base_url
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                TranscribeError::NotConfigured(
                    "cloud transcription needs [ml] base_url (e.g. https://api.openai.com/v1)"
                        .into(),
                )
            })?;
        let model = cfg
            .cloud_model
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| {
                TranscribeError::NotConfigured(
                    "cloud transcription needs [ml] cloud_model (e.g. whisper-1)".into(),
                )
            })?;
        let api_key = cfg.resolve_api_key().map_err(TranscribeError::NotConfigured)?;
        Ok(Self::new(base_url, model, api_key))
    }
}

impl Transcribe for CloudTranscriber {
    fn transcribe(
        &self,
        audio: &[f32],
        sample_rate: u32,
        options: &TranscribeOptions,
        cancel: &AtomicBool,
        on_progress: &mut dyn FnMut(f32),
    ) -> Result<Transcript, TranscribeError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(TranscribeError::Cancelled);
        }
        if audio.is_empty() {
            return Ok(Transcript::default());
        }
        on_progress(0.0);

        let wav = encode_wav_pcm16(audio, sample_rate);
        // Translate-to-English uses the sibling endpoint; both share the schema.
        let endpoint = if options.translate {
            "audio/translations"
        } else {
            "audio/transcriptions"
        };
        let url = format!("{}/{}", self.base_url, endpoint);

        let mut fields: Vec<(&str, String)> = vec![
            ("model", self.model.clone()),
            ("response_format", "verbose_json".into()),
            // Word + segment spans; servers that don't support it return segments.
            ("timestamp_granularities[]", "segment".into()),
            ("timestamp_granularities[]", "word".into()),
        ];
        // The translations endpoint always outputs English, so it has no
        // language input — only send the hint when transcribing verbatim.
        if !options.translate {
            if let Some(lang) = options.language.as_deref().filter(|s| !s.is_empty()) {
                fields.push(("language", lang.to_string()));
            }
        }

        let boundary = make_boundary();
        let body = build_multipart(&boundary, &fields, "audio.wav", "audio/wav", &wav);

        if cancel.load(Ordering::Relaxed) {
            return Err(TranscribeError::Cancelled);
        }

        let mut req = self.agent.post(&url).set(
            "Content-Type",
            &format!("multipart/form-data; boundary={boundary}"),
        );
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }

        let response = match req.send_bytes(&body) {
            Ok(response) => response,
            Err(ureq::Error::Status(status, response)) => {
                let message = response
                    .into_string()
                    .unwrap_or_else(|_| "<unreadable error body>".to_string());
                return Err(TranscribeError::Backend(format!(
                    "HTTP {status} from {url}: {}",
                    truncate(message.trim(), 400)
                )));
            }
            Err(ureq::Error::Transport(t)) => {
                return Err(TranscribeError::Backend(format!("{url}: {t}")));
            }
        };

        let raw = response
            .into_string()
            .map_err(|e| TranscribeError::Backend(format!("could not read response: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| TranscribeError::Backend(format!("response is not JSON: {e}")))?;
        on_progress(1.0);
        Ok(parse_verbose_json(&json))
    }
}

/// Parse an OpenAI-style `verbose_json` transcription into a [`Transcript`].
/// Word timing comes back as a flat top-level `words` array; we fold it into
/// the matching segment. Factored out so fixtures can drive it without HTTP.
fn parse_verbose_json(json: &serde_json::Value) -> Transcript {
    let language = json["language"].as_str().map(str::to_string);

    let mut segments: Vec<Segment> = json["segments"]
        .as_array()
        .map(|segs| {
            segs.iter()
                .map(|s| Segment {
                    text: s["text"].as_str().unwrap_or_default().to_string(),
                    start: s["start"].as_f64().unwrap_or(0.0),
                    end: s["end"].as_f64().unwrap_or(0.0),
                    words: Vec::new(),
                })
                .collect()
        })
        .unwrap_or_default();

    if let Some(words) = json["words"].as_array() {
        let parsed: Vec<Word> = words
            .iter()
            .filter_map(|w| {
                let text = w["word"].as_str().or_else(|| w["text"].as_str())?;
                Some(Word {
                    text: text.to_string(),
                    start: w["start"].as_f64().unwrap_or(0.0),
                    end: w["end"].as_f64().unwrap_or(0.0),
                    confidence: None,
                })
            })
            .collect();
        assign_words_to_segments(&mut segments, parsed);
    }

    // No segments (some servers only return a bare `text`): keep the words.
    if segments.is_empty() {
        if let Some(text) = json["text"].as_str().map(str::trim).filter(|t| !t.is_empty()) {
            segments.push(Segment {
                text: text.to_string(),
                start: 0.0,
                end: 0.0,
                words: Vec::new(),
            });
        }
    }

    Transcript { segments, language }
}

/// Fold a flat, time-ordered word list into time-ordered segments: a word
/// belongs to the last segment that starts at or before it. Linear (both are
/// already ordered); when there are no segments the words are dropped (no span
/// to attach them to).
fn assign_words_to_segments(segments: &mut [Segment], words: Vec<Word>) {
    if segments.is_empty() {
        return;
    }
    let mut seg = 0;
    for word in words {
        while seg + 1 < segments.len() && word.start >= segments[seg + 1].start {
            seg += 1;
        }
        segments[seg].words.push(word);
    }
}

/// Mux mono `f32` samples in `[-1, 1]` into a canonical 16-bit PCM WAV.
fn encode_wav_pcm16(audio: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (audio.len() * 2) as u32;
    let byte_rate = sample_rate * 2; // mono, 2 bytes/sample
    let mut buf = Vec::with_capacity(44 + data_len as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // format = PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // channels = mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    for &s in audio {
        let v = (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Assemble a `multipart/form-data` body: text `fields` (duplicate names are
/// allowed, for `timestamp_granularities[]`) followed by the `file` part.
fn build_multipart(
    boundary: &str,
    fields: &[(&str, String)],
    filename: &str,
    file_content_type: &str,
    file: &[u8],
) -> Vec<u8> {
    let mut body = Vec::with_capacity(file.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!("Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n")
            .as_bytes(),
    );
    body.extend_from_slice(format!("Content-Type: {file_content_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(file);
    body.extend_from_slice(b"\r\n");
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}

/// A unique multipart boundary. The body is mostly binary WAV, so a long,
/// timestamp-seeded ASCII boundary effectively never collides with content.
fn make_boundary() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("----cutlass-ml-{nanos:032x}")
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_config_requires_base_url_and_model() {
        let mut cfg = MlSection {
            transcribe_provider: crate::TranscribeProvider::Cloud,
            ..MlSection::default()
        };
        assert!(matches!(
            CloudTranscriber::from_config(&cfg),
            Err(TranscribeError::NotConfigured(m)) if m.contains("base_url")
        ));

        cfg.base_url = Some("https://api.openai.com/v1/".into());
        assert!(matches!(
            CloudTranscriber::from_config(&cfg),
            Err(TranscribeError::NotConfigured(m)) if m.contains("cloud_model")
        ));

        cfg.cloud_model = Some("whisper-1".into());
        let t = CloudTranscriber::from_config(&cfg).expect("now configured");
        // Trailing slash trimmed so endpoint joins cleanly.
        assert_eq!(t.base_url, "https://api.openai.com/v1");
        assert_eq!(t.model, "whisper-1");
    }

    #[test]
    fn wav_header_is_well_formed() {
        let wav = encode_wav_pcm16(&[0.0, 1.0, -1.0], 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 3 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 6);
        let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_len, 6);
        // Sample rate round-trips at byte offset 24.
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        // Full-scale samples clamp to i16 extremes.
        assert_eq!(i16::from_le_bytes([wav[46], wav[47]]), i16::MAX);
        assert_eq!(i16::from_le_bytes([wav[48], wav[49]]), -i16::MAX);
    }

    #[test]
    fn multipart_carries_fields_and_file() {
        let body = build_multipart(
            "BND",
            &[
                ("model", "whisper-1".into()),
                ("timestamp_granularities[]", "segment".into()),
                ("timestamp_granularities[]", "word".into()),
            ],
            "audio.wav",
            "audio/wav",
            b"RIFFdata",
        );
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("--BND\r\n"));
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("whisper-1"));
        // Duplicate field name appears twice (segment + word granularity).
        assert_eq!(text.matches("timestamp_granularities[]").count(), 2);
        assert!(text.contains("filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        assert!(text.contains("RIFFdata"));
        assert!(text.trim_end().ends_with("--BND--"));
    }

    #[test]
    fn parses_verbose_json_with_words_folded_into_segments() {
        let json = serde_json::json!({
            "language": "english",
            "text": "hello world again",
            "segments": [
                { "start": 0.0, "end": 1.0, "text": " hello world" },
                { "start": 1.0, "end": 1.6, "text": " again" }
            ],
            "words": [
                { "word": "hello", "start": 0.0, "end": 0.5 },
                { "word": "world", "start": 0.5, "end": 1.0 },
                { "word": "again", "start": 1.1, "end": 1.6 }
            ]
        });
        let t = parse_verbose_json(&json);
        assert_eq!(t.language.as_deref(), Some("english"));
        assert_eq!(t.segments.len(), 2);
        let first: Vec<&str> = t.segments[0].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(first, ["hello", "world"]);
        let second: Vec<&str> = t.segments[1].words.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(second, ["again"]);
        assert_eq!(t.text(), "hello world again");
    }

    #[test]
    fn parses_segments_only_response() {
        let json = serde_json::json!({
            "language": "en",
            "segments": [ { "start": 0.0, "end": 2.0, "text": "just segments" } ]
        });
        let t = parse_verbose_json(&json);
        assert_eq!(t.segments.len(), 1);
        assert!(t.segments[0].words.is_empty());
        assert_eq!(t.segments[0].text, "just segments");
    }

    #[test]
    fn falls_back_to_bare_text_when_no_segments() {
        let json = serde_json::json!({ "text": "  bare transcript  " });
        let t = parse_verbose_json(&json);
        assert_eq!(t.segments.len(), 1);
        assert_eq!(t.segments[0].text, "bare transcript");
        assert!(t.language.is_none());
    }

    #[test]
    fn empty_audio_short_circuits_without_request() {
        let t = CloudTranscriber::new("http://127.0.0.1:1/v1", "whisper-1", None);
        let cancel = AtomicBool::new(false);
        let out = t
            .transcribe(&[], 16_000, &TranscribeOptions::default(), &cancel, &mut |_| {})
            .expect("empty audio yields an empty transcript without hitting the network");
        assert!(out.is_empty());
    }

    #[test]
    fn cancelled_before_request() {
        let t = CloudTranscriber::new("http://127.0.0.1:1/v1", "whisper-1", None);
        let cancel = AtomicBool::new(true);
        let err = t
            .transcribe(&[0.1, 0.2], 16_000, &TranscribeOptions::default(), &cancel, &mut |_| {})
            .unwrap_err();
        assert!(matches!(err, TranscribeError::Cancelled));
    }
}
