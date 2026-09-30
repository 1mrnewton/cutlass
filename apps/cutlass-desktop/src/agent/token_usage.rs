//! Format provider-reported token usage for the agent chat transcript.

use cutlass_ai::TokenUsage;

/// Human-readable per-prompt usage line, or `None` when the provider reported
/// nothing (local models often leave usage at zero).
///
/// Example: `12.3k tokens in (0% cached) · 456 out · $0.68`
pub(crate) fn format_usage_line(usage: &TokenUsage) -> Option<String> {
    if usage.is_empty() {
        return None;
    }
    let cached_pct = if usage.input_tokens == 0 {
        0
    } else {
        ((usage.cached_input_tokens as f64 / usage.input_tokens as f64) * 100.0).round() as u64
    };
    let mut line = format!(
        "{} tokens in ({}% cached) · {} out",
        format_token_count(usage.input_tokens),
        cached_pct,
        format_token_count(usage.output_tokens),
    );
    if let Some(cost) = usage.cost {
        line.push_str(&format!(" · {}", format_cost(cost)));
    }
    Some(line)
}

fn format_cost(cost: f64) -> String {
    if cost >= 0.01 {
        format!("${cost:.2}")
    } else if cost > 0.0 {
        format!("${cost:.4}")
    } else {
        "$0.00".to_string()
    }
}

fn format_token_count(n: u64) -> String {
    if n >= 10_000 {
        // Round to one decimal place in thousands (12_960 → 13k).
        let tenths = ((n as f64) / 100.0).round() as u64;
        let whole = tenths / 10;
        let frac = tenths % 10;
        if frac == 0 {
            format!("{whole}k")
        } else {
            format!("{whole}.{frac}k")
        }
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_usage_yields_no_line() {
        assert_eq!(format_usage_line(&TokenUsage::default()), None);
    }

    #[test]
    fn formats_small_counts_without_k_suffix() {
        let line = format_usage_line(&TokenUsage {
            input_tokens: 1_234,
            cached_input_tokens: 0,
            output_tokens: 456,
            cost: None,
        })
        .expect("line");
        assert_eq!(line, "1234 tokens in (0% cached) · 456 out");
    }

    #[test]
    fn formats_thousands_with_k_when_at_least_ten_thousand() {
        let line = format_usage_line(&TokenUsage {
            input_tokens: 12_300,
            cached_input_tokens: 0,
            output_tokens: 10_000,
            cost: None,
        })
        .expect("line");
        assert_eq!(line, "12.3k tokens in (0% cached) · 10k out");
    }

    #[test]
    fn rounds_thousands_suffix_instead_of_truncating() {
        assert_eq!(format_token_count(12_960), "13k");
        assert_eq!(format_token_count(12_949), "12.9k");
        assert_eq!(format_token_count(12_950), "13k");
    }

    #[test]
    fn rounds_cache_percent_from_cached_over_input() {
        let line = format_usage_line(&TokenUsage {
            input_tokens: 1_000,
            cached_input_tokens: 333,
            output_tokens: 1,
            cost: None,
        })
        .expect("line");
        assert_eq!(line, "1000 tokens in (33% cached) · 1 out");
    }

    #[test]
    fn includes_cost_only_when_present() {
        let with_cost = format_usage_line(&TokenUsage {
            input_tokens: 100,
            cached_input_tokens: 50,
            output_tokens: 20,
            cost: Some(0.68),
        })
        .expect("line");
        assert_eq!(with_cost, "100 tokens in (50% cached) · 20 out · $0.68");

        let without_cost = format_usage_line(&TokenUsage {
            input_tokens: 100,
            cached_input_tokens: 50,
            output_tokens: 20,
            cost: None,
        })
        .expect("line");
        assert_eq!(without_cost, "100 tokens in (50% cached) · 20 out");
        assert!(!without_cost.contains('$'));
    }

    #[test]
    fn cost_precision_adapts_for_sub_cent_amounts() {
        assert_eq!(format_cost(0.68), "$0.68");
        assert_eq!(format_cost(0.01), "$0.01");
        assert_eq!(format_cost(0.0042), "$0.0042");
        assert_eq!(format_cost(0.0099), "$0.0099");
        assert_eq!(format_cost(0.0), "$0.00");

        let line = format_usage_line(&TokenUsage {
            input_tokens: 100,
            cached_input_tokens: 0,
            output_tokens: 20,
            cost: Some(0.0042),
        })
        .expect("line");
        assert_eq!(line, "100 tokens in (0% cached) · 20 out · $0.0042");
    }

    #[test]
    fn example_line_matches_product_format() {
        let line = format_usage_line(&TokenUsage {
            input_tokens: 12_300,
            cached_input_tokens: 0,
            output_tokens: 456,
            cost: Some(0.68),
        })
        .expect("line");
        assert_eq!(line, "12.3k tokens in (0% cached) · 456 out · $0.68");
    }
}
