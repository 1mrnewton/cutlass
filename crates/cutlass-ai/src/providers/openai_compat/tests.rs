use super::*;

fn run(fixture: &str) -> (ChatTurn, String) {
    let cancel = AtomicBool::new(false);
    let mut streamed = String::new();
    let turn = consume_sse(fixture.as_bytes(), &cancel, &mut |t| streamed.push_str(t))
        .expect("fixture parses");
    (turn, streamed)
}

#[test]
fn text_only_stream() {
    let fixture = concat!(
        "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"The timeline \"}}]}\n",
        "\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"is 12s long.\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
        "data: [DONE]\n",
    );
    let (turn, streamed) = run(fixture);
    assert_eq!(turn.text, "The timeline is 12s long.");
    assert_eq!(streamed, turn.text);
    assert_eq!(turn.finish, FinishReason::Stop);
    assert!(turn.tool_calls.is_empty());
    assert!(turn.usage.is_none());
}

#[test]
fn usage_only_final_chunk_is_parsed() {
    let fixture = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":8,\"prompt_tokens_details\":{\"cached_tokens\":100},\"cost\":0.0042}}\n",
        "data: [DONE]\n",
    );
    let (turn, _) = run(fixture);
    assert_eq!(
        turn.usage,
        Some(TokenUsage {
            input_tokens: 120,
            cached_input_tokens: 100,
            output_tokens: 8,
            cost: Some(0.0042),
        })
    );
}

#[test]
fn tool_call_assembles_from_fragments() {
    let fixture = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"trim_clip\",\"arguments\":\"\"}}]}}]}\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"clip\\\": 12,\"}}]}}]}\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\" \\\"start\\\": 14.0, \\\"duration\\\": 4.0}\"}}]}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n",
        "data: [DONE]\n",
    );
    let (turn, _) = run(fixture);
    assert_eq!(turn.finish, FinishReason::ToolCalls);
    assert_eq!(turn.tool_calls.len(), 1);
    let call = &turn.tool_calls[0];
    assert_eq!(call.id, "call_1");
    assert_eq!(call.name, "trim_clip");
    assert_eq!(
        call.arguments,
        serde_json::json!({ "clip": 12, "start": 14.0, "duration": 4.0 })
    );
}

#[test]
fn parallel_tool_calls_keep_their_indices() {
    let fixture = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"function\":{\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":1}\"}}]}}]}\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":2}\"}}]}}]}\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n",
        "data: [DONE]\n",
    );
    let (turn, _) = run(fixture);
    assert_eq!(turn.tool_calls.len(), 2);
    assert_eq!(turn.tool_calls[0].arguments["clip"], 1);
    assert_eq!(turn.tool_calls[1].arguments["clip"], 2);
}

#[test]
fn cancellation_stops_the_stream() {
    let cancel = AtomicBool::new(true);
    let err = consume_sse(
        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(err, ProviderError::Cancelled));
}

#[test]
fn malformed_chunks_and_arguments_are_protocol_errors() {
    let cancel = AtomicBool::new(false);
    let err = consume_sse("data: {not json}\n".as_bytes(), &cancel, &mut |_| {}).unwrap_err();
    assert!(matches!(err, ProviderError::Protocol(_)));

    let fixture = concat!(
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"x\",\"function\":{\"name\":\"trim_clip\",\"arguments\":\"{oops\"}}]}}]}\n",
        "data: [DONE]\n",
    );
    let err = consume_sse(fixture.as_bytes(), &cancel, &mut |_| {}).unwrap_err();
    match err {
        ProviderError::Protocol(msg) => assert!(msg.contains("trim_clip"), "{msg}"),
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[test]
fn request_body_includes_tools_and_messages() {
    let provider = OpenAiCompatProvider::new("http://localhost:11434/v1/", "qwen3", None);
    assert_eq!(provider.base_url, "http://localhost:11434/v1");

    let messages = vec![
        Message::system("You edit video timelines."),
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_1".into(),
                name: "remove_clip".into(),
                arguments: serde_json::json!({"clip": 3}),
            }],
        },
        Message::tool_result("call_1", "removed clip 3"),
    ];
    let tools = crate::wire::tool_specs();
    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &tools,
    });

    assert_eq!(body["model"], "qwen3");
    assert_eq!(body["stream"], true);
    assert_eq!(body["messages"][0]["role"], "system");
    assert_eq!(
        body["messages"][1]["tool_calls"][0]["function"]["arguments"],
        "{\"clip\":3}"
    );
    assert_eq!(body["messages"][2]["role"], "tool");
    assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
    assert_eq!(body["tools"].as_array().unwrap().len(), 51);
    assert_eq!(body["tools"][0]["function"]["name"], "add_track");
    assert!(body.get("provider").is_none());
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert!(
        body.get("usage").is_none(),
        "usage.include is OpenRouter-only"
    );
}

#[test]
fn request_body_includes_openrouter_provider_pin() {
    let extras = OpenAiCompatExtras {
        provider_order: Some(vec!["groq".into()]),
        allow_fallbacks: false,
        openrouter_headers: true,
        usage_accounting: true,
    };
    let provider = OpenAiCompatProvider::with_extras(
        "https://openrouter.ai/api/v1",
        "openai/gpt-oss-120b",
        Some("sk-or".into()),
        extras,
    );
    let body = provider.request_body(&ChatRequest {
        messages: &[],
        tools: &[],
    });
    assert_eq!(body["provider"]["order"][0], "groq");
    assert_eq!(body["provider"]["allow_fallbacks"], false);
    assert_eq!(body["stream_options"]["include_usage"], true);
    assert_eq!(body["usage"]["include"], true);
}

#[test]
fn request_body_usage_accounting_only_for_openrouter_extras() {
    let default_body = OpenAiCompatProvider::new("http://localhost:11434/v1", "qwen3", None)
        .request_body(&ChatRequest {
            messages: &[],
            tools: &[],
        });
    assert_eq!(default_body["stream_options"]["include_usage"], true);
    assert!(default_body.get("usage").is_none());

    let openrouter_body = OpenAiCompatProvider::with_extras(
        "https://openrouter.ai/api/v1",
        "openai/gpt-oss-120b",
        Some("sk-or".into()),
        crate::providers::openrouter_compat_extras("openai/gpt-oss-120b"),
    )
    .request_body(&ChatRequest {
        messages: &[],
        tools: &[],
    });
    assert_eq!(openrouter_body["stream_options"]["include_usage"], true);
    assert_eq!(openrouter_body["usage"]["include"], true);
}

#[test]
fn user_message_with_image_becomes_content_parts() {
    let bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a];
    let message = Message::User {
        content: "what's on the timeline?".into(),
        images: vec![ImagePart::png(bytes.clone(), "timeline at 12.40s")],
    };
    let wire = to_openai(&message);
    assert_eq!(wire.len(), 1);
    assert_eq!(wire[0]["role"], "user");
    let parts = wire[0]["content"].as_array().expect("content parts array");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "what's on the timeline?");
    assert_eq!(parts[1]["type"], "image_url");
    let url = parts[1]["image_url"]["url"].as_str().unwrap();
    let b64 = url
        .strip_prefix("data:image/png;base64,")
        .expect("png data URL prefix");
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("valid base64");
    assert_eq!(decoded, bytes, "round-trips the original bytes");
}

#[test]
fn tool_result_images_hoist_into_a_synthetic_user_message() {
    let with_images = Message::ToolResult {
        call_id: "call_1".into(),
        content: "screenshot taken".into(),
        images: vec![ImagePart::jpeg(vec![1, 2, 3], "preview at 3.00s")],
    };
    let wire = to_openai(&with_images);
    assert_eq!(wire.len(), 2);
    assert_eq!(wire[0]["role"], "tool");
    assert_eq!(wire[0]["tool_call_id"], "call_1");
    assert!(
        wire[0]["content"].is_string(),
        "the tool role carries only the text content"
    );
    assert_eq!(wire[0]["content"], "screenshot taken");
    assert_eq!(wire[1]["role"], "user");
    let parts = wire[1]["content"].as_array().expect("content parts array");
    assert_eq!(parts[0]["type"], "text");
    assert!(
        parts[0]["text"]
            .as_str()
            .unwrap()
            .contains("preview at 3.00s"),
        "{}",
        parts[0]["text"]
    );
    assert_eq!(parts[1]["type"], "image_url");
    assert!(
        parts[1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,")
    );

    let without = Message::tool_result("call_2", "removed clip 3");
    assert_eq!(to_openai(&without).len(), 1);
}

#[test]
fn parallel_tool_results_all_precede_hoisted_images() {
    let messages = vec![
        Message::Assistant {
            content: String::new(),
            tool_calls: vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "media_preview_frame".into(),
                    arguments: serde_json::json!({}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "describe_project".into(),
                    arguments: serde_json::json!({}),
                },
            ],
        },
        Message::ToolResult {
            call_id: "call_a".into(),
            content: "frame ready".into(),
            images: vec![ImagePart::png(vec![1, 2], "preview frame")],
        },
        Message::tool_result("call_b", "project summary"),
    ];

    let wire = to_openai_messages(&messages);
    let roles: Vec<_> = wire
        .iter()
        .map(|message| message["role"].as_str().unwrap())
        .collect();
    assert_eq!(roles, ["assistant", "tool", "tool", "user"]);
    assert_eq!(wire[1]["tool_call_id"], "call_a");
    assert_eq!(wire[2]["tool_call_id"], "call_b");
    assert!(
        wire[3]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("call_a: preview frame")
    );
    assert_eq!(wire[3]["content"][1]["type"], "image_url");
}

#[test]
fn user_message_without_images_keeps_plain_string_content() {
    let wire = to_openai(&Message::user("split the clip"));
    assert_eq!(wire.len(), 1);
    assert!(
        wire[0]["content"].is_string(),
        "plain string, not a parts array, so array-less servers keep working"
    );
    assert_eq!(wire[0]["content"], "split the clip");
}

#[test]
fn retry_backoff_is_two_attempts_then_give_up() {
    assert_eq!(retry_delay(0), Some(Duration::from_millis(300)));
    assert_eq!(retry_delay(1), Some(Duration::from_millis(900)));
    assert_eq!(retry_delay(2), None);
    assert_eq!(retry_delay(99), None);
}

#[test]
fn only_transient_http_statuses_retry() {
    for status in [408, 429, 500, 502, 599] {
        assert!(retryable_status(status), "{status}");
    }
    for status in [400, 401, 402, 403, 404, 422, 600] {
        assert!(!retryable_status(status), "{status}");
    }
}

#[test]
fn cancellation_shortcuts_the_retry_sleep() {
    let cancel = AtomicBool::new(true);
    let start = std::time::Instant::now();
    assert!(!sleep_unless_cancelled(Duration::from_millis(900), &cancel));
    assert!(
        start.elapsed() < Duration::from_millis(300),
        "a raised cancel flag must not wait out the backoff"
    );

    let live = AtomicBool::new(false);
    assert!(sleep_unless_cancelled(Duration::from_millis(10), &live));
}

#[test]
fn reasoning_tool_compatibility_errors_point_to_responses_setting() {
    let message = chat_error_message(
        400,
        "reasoning_effort is not supported when function tools are present",
    );
    assert!(
        message.contains("choose the Responses API protocol"),
        "{message}"
    );

    let unrelated = chat_error_message(400, "invalid JSON schema");
    assert!(!unrelated.contains("Responses API protocol"), "{unrelated}");
}
