use super::*;
use crate::wire::ToolSpec;
use std::io::Write as _;
use std::net::{TcpListener, TcpStream};

fn provider(reasoning_summary: bool) -> OpenAiResponsesProvider {
    OpenAiResponsesProvider::new(
        "https://api.example.test/v1/",
        "gpt-reasoning",
        None,
        reasoning_summary,
    )
}

fn run(fixture: &str) -> (ParsedResponse, String, String) {
    let cancel = AtomicBool::new(false);
    let mut text = String::new();
    let mut reasoning = String::new();
    let parsed = consume_responses_sse(fixture.as_bytes(), &cancel, &mut |event| match event {
        ProviderStreamEvent::TextDelta(delta) => text.push_str(delta),
        ProviderStreamEvent::ReasoningSummaryDelta(delta) => reasoning.push_str(delta),
    })
    .expect("fixture parses");
    (parsed, text, reasoning)
}

fn read_http_request(stream: &mut TcpStream) -> String {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 2048];
    let mut expected = None;
    loop {
        let count = stream.read(&mut buffer).expect("read HTTP request");
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if expected.is_none()
            && let Some(header_end) = bytes.windows(4).position(|window| window == b"\r\n\r\n")
        {
            let headers = String::from_utf8_lossy(&bytes[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.to_ascii_lowercase()
                        .strip_prefix("content-length:")
                        .and_then(|value| value.trim().parse::<usize>().ok())
                })
                .unwrap_or(0);
            expected = Some(header_end + 4 + content_length);
        }
        if expected.is_some_and(|expected| bytes.len() >= expected) {
            break;
        }
    }
    String::from_utf8(bytes).expect("ASCII HTTP request")
}

#[test]
fn request_maps_instructions_multimodal_input_and_native_tools() {
    let provider = provider(true);
    assert_eq!(provider.base_url, "https://api.example.test/v1");
    let messages = vec![
        Message::system("Fresh project snapshot."),
        Message::User {
            content: "Inspect this frame.".into(),
            images: vec![ImagePart::png(vec![1, 2, 3], "preview")],
        },
    ];
    let tools = vec![ToolSpec {
        name: "trim_clip".into(),
        description: "Trim a clip".into(),
        parameters: serde_json::json!({"type": "object"}),
    }];
    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &tools,
        session_id: None,
    });

    assert_eq!(body["model"], "gpt-reasoning");
    assert_eq!(body["stream"], true);
    assert_eq!(body["store"], false);
    assert_eq!(body["instructions"], "Fresh project snapshot.");
    assert_eq!(body["include"][0], "reasoning.encrypted_content");
    assert_eq!(body["reasoning"]["summary"], "auto");
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][1]["type"], "input_image");
    assert!(
        body["input"][0]["content"][1]["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "trim_clip");
    assert_eq!(body["tools"][0]["strict"], false);
    assert!(body["tools"][0].get("function").is_none());
}

#[test]
fn mixed_history_uses_role_specific_content_parts_and_summary_off() {
    let provider = provider(false);
    let messages = [
        Message::system("system"),
        Message::user("Prior question"),
        Message::assistant_text("Prior answer"),
        Message::user("Current question"),
    ];
    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &[],
        session_id: None,
    });
    assert!(body.get("reasoning").is_none());
    assert!(body.get("tools").is_none());
    assert_eq!(body["store"], false);
    assert_eq!(body["input"][0]["role"], "user");
    assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][0]["content"][0]["text"], "Prior question");
    assert_eq!(body["input"][1]["role"], "assistant");
    assert_eq!(body["input"][1]["content"][0]["type"], "output_text");
    assert_eq!(body["input"][1]["content"][0]["text"], "Prior answer");
    assert_eq!(body["input"][2]["role"], "user");
    assert_eq!(body["input"][2]["content"][0]["type"], "input_text");
    assert_eq!(body["input"][2]["content"][0]["text"], "Current question");
}

#[test]
fn encrypted_reasoning_and_parallel_calls_replay_with_native_outputs() {
    let provider = provider(true);
    let output = vec![
        serde_json::json!({
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "encrypted-private-state",
            "summary": [{"type": "summary_text", "text": "I inspected both clips."}],
        }),
        serde_json::json!({
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_a",
            "name": "media_asset_strip",
            "arguments": "{\"media_id\":1}",
        }),
        serde_json::json!({
            "type": "function_call",
            "id": "fc_2",
            "call_id": "call_b",
            "name": "describe_project",
            "arguments": "{}",
        }),
    ];
    let turn = ChatTurn {
        text: String::new(),
        reasoning_summary: "I inspected both clips.".into(),
        tool_calls: vec![
            ToolCall {
                id: "call_a".into(),
                name: "media_asset_strip".into(),
                arguments: serde_json::json!({"media_id": 1}),
            },
            ToolCall {
                id: "call_b".into(),
                name: "describe_project".into(),
                arguments: serde_json::json!({}),
            },
        ],
        finish: FinishReason::ToolCalls,
        usage: None,
    };
    let base_messages = vec![Message::system("system"), Message::user("make a montage")];
    provider.update_replay(&base_messages, &turn, output);

    let mut messages = base_messages;
    messages.extend([
        Message::Assistant {
            content: String::new(),
            tool_calls: turn.tool_calls.clone(),
        },
        Message::ToolResult {
            call_id: "call_a".into(),
            content: "strip ready".into(),
            images: vec![ImagePart::jpeg(vec![9, 8], "source strip")],
        },
        Message::tool_result("call_b", "project state"),
    ]);
    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &[],
        session_id: None,
    });
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 6);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["type"], "reasoning");
    assert_eq!(input[1]["encrypted_content"], "encrypted-private-state");
    assert_eq!(input[2]["call_id"], "call_a");
    assert_eq!(input[3]["call_id"], "call_b");
    assert_eq!(input[4]["type"], "function_call_output");
    assert_eq!(input[4]["call_id"], "call_a");
    assert_eq!(input[4]["output"][1]["type"], "input_image");
    assert_eq!(input[5]["call_id"], "call_b");
}

#[test]
fn unrelated_request_does_not_replay_stale_output() {
    let provider = provider(true);
    let turn = ChatTurn {
        text: String::new(),
        reasoning_summary: String::new(),
        tool_calls: vec![ToolCall {
            id: "old_call".into(),
            name: "describe_project".into(),
            arguments: serde_json::json!({}),
        }],
        finish: FinishReason::ToolCalls,
        usage: None,
    };
    provider.update_replay(
        &[Message::system("old"), Message::user("old prompt")],
        &turn,
        vec![serde_json::json!({
            "type": "reasoning",
            "encrypted_content": "must-not-leak",
        })],
    );
    let messages = [Message::system("new"), Message::user("new prompt")];
    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &[],
        session_id: None,
    });
    assert_eq!(body["input"].as_array().unwrap().len(), 1);
    assert!(!body.to_string().contains("must-not-leak"));
}

#[test]
fn consecutive_tool_rounds_preserve_every_exact_item_since_the_user() {
    let provider = provider(true);
    let mut messages = vec![Message::system("system"), Message::user("edit this")];
    let first_turn = ChatTurn {
        text: String::new(),
        reasoning_summary: "first".into(),
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            name: "describe_project".into(),
            arguments: serde_json::json!({}),
        }],
        finish: FinishReason::ToolCalls,
        usage: None,
    };
    provider.update_replay(
        &messages,
        &first_turn,
        vec![
            serde_json::json!({
                "type": "reasoning",
                "encrypted_content": "encrypted-round-1",
                "summary": [],
            }),
            serde_json::json!({
                "type": "function_call",
                "call_id": "call_1",
                "name": "describe_project",
                "arguments": "{}",
            }),
        ],
    );
    messages.push(Message::Assistant {
        content: String::new(),
        tool_calls: first_turn.tool_calls,
    });
    messages.push(Message::tool_result("call_1", "round one result"));

    let second_turn = ChatTurn {
        text: String::new(),
        reasoning_summary: "second".into(),
        tool_calls: vec![ToolCall {
            id: "call_2".into(),
            name: "remove_clip".into(),
            arguments: serde_json::json!({"clip": 7}),
        }],
        finish: FinishReason::ToolCalls,
        usage: None,
    };
    provider.update_replay(
        &messages,
        &second_turn,
        vec![
            serde_json::json!({
                "type": "reasoning",
                "encrypted_content": "encrypted-round-2",
                "summary": [],
            }),
            serde_json::json!({
                "type": "function_call",
                "call_id": "call_2",
                "name": "remove_clip",
                "arguments": "{\"clip\":7}",
            }),
        ],
    );
    messages.push(Message::Assistant {
        content: String::new(),
        tool_calls: second_turn.tool_calls,
    });
    messages.push(Message::tool_result("call_2", "round two result"));

    let body = provider.request_body(&ChatRequest {
        messages: &messages,
        tools: &[],
        session_id: None,
    });
    let input = body["input"].as_array().unwrap();
    assert_eq!(input.len(), 7);
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["encrypted_content"], "encrypted-round-1");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[3]["call_id"], "call_1");
    assert_eq!(input[3]["output"], "round one result");
    assert_eq!(input[4]["encrypted_content"], "encrypted-round-2");
    assert_eq!(input[5]["call_id"], "call_2");
    assert_eq!(input[6]["call_id"], "call_2");
    assert_eq!(input[6]["output"], "round two result");
}

#[test]
fn text_and_reasoning_summary_streams_stay_separate() {
    let fixture = concat!(
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"I checked \"}\n",
        "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"the cuts.\"}\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Done\"}\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\".\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[",
        "{\"type\":\"reasoning\",\"encrypted_content\":\"enc\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"I checked the cuts.\"}]},",
        "{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Done.\"}]}",
        "]}}\n",
    );
    let (parsed, streamed, reasoning) = run(fixture);
    assert_eq!(streamed, "Done.");
    assert_eq!(reasoning, "I checked the cuts.");
    assert_eq!(parsed.turn.text, "Done.");
    assert_eq!(parsed.turn.reasoning_summary, "I checked the cuts.");
    assert_eq!(parsed.turn.finish, FinishReason::Stop);
    assert!(parsed.turn.tool_calls.is_empty());
    assert!(parsed.turn.usage.is_none());
    assert_eq!(parsed.output[0]["encrypted_content"], "enc");
}

#[test]
fn completed_event_parses_token_usage() {
    let fixture = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{",
        "\"usage\":{\"input_tokens\":200,\"output_tokens\":12,\"input_tokens_details\":{\"cached_tokens\":50}},",
        "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]",
        "}}\n",
    );
    let (parsed, _, _) = run(fixture);
    assert_eq!(
        parsed.turn.usage,
        Some(crate::provider::TokenUsage {
            input_tokens: 200,
            cached_input_tokens: 50,
            output_tokens: 12,
            cost: None,
        })
    );
}

#[test]
fn incomplete_event_parses_token_usage() {
    let fixture = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{",
        "\"incomplete_details\":{\"reason\":\"max_output_tokens\"},",
        "\"usage\":{\"input_tokens\":200,\"output_tokens\":64,\"input_tokens_details\":{\"cached_tokens\":50}},",
        "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"partial\"}]}]",
        "}}\n",
    );
    let (parsed, _, _) = run(fixture);
    assert_eq!(parsed.turn.finish, FinishReason::Length);
    assert_eq!(parsed.turn.text, "partial");
    assert_eq!(
        parsed.turn.usage,
        Some(crate::provider::TokenUsage {
            input_tokens: 200,
            cached_input_tokens: 50,
            output_tokens: 64,
            cost: None,
        })
    );
}

#[test]
fn usage_parses_float_encoded_token_counts() {
    let fixture = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{",
        "\"usage\":{\"input_tokens\":200.0,\"output_tokens\":12.4,\"input_tokens_details\":{\"cached_tokens\":50.6}},",
        "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]",
        "}}\n",
    );
    let (parsed, _, _) = run(fixture);
    assert_eq!(
        parsed.turn.usage,
        Some(crate::provider::TokenUsage {
            input_tokens: 200,
            cached_input_tokens: 51,
            output_tokens: 12,
            cost: None,
        })
    );
}

#[test]
fn completed_output_parses_parallel_function_calls() {
    let fixture = concat!(
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"clip\\\":\"}\n",
        "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"clip\\\":1}\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[",
        "{\"type\":\"function_call\",\"id\":\"fc_a\",\"call_id\":\"call_a\",\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":1}\"},",
        "{\"type\":\"function_call\",\"id\":\"fc_b\",\"call_id\":\"call_b\",\"name\":\"remove_clip\",\"arguments\":\"{\\\"clip\\\":2}\"}",
        "]}}\n",
    );
    let (parsed, streamed, reasoning) = run(fixture);
    assert!(streamed.is_empty());
    assert!(reasoning.is_empty());
    assert_eq!(parsed.turn.finish, FinishReason::ToolCalls);
    assert_eq!(parsed.turn.tool_calls.len(), 2);
    assert_eq!(parsed.turn.tool_calls[0].id, "call_a");
    assert_eq!(parsed.turn.tool_calls[0].arguments["clip"], 1);
    assert_eq!(parsed.turn.tool_calls[1].id, "call_b");
    assert_eq!(parsed.turn.tool_calls[1].arguments["clip"], 2);
}

#[test]
fn output_item_events_are_a_terminal_output_fallback() {
    let fixture = concat!(
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"remove_clip\",\"arguments\":\"\"}}\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"clip\\\":3}\"}\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n",
    );
    let (parsed, _, _) = run(fixture);
    assert_eq!(parsed.turn.tool_calls.len(), 1);
    assert_eq!(parsed.turn.tool_calls[0].arguments["clip"], 3);
}

#[test]
fn terminal_display_events_are_forwarded_when_deltas_are_absent() {
    let fixture = concat!(
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[",
        "{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Checked the result.\"}]},",
        "{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Complete answer\"}]}",
        "]}}\n",
    );
    let (parsed, streamed, reasoning) = run(fixture);
    assert_eq!(streamed, "Complete answer");
    assert_eq!(reasoning, "Checked the result.");
    assert_eq!(parsed.turn.text, streamed);
    assert_eq!(parsed.turn.reasoning_summary, reasoning);
}

#[test]
fn malformed_events_arguments_and_missing_terminal_are_errors() {
    let cancel = AtomicBool::new(false);
    let malformed =
        consume_responses_sse("data: {not json}\n".as_bytes(), &cancel, &mut |_| {}).unwrap_err();
    assert!(matches!(malformed, ProviderError::Protocol(_)));

    let bad_arguments = concat!(
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[",
        "{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"trim_clip\",\"arguments\":\"{oops\"}",
        "]}}\n",
    );
    let error = consume_responses_sse(bad_arguments.as_bytes(), &cancel, &mut |_| {}).unwrap_err();
    match error {
        ProviderError::Protocol(message) => assert!(message.contains("trim_clip"), "{message}"),
        other => panic!("expected protocol error, got {other:?}"),
    }

    let no_terminal = consume_responses_sse(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n".as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    match no_terminal {
        ProviderError::Protocol(message) => {
            assert!(message.contains("terminal event"), "{message}");
        }
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[test]
fn cancellation_and_incomplete_responses_are_distinct() {
    let cancel = AtomicBool::new(true);
    let cancelled = consume_responses_sse(
        "data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n".as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    assert!(matches!(cancelled, ProviderError::Cancelled));

    let fixture = concat!(
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Partial\"}\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[]}}\n",
    );
    let (parsed, streamed, reasoning) = run(fixture);
    assert_eq!(streamed, "Partial");
    assert!(reasoning.is_empty());
    assert_eq!(parsed.turn.finish, FinishReason::Length);
    assert!(parsed.turn.tool_calls.is_empty());
}

#[test]
fn failed_and_unknown_incomplete_responses_report_provider_details() {
    let cancel = AtomicBool::new(false);
    let stream_error = consume_responses_sse(
        "data: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"stream broke\",\"param\":null}\n"
            .as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    match stream_error {
        ProviderError::Protocol(message) => {
            assert!(message.contains("stream broke"), "{message}");
        }
        other => panic!("expected protocol error, got {other:?}"),
    }

    let failed = consume_responses_sse(
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"model unavailable\"}}}\n"
            .as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    match failed {
        ProviderError::Protocol(message) => {
            assert!(message.contains("model unavailable"), "{message}");
        }
        other => panic!("expected protocol error, got {other:?}"),
    }

    let incomplete = consume_responses_sse(
        "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"content_filter\"},\"output\":[]}}\n"
            .as_bytes(),
        &cancel,
        &mut |_| {},
    )
    .unwrap_err();
    match incomplete {
        ProviderError::Protocol(message) => {
            assert!(message.contains("content_filter"), "{message}");
        }
        other => panic!("expected protocol error, got {other:?}"),
    }
}

#[test]
fn initial_transient_status_retries_but_preflight_cancellation_does_not_connect() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for attempt in 0..2 {
            let (mut stream, _) = listener.accept().expect("provider connects");
            requests.push(read_http_request(&mut stream));
            let (status, body) = if attempt == 0 {
                ("500 Internal Server Error", "temporary")
            } else {
                (
                    "200 OK",
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\
                     data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n",
                )
            };
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\n\
                 Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
            .unwrap();
            stream.flush().unwrap();
        }
        requests
    });

    let provider = OpenAiResponsesProvider::new(
        &format!("http://{address}/v1"),
        "reasoning-model",
        None,
        true,
    );
    let messages = [Message::system("system"), Message::user("hello")];
    let request = ChatRequest {
        messages: &messages,
        tools: &[],
        session_id: None,
    };
    let cancel = AtomicBool::new(false);
    let mut streamed = String::new();
    let turn = provider
        .chat(&request, &cancel, &mut |event| match event {
            ProviderStreamEvent::TextDelta(delta) => streamed.push_str(delta),
            ProviderStreamEvent::ReasoningSummaryDelta(_) => {}
        })
        .expect("second initial request succeeds");
    assert_eq!(turn.text, "ok");
    assert_eq!(streamed, "ok");
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.starts_with("POST /v1/responses HTTP/1.1")),
        "{requests:#?}"
    );

    let cancelled = AtomicBool::new(true);
    let error = provider
        .chat(&request, &cancelled, &mut |_| {})
        .unwrap_err();
    assert!(matches!(error, ProviderError::Cancelled));
}
