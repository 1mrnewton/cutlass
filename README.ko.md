# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · **한국어** · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass는 타임라인 동영상 편집기입니다. 데스크톱, iOS, Android가 하나의 드래프트와 하나의 세션을 공유합니다. 어느 셸의 제스처도 같은 요청이 됩니다. 같은 드래프트와 같은 시각에서 재생과 내보내기는 같은 프레임을 봅니다.

문서의 시간은 부호 있는 마이크로초입니다. 세그먼트는 그 단위로 두 범위를 저장합니다. 소스 범위는 사용하는 미디어 구간입니다. 타깃 범위는 타임라인 위의 위치입니다.

이 번역과 트리가 어긋나면 [영어 README](README.md)를 기준으로 합니다.

## 현재

워크스페이스는 초기입니다.

| 부분 | 상태 |
| --- | --- |
| `cutlass-model` | 드래프트. 트랙, 세그먼트, 머티리얼 풀, 마이크로초 시간 |
| `cutlass-session` | 배치, 이동, 트리밍을 하나의 실행 취소 단계로 적용합니다. 실패한 요청은 드래프트를 그대로 둡니다. 실행 취소와 다시 실행은 리비전을 올립니다 |
| 나머지 크레이트 | 빈 멤버입니다. 엔진, 컨테이너, 코덱, 재생, 내보내기, 에이전트의 자리를 잡아 둡니다 |
| `apps/desktop` | `cutlass`를 출력하고 돌아갑니다. 창은 이후입니다 |
| `apps/ios`, `apps/android` | 이 트리에는 이후에 들어옵니다 |

컨테이너와 코덱은 네이티브 Rust이며, 계열 또는 코덱마다 크레이트 하나입니다. Apple, Windows, Android에서는 플랫폼 미디어 프레임워크가 빠른 경로입니다. Linux는 Rust 코덱과 소프트웨어 미리보기를 사용합니다.

## 구성

```text
apps/desktop/          egui와 eframe
crates/                구성 요소마다 크레이트 하나
crates/containers/     컨테이너 계열마다 크레이트 하나
crates/codecs/         코덱마다 크레이트 하나
```

크레이트 이름은 `cutlass-` 접두사를 씁니다. 데스크톱 바이너리는 크레이트를 직접 링크합니다. 전화 셸이 링크하는 라이브러리는 `cutlass-ffi`뿐입니다.

## 빌드

stable Rust를 설치하세요. `rust-toolchain.toml`이 채널과 Clippy, rustfmt를 고릅니다.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint`는 워크스페이스에서 Clippy를 실행하고 경고를 거부합니다. Ubuntu CI는 이 검사와 `cargo deny`를 실행합니다.

## 라이선스

Cutlass는 다음 중 하나를 선택해 사용할 수 있습니다.

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) 또는 <https://www.apache.org/licenses/LICENSE-2.0>)
* MIT 라이선스 ([LICENSE-MIT](LICENSE-MIT) 또는 <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
