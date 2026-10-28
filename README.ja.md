# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · **日本語** · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass はタイムライン動画エディタです。デスクトップ、iOS、Android は一つのドラフトと一つのセッションを共有します。どのシェルの操作も同じリクエストになります。同じドラフト、同じ時刻では、再生と書き出しは同じフレームを見ます。

文書の時刻は符号付きのマイクロ秒です。セグメントはその単位で二つの範囲を持ちます。ソース範囲は使うメディアの部分です。ターゲット範囲はタイムライン上の位置です。

この訳とツリーの内容が食い違うときは、[英語の README](README.md) を基準にしてください。

## 現状

ワークスペースは初期段階です。

| 部分 | 状態 |
| --- | --- |
| `cutlass-model` | ドラフト。トラック、セグメント、マテリアル、マイクロ秒 |
| `cutlass-session` | 配置、移動、トリムを一つのアンドゥとして適用します。失敗したリクエストはドラフトをそのまま残します。アンドゥとリドゥはリビジョンを進めます |
| その他のクレート | 空のメンバーです。エンジン、コンテナ、コーデック、再生、書き出し、エージェントの場所を確保しています |
| `apps/desktop` | `cutlass` を表示して戻ります。ウィンドウは後からです |
| `apps/ios`、`apps/android` | このツリーには後から入ります |

コンテナとコーデックはネイティブな Rust で、ファミリーまたはコーデックごとに一つのクレートです。Apple、Windows、Android ではプラットフォームのメディアフレームワークが速い経路です。Linux は Rust のコーデックとソフトウェアプレビューを使います。

## 構成

```text
apps/desktop/          egui と eframe
crates/                コンポーネントごとに一つのクレート
crates/containers/     コンテナファミリーごとに一つのクレート
crates/codecs/         コーデックごとに一つのクレート
```

クレート名は `cutlass-` で始まります。デスクトップのバイナリはクレートを直接リンクします。電話のシェルがリンクするのは `cutlass-ffi` だけです。

## ビルド

stable の Rust を入れてください。`rust-toolchain.toml` がチャンネル、Clippy、rustfmt を選びます。

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` はワークスペース全体で Clippy を実行し、警告を拒否します。Ubuntu の CI はこれらの検査と `cargo deny` を実行します。

## ライセンス

Cutlass は次のいずれかのライセンスを選んで利用できます。

* Apache License, Version 2.0（[LICENSE-APACHE](LICENSE-APACHE) または <https://www.apache.org/licenses/LICENSE-2.0>）
* MIT ライセンス（[LICENSE-MIT](LICENSE-MIT) または <https://opensource.org/licenses/MIT>）

Copyright (c) 2026 Emmanuel Aidoo
