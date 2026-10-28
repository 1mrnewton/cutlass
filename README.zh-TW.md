# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · **繁體中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass 是一套時間軸影片編輯器。桌面、iOS 與 Android 共用同一份草稿與同一個工作階段。任一介面上的手勢都會變成同一項請求。同一份草稿、同一個時刻，播放與匯出看到的是同一幀。

文件中的時間是帶正負號的微秒計數。片段以這個單位保存兩個範圍。來源範圍是它使用的媒體部分。目標範圍是它在時間軸上的位置。

譯文與這份程式樹不一致時，以[英文 README](README.md) 為準。

## 現況

工作區仍在早期。

| 部分 | 狀態 |
| --- | --- |
| `cutlass-model` | 草稿：軌道、片段、素材池，以及微秒時間 |
| `cutlass-session` | 放置、移動與修剪作為一步復原。失敗的請求讓草稿保持原狀。復原與重做都會推進修訂號 |
| 其餘 crate | 空成員。它們為引擎、容器、編解碼器、播放、匯出與 agent 保留位置 |
| `apps/desktop` | 印出 `cutlass` 後返回。視窗稍後才有 |
| `apps/ios`、`apps/android` | 稍後進入這個目錄樹 |

容器與編解碼器是原生 Rust，每個系列或每種編解碼器一個 crate。在 Apple、Windows 與 Android 上，平台媒體框架是快速路徑。Linux 使用 Rust 編解碼器與軟體預覽。

## 配置

```text
apps/desktop/          egui 與 eframe
crates/                每個元件一個 crate
crates/containers/     每個容器系列一個 crate
crates/codecs/         每種編解碼器一個 crate
```

crate 名稱使用 `cutlass-` 前綴。桌面二進位直接連結這些 crate。手機介面只連結 `cutlass-ffi`。

## 建置

安裝 stable Rust。`rust-toolchain.toml` 選定通道，以及 Clippy 與 rustfmt。

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` 在整個工作區執行 Clippy，並把警告視為錯誤。Ubuntu 上的 CI 會執行這些檢查以及 `cargo deny`。

## 授權

Cutlass 可依你的選擇，使用下列任一授權：

* Apache License 2.0（[LICENSE-APACHE](LICENSE-APACHE) 或 <https://www.apache.org/licenses/LICENSE-2.0>）
* MIT 授權（[LICENSE-MIT](LICENSE-MIT) 或 <https://opensource.org/licenses/MIT>）

Copyright (c) 2026 Emmanuel Aidoo
