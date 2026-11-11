<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>免費、開源的影片剪輯軟體，還有一個可以對話的助理。</b><br>
  支援 macOS、Windows 和 Linux。
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">下載</a> ·
  <a href="#what-you-can-do">功能</a> ·
  <a href="#the-assistant">助理</a> ·
  <a href="#formats">格式</a> ·
  <a href="#build-from-source">建置</a> ·
  <a href="#not-here-yet">尚未完成</a> ·
  <a href="https://cutlass.sh">網站</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · **繁體中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass 是一款時間軸影片剪輯軟體。匯入影片、聲音和圖片，在軌道上剪輯，加上文字、特效和調色，然後匯出成品。每一步都可以自己動手，也可以用平常的話描述想要的剪輯，讓助理在時間軸上完成，之後你可以修改或還原。

它是免費的。不需要帳號，沒有浮水印，也不上傳任何內容。素材留在你的硬碟上，用到機器學習的工具都在你自己的電腦上執行。

在剪輯軟體之下，編解碼器是我們自己寫的。H.264、HEVC、AV1、ProRes、AAC、Opus 以及其餘的編解碼器都在這個儲存庫裡用 Rust 寫成。Cutlass 不內含 FFmpeg。

<a name="download"></a>
## 下載

| 系統 | 檔案 |
| --- | --- |
| macOS，Apple 晶片 | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows，64 位元 | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux，x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

在[發行版本](https://github.com/1mrnewton/cutlass/releases)頁面上，每個檔案旁邊都有一個 `.sha256` 檢查碼。安裝之後，Cutlass 會自己檢查新版本：選單裡的 **Check for updates…**。

這些版本還沒有用開發者憑證簽署，所以第一次啟動要多一個步驟：

- **macOS:** 先打開一次應用程式，然後在 System Settings › Privacy & Security 裡按 **Open Anyway** 允許它。
- **Windows:** 如果 SmartScreen 擋住了安裝程式，選擇 **More info** › **Run anyway**。
- **Linux:** 用 `chmod +x Cutlass-x86_64.AppImage` 讓檔案可以執行，然後執行它。

<a name="what-you-can-do"></a>
## 你可以做什麼

**在時間軸上剪輯。** 影片、音訊和文字軌道。分割、修剪、合攏空隙、定格、倒轉、鏡像、旋轉和裁切。把片段編成群組，或者摺疊成複合片段，之後再打開。取代一個片段並保留它的編輯，或者把一個片段的屬性複製到另一個上。大型素材可以透過代理檔來剪。

**速度和動畫。** 設定固定速度，或者選一條速度曲線，拖曳上面的點來調整形狀。用關鍵影格讓片段動起來，可以一次移動或刪除多個關鍵影格。進場、出場和組合動畫可以直接套用。

**調色。** 基本調整、HSL、曲線和色輪。載入 `.cube` LUT，並讓膚色在它之下保持自然。200 個濾鏡，以及時間軸上的調整圖層。

**特效和轉場。** 200 個特效、180 個轉場和 13 種混合模式。用線性、鏡像、圓形、矩形、心形或星形遮罩遮住片段。去掉綠幕，或者點一下物體把它去背。

**文字、字幕和貼圖。** 文字可以設定字體、外框、陰影、背景、對話框和特效，並帶進場、出場或循環動畫。字幕是在你的電腦上從片段裡聽出來的（目前只支援英語），之後以清單形式編輯：重打一行、尋找和取代、選一個範本、為單一個詞設定樣式，再把結果匯出為 SRT 檔。貼圖和形狀像其他片段一樣放在時間軸上。

**在你電腦上執行的畫面工具。** 防手震、提升畫質、畫面降噪、用於流暢慢動作的光流、重新打光、自動重新構圖和動態模糊。什麼都不會傳到別處。

**聲音。** 音量和淡入淡出，從影片中擷取音訊，依聲音同步影片和音訊，以及標記節拍。

**匯出。** MP4 或 MOV，可以匯出整條時間軸、一個範圍或選取的片段。也可以只匯出聲音、GIF 或動態 WebP。在 macOS 和 Windows 上，系統的硬體編解碼器能用時就由它來做，設定裡有一個開關可以關掉。

<a name="the-assistant"></a>
## 助理

說出你想要的，助理就用時間軸送出的同一種請求來編輯打開的專案。它做的每一次修改都是一步還原。

- **它在你的電腦上執行。** 助理和你自己電腦上的模型執行器對話，例如 [Ollama](https://ollama.com)、llama.cpp 伺服器或 LM Studio。在 Settings › Assistant 裡設定位址和模型。走這條路時，提示詞不會離開你的電腦。
- **它會先問你。** 任何會把資料送出這台電腦的事，都要等你允許。
- **它有技能。** 字幕、移除停頓、動態圖形、產品示範和專案設定隨應用程式一起提供，它還能呼叫你加入的 MCP 伺服器裡的工具。
- **其他代理程式也能操作 Cutlass。** `cutlass mcp /path/to/project` 不開視窗，透過 stdio 提供同樣的工具，所以 Claude 或任何 MCP 用戶端都能讀取和編輯專案。

<a name="formats"></a>
## 格式

下面的解碼器和編碼器是 [`crates/codecs`](../../crates/codecs) 和 [`crates/containers`](../../crates/containers) 裡的 Rust 程式碼，以 FFmpeg 作為外部參照來測試。每個 crate 的 README 都寫明它支援什麼，以及它的速度和 FFmpeg 相比如何。

| | 開啟 | 匯出 |
| --- | :---: | :---: |
| **檔案** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4、MOV，以及單獨的聲音 |
| **影片** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG。HEVC 在 macOS 和 Windows 上透過系統編碼器匯出 |
| **聲音** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **圖片** | JPEG, PNG, WebP, GIF，動態或靜態 | GIF、動態 WebP |

<a name="build-from-source"></a>
## 從原始碼建置

你需要穩定版 [Rust](https://rustup.rs) 工具鏈和一個 C 編譯器。在 Linux 上還需要 GTK 和 ALSA 的標頭檔：

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

然後：

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

CI 執行的檢查（最後一項需要 [cargo-nextest](https://nexte.st)）：

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## 尚未完成

Cutlass 還很年輕。在應用程式裡，畫出來但還沒做好的功能會變暗並標上 **Later**，所以你總能知道哪些能用。目前比較大的缺口：

- **更多容器。** MKV、WebM、AVI 和 MPEG-TS 檔還打不開。
- **人聲工具。** 降噪、人聲增強、響度正規化、人聲分離和文字轉語音。
- **生成式工具。** AI 移除、擴展和風格化。
- **追蹤。** 跟隨主體的文字、貼圖和遮罩。
- **更多語言的字幕。**
- **給助理用的託管模型。** 目前它使用本機模型。
- **手機。** iOS 和 Android 應用程式在 [`apps`](../../apps) 裡，共用同一個 Rust 核心，但桌面版優先。

附上出問題的檔案的錯誤回報，是你能寄給我們最有用的東西。[提交 issue](https://github.com/1mrnewton/cutlass/issues)。

## 程式碼結構

| 資料夾 | 裡面有什麼 |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | 桌面剪輯軟體，一個 egui 視窗。它建置出 `cutlass` 執行檔 |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | 手機應用程式，用 SwiftUI 和 Kotlin Compose 撰寫。它們透過 `crates/ffi` 使用 Rust 核心 |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | 專案、可還原的編輯請求，以及所有應用程式共用的剪輯行為 |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | 把一個專案和一個時間點變成一格畫面和它的聲音，用於預覽和匯出 |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | 每種編解碼器和每種檔案格式各一個 crate |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | 系統的硬體編解碼器 |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | 濾鏡、特效、轉場、遮罩、調色、文字和字幕 |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | 在裝置上執行的畫面模型和語音模型 |
| [`crates/agent`](../../crates/agent) | 助理：它的工具、技能和 MCP |

## 參與貢獻

用真實素材來使用 Cutlass，告訴我們哪裡出了問題。[issue 表單](https://github.com/1mrnewton/cutlass/issues/new/choose)涵蓋錯誤、功能、效能，以及打不開的檔案。

要提交程式碼？請先執行上面的三項檢查。共用的 crate 不依賴任何單一平台，所以同一個編輯在桌面和手機上都能用。

<a name="license"></a>
## 授權

Cutlass 可以在兩種授權中任選一種免費使用。請選一種。

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

