<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>話しかけられるアシスタントが付いた、無料でオープンソースの動画エディタ。</b><br>
  macOS、Windows、Linux に対応。
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">ダウンロード</a> ·
  <a href="#what-you-can-do">機能</a> ·
  <a href="#the-assistant">アシスタント</a> ·
  <a href="#formats">フォーマット</a> ·
  <a href="#build-from-source">ビルド</a> ·
  <a href="#not-here-yet">まだないもの</a> ·
  <a href="https://cutlass.sh">ウェブサイト</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · **日本語** · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass はタイムライン型の動画エディタです。映像、音声、画像を読み込み、トラックの上でカットし、テキスト、エフェクト、色を加えて、完成した動画を書き出します。すべての作業を自分の手で行うことも、やりたい編集を普通の言葉で伝えてアシスタントにタイムライン上で行わせることもできます。その結果は変更も取り消しもできます。

無料です。アカウントも、ウォーターマークも、アップロードもありません。素材はあなたのディスクに残り、機械学習を使うツールはあなたのコンピュータの上で動きます。

エディタの下にあるコーデックは自前のものです。H.264、HEVC、AV1、ProRes、AAC、Opus などは、このリポジトリの中で Rust で書かれています。Cutlass は FFmpeg を同梱していません。

<a name="download"></a>
## ダウンロード

| システム | ファイル |
| --- | --- |
| macOS、Apple シリコン | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows、64 ビット | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux、x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

[リリース](https://github.com/1mrnewton/cutlass/releases)ページでは、どのファイルにも `.sha256` チェックサムが添えられています。インストール後は、Cutlass が自分で新しいリリースを確認します。メニューの **Check for updates…** です。

ビルドはまだ開発者証明書で署名されていないため、最初の起動には手順がひとつ増えます。

- **macOS:** アプリを一度開き、System Settings › Privacy & Security の **Open Anyway** で許可します。
- **Windows:** SmartScreen がインストーラを止めたら、**More info** › **Run anyway** を選びます。
- **Linux:** `chmod +x Cutlass-x86_64.AppImage` でファイルを実行可能にしてから、実行します。

<a name="what-you-can-do"></a>
## できること

**タイムラインでカットする。** ビデオ、オーディオ、テキストのトラック。分割、トリム、ギャップを詰める、フリーズフレーム、逆再生、ミラー、回転、クロップ。クリップをグループ化したり、複合クリップにまとめて後で開いたりできます。編集内容を保ったままクリップを置き換えたり、あるクリップの属性を別のクリップにコピーしたりできます。重い素材はプロキシを通してカットできます。

**速度と動き。** 一定の速度を設定するか、速度カーブを選び、点をドラッグして形を変えます。キーフレームでクリップを動かし、複数のキーフレームをまとめて移動したり削除したりできます。イン、アウト、コンボのアニメーションをすぐに適用できます。

**カラー。** 基本の調整、HSL、カーブ、カラーホイール。`.cube` の LUT を読み込み、その下でも肌の色を自然に保てます。200 種類のフィルターと、タイムライン上の調整レイヤー。

**エフェクトとトランジション。** 200 種類のエフェクト、180 種類のトランジション、13 種類のブレンドモード。クリップを線、ミラー、円、長方形、ハート、星の形でマスクできます。グリーンバックを抜いたり、被写体をクリックして切り抜いたりできます。

**テキスト、字幕、ステッカー。** テキストにはフォント、縁取り、影、背景、吹き出し、エフェクトを設定でき、イン、アウト、ループのアニメーションを付けられます。字幕はあなたのマシンの上でクリップの音声から聞き取られ（今のところ英語のみ）、その後リストとして編集します。行を打ち直す、検索して置換する、テンプレートを選ぶ、単語ごとにスタイルを付ける、そして結果を SRT ファイルとして書き出す、といったことができます。ステッカーと図形は、他のクリップと同じようにタイムラインに置かれます。

**あなたのコンピュータで動く映像ツール。** 手ぶれ補正、画質の向上、映像のノイズ低減、なめらかなスローモーションのためのオプティカルフロー、リライト、自動リフレーム、モーションブラー。どこにも送信されません。

**サウンド。** 音量とフェード、動画からの音声の抽出、音をもとにした映像と音声の同期、ビートのマーク付け。

**書き出し。** MP4 または MOV で、タイムライン全体、範囲、選択したクリップを書き出せます。音声だけ、GIF、アニメーション WebP も書き出せます。macOS と Windows では、使えるときはシステムのハードウェアコーデックが処理し、設定のスイッチでオフにできます。

<a name="the-assistant"></a>
## アシスタント

やりたいことを伝えると、アシスタントはタイムラインが送るのと同じリクエストで、開いているプロジェクトを編集します。アシスタントが行う変更は、それぞれ 1 回の取り消しで戻せます。

- **あなたのマシンで動きます。** アシスタントは、[Ollama](https://ollama.com)、llama.cpp のサーバー、LM Studio など、あなたのコンピュータ上のモデルランナーと話します。アドレスとモデルは Settings › Assistant で設定します。この経路では、プロンプトがあなたのコンピュータの外に出ることはありません。
- **先に確認します。** データをマシンの外に送ることになる操作は、あなたが許可するまで待ちます。
- **スキルがあります。** 字幕、無音部分の削除、モーショングラフィックス、製品デモ、プロジェクト設定がアプリに付属し、あなたが追加した MCP サーバーのツールも呼び出せます。
- **他のエージェントも Cutlass を操作できます。** `cutlass mcp /path/to/project` は、ウィンドウなしで同じツールを stdio で提供するので、Claude やどの MCP クライアントでもプロジェクトを読んで編集できます。

<a name="formats"></a>
## フォーマット

以下のデコーダとエンコーダは、[`crates/codecs`](../../crates/codecs) と [`crates/containers`](../../crates/containers) にある Rust のコードで、FFmpeg を外部の基準としてテストされています。ほとんどのクレートは、対応している内容の記録をソースの横に置いています。

| | 開く | 書き出し |
| --- | :---: | :---: |
| **ファイル** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4、MOV、音声のみ |
| **ビデオ** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG。HEVC は macOS と Windows でシステムのエンコーダを通して書き出します |
| **サウンド** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **画像** | JPEG, PNG, WebP, GIF（アニメーションまたは静止画） | GIF、アニメーション WebP |

<a name="build-from-source"></a>
## ソースからビルドする

安定版の [Rust](https://rustup.rs) ツールチェーンと C コンパイラが必要です。Linux では GTK と ALSA のヘッダも必要です。

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

続いて：

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

CI が実行するチェック（最後のものには [cargo-nextest](https://nexte.st) が必要です）：

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## まだないもの

Cutlass はまだ若いプロジェクトです。アプリの中では、描かれているがまだ作られていない機能は薄く表示され、**Later** と記されているので、何が動くのかがいつでも分かります。今ある大きな不足は次のとおりです。

- **その他のコンテナ。** MKV、WebM、AVI、MPEG-TS のファイルはまだ開けません。
- **音声ツール。** ノイズ低減、声の強調、ラウドネスの正規化、声の分離、テキスト読み上げ。
- **生成系のツール。** AI による除去、拡張、スタイル変換。
- **トラッキング。** 被写体を追いかけるテキスト、ステッカー、マスク。
- **より多くの言語の字幕。**
- **アシスタント用のホスト型モデル。** 現在はローカルのモデルを使います。
- **スマートフォン。** iOS と Android のアプリは [`apps`](../../apps) にあり、同じ Rust のコアを共有していますが、デスクトップが先です。

問題を起こしたファイルを添えたバグ報告が、送っていただけるものの中でいちばん役に立ちます。[issue を開く](https://github.com/1mrnewton/cutlass/issues)。

## コードの構成

| フォルダ | 中身 |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | デスクトップのエディタで、egui のウィンドウです。`cutlass` バイナリをビルドします |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | SwiftUI と Kotlin Compose で書かれたスマートフォンアプリ。`crates/ffi` を通して Rust のコアを使います |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | プロジェクト、取り消しできる編集リクエスト、すべてのアプリが共有するエディタの動作 |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | プロジェクトと時刻を、プレビューと書き出しのために 1 フレームとその音に変える部分 |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | コーデックごと、ファイル形式ごとに 1 つのクレート |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | システムのハードウェアコーデック |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | フィルター、エフェクト、トランジション、マスク、カラー、テキスト、字幕 |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | 映像用と音声認識用の、デバイス上で動くモデル |
| [`crates/agent`](../../crates/agent) | アシスタント：そのツール、スキル、MCP |

## コントリビュート

実際の素材で Cutlass を使い、どこで壊れるかを教えてください。[issue フォーム](https://github.com/1mrnewton/cutlass/issues/new/choose)は、バグ、機能、パフォーマンス、開けないファイルを扱います。

コードを送りますか？ まず上の 3 つのチェックを実行してください。共有クレートは特定のプラットフォームに依存しないので、同じ編集がデスクトップでもスマートフォンでも動きます。

<a name="license"></a>
## ライセンス

Cutlass はどちらのライセンスでも無料で使えます。ひとつ選んでください。

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

