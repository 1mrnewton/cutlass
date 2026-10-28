# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · **Deutsch** · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass ist ein Timeline-Videoeditor. Desktop, iOS und Android teilen einen Entwurf und eine Sitzung. Eine Geste auf jeder Oberfläche wird dieselbe Anfrage. Wiedergabe und Export sehen für denselben Entwurf und denselben Zeitpunkt dasselbe Bild.

Die Zeit im Dokument ist eine vorzeichenbehaftete Zahl von Mikrosekunden. Ein Segment speichert zwei Bereiche in dieser Einheit. Der Quellbereich ist der genutzte Teil des Mediums. Der Zielbereich ist die Stelle auf der Timeline.

Weichen diese Übersetzung und der Baum voneinander ab, gilt die [englische README](README.md).

## Stand

Der Workspace steht am Anfang.

| Teil | Stand |
| --- | --- |
| `cutlass-model` | Der Entwurf: Spuren, Segmente, Materialpools und Zeit in Mikrosekunden |
| `cutlass-session` | Platzieren, verschieben und trimmen als ein Undo-Schritt. Eine fehlgeschlagene Anfrage lässt den Entwurf so, wie er war. Undo und Redo erhöhen die Revision |
| Übrige Crates | Leere Member. Sie halten den Platz für die Engine, Container, Codecs, Wiedergabe, Export und den Agenten |
| `apps/desktop` | Gibt `cutlass` aus und kehrt zurück. Das Fenster folgt |
| `apps/ios`, `apps/android` | Kommen später in diesen Baum |

Container und Codecs sind natives Rust, ein Crate je Familie oder Codec. Auf Apple, Windows und Android sind die Medien-Frameworks der Plattform der schnelle Weg. Linux nutzt die Rust-Codecs und eine Software-Vorschau.

## Aufbau

```text
apps/desktop/          egui und eframe
crates/                eine Komponente je Crate
crates/containers/     eine Container-Familie je Crate
crates/codecs/         ein Codec je Crate
```

Crate-Namen tragen das Präfix `cutlass-`. Die Desktop-Binary verlinkt die Crates direkt. Die Telefon-Oberflächen verlinken nur `cutlass-ffi`.

## Bauen

Installiere stabiles Rust. `rust-toolchain.toml` wählt den Channel, Clippy und rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` führt Clippy über den Workspace aus und weist Warnungen zurück. Die Ubuntu-CI führt diese Prüfungen und `cargo deny` aus.

## Lizenz

Cutlass steht nach deiner Wahl unter einer dieser Lizenzen:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) oder <https://www.apache.org/licenses/LICENSE-2.0>)
* MIT-Lizenz ([LICENSE-MIT](LICENSE-MIT) oder <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
