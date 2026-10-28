# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · **Italiano** · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass è un editor video a timeline. Desktop, iOS e Android condividono una bozza e una sessione. Un gesto su qualunque interfaccia diventa la stessa richiesta. Riproduzione ed esportazione vedono lo stesso fotogramma per la stessa bozza e lo stesso istante.

Il tempo nel documento è un conteggio con segno di microsecondi. Un segmento conserva due intervalli in quell’unità. L’intervallo sorgente è la parte del media che usa. L’intervallo di destinazione è il punto sulla timeline.

Se questa traduzione e l’albero divergono, fa fede il [README in inglese](README.md).

## Stato

Il workspace è all’inizio.

| Pezzo | Stato |
| --- | --- |
| `cutlass-model` | La bozza: tracce, segmenti, pool di materiali e tempo in microsecondi |
| `cutlass-session` | Posizionare, spostare e rifilare come un solo passo di annullamento. Una richiesta fallita lascia la bozza com’era. Annulla e ripeti fanno avanzare la revisione |
| Gli altri crate | Membri vuoti. Riservano il motore, i container, i codec, la riproduzione, l’esportazione e l’agente |
| `apps/desktop` | Stampa `cutlass` e torna. La finestra arriva dopo |
| `apps/ios`, `apps/android` | Entrano in questo albero più avanti |

Container e codec sono Rust nativo, un crate per famiglia o codec. Su Apple, Windows e Android i framework multimediali della piattaforma sono il percorso rapido. Linux usa i codec Rust e un’anteprima software.

## Disposizione

```text
apps/desktop/          egui ed eframe
crates/                un componente per crate
crates/containers/     una famiglia di container per crate
crates/codecs/         un codec per crate
```

I nomi dei crate usano il prefisso `cutlass-`. Il binario desktop collega i crate direttamente. Le interfacce del telefono collegano solo `cutlass-ffi`.

## Compilazione

Installa Rust stable. `rust-toolchain.toml` sceglie il canale, Clippy e rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` esegue Clippy sul workspace e rifiuta gli avvisi. La CI su Ubuntu esegue questi controlli e `cargo deny`.

## Licenza

Cutlass è disponibile, a tua scelta, sotto una di queste licenze:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) oppure <https://www.apache.org/licenses/LICENSE-2.0>)
* Licenza MIT ([LICENSE-MIT](LICENSE-MIT) oppure <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
