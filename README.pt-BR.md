# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · **Português** · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass é um editor de vídeo em linha do tempo. O desktop, o iOS e o Android compartilham um rascunho e uma sessão. Um gesto em qualquer interface vira o mesmo pedido. A reprodução e a exportação veem o mesmo quadro para o mesmo rascunho e o mesmo instante.

O tempo no documento é uma contagem com sinal de microssegundos. Um segmento guarda dois intervalos nessa unidade. O intervalo de origem é a parte da mídia que ele usa. O intervalo de destino é o lugar na linha do tempo.

Quando esta tradução e a árvore divergem, vale o [README em inglês](README.md).

## Estado

O workspace ainda é o começo.

| Peça | Estado |
| --- | --- |
| `cutlass-model` | O rascunho: trilhas, segmentos, conjuntos de materiais e tempo em microssegundos |
| `cutlass-session` | Colocar, mover e aparar como um único passo de desfazer. Um pedido que falha deixa o rascunho como estava. Desfazer e refazer avançam a revisão |
| Os outros crates | Membros vazios. Reservam o motor, os contêineres, os codecs, a reprodução, a exportação e o agente |
| `apps/desktop` | Imprime `cutlass` e retorna. A janela vem depois |
| `apps/ios`, `apps/android` | Entram nesta árvore mais adiante |

Contêineres e codecs são Rust nativo, um crate por família ou codec. Na Apple, no Windows e no Android, os frameworks de mídia da plataforma são o caminho rápido. O Linux usa os codecs em Rust e uma prévia em software.

## Disposição

```text
apps/desktop/          egui e eframe
crates/                um componente por crate
crates/containers/     uma família de contêiner por crate
crates/codecs/         um codec por crate
```

Os nomes dos crates usam o prefixo `cutlass-`. O binário de desktop liga os crates diretamente. As interfaces do telefone ligam apenas `cutlass-ffi`.

## Compilação

Instale o Rust stable. `rust-toolchain.toml` escolhe o canal, o Clippy e o rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` roda o Clippy no workspace e recusa avisos. A CI no Ubuntu roda essas verificações e o `cargo deny`.

## Licença

O Cutlass está disponível, à sua escolha, sob uma destas licenças:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) ou <https://www.apache.org/licenses/LICENSE-2.0>)
* Licença MIT ([LICENSE-MIT](LICENSE-MIT) ou <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
