# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass es un editor de vídeo por línea de tiempo. El escritorio, iOS y Android comparten un borrador y una sesión. Un gesto en cualquier interfaz se convierte en la misma petición. La reproducción y la exportación ven el mismo fotograma para el mismo borrador y el mismo instante.

El tiempo del documento es un recuento con signo de microsegundos. Un segmento guarda dos intervalos en esa unidad. El intervalo de origen es la parte del medio que usa. El intervalo de destino es su lugar en la línea de tiempo.

Si esta traducción y el árbol no coinciden, manda el [README en inglés](README.md).

## Estado

El espacio de trabajo está al principio.

| Pieza | Estado |
| --- | --- |
| `cutlass-model` | El borrador: pistas, segmentos, grupos de materiales y tiempo en microsegundos |
| `cutlass-session` | Colocar, mover y recortar como un solo paso de deshacer. Una petición fallida deja el borrador como estaba. Deshacer y rehacer avanzan la revisión |
| Los demás crates | Miembros vacíos. Reservan el motor, los contenedores, los códecs, la reproducción, la exportación y el agente |
| `apps/desktop` | Escribe `cutlass` y vuelve. La ventana llega después |
| `apps/ios`, `apps/android` | Entran en este árbol más adelante |

Los contenedores y los códecs son Rust nativo, un crate por familia o códec. En Apple, Windows y Android los marcos multimedia de la plataforma son el camino rápido. Linux usa los códecs en Rust y una vista previa por software.

## Disposición

```text
apps/desktop/          egui y eframe
crates/                un componente por crate
crates/containers/     una familia de contenedor por crate
crates/codecs/         un códec por crate
```

Los nombres de los crates usan el prefijo `cutlass-`. El binario de escritorio enlaza los crates directamente. Las interfaces del teléfono enlazan solo `cutlass-ffi`.

## Compilación

Instala Rust stable. `rust-toolchain.toml` elige el canal, Clippy y rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` ejecuta Clippy en el espacio de trabajo y rechaza las advertencias. La CI de Ubuntu ejecuta estas comprobaciones y `cargo deny`.

## Licencia

Cutlass está disponible, a tu elección, bajo una de estas licencias:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) o <https://www.apache.org/licenses/LICENSE-2.0>)
* Licencia MIT ([LICENSE-MIT](LICENSE-MIT) o <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
