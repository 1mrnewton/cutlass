<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Un editor de vídeo gratuito y de código abierto, con un asistente al que puedes hablarle.</b><br>
  Para macOS, Windows y Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Descargar</a> ·
  <a href="#what-you-can-do">Funciones</a> ·
  <a href="#the-assistant">Asistente</a> ·
  <a href="#formats">Formatos</a> ·
  <a href="#build-from-source">Compilar</a> ·
  <a href="#not-here-yet">Lo que falta</a> ·
  <a href="https://cutlass.sh">Sitio web</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Español** · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass es un editor de vídeo con línea de tiempo. Importa vídeo, sonido e imágenes, córtalos en pistas, añade texto, efectos y color, y exporta un vídeo terminado. Puedes hacer cada paso a mano, o describir la edición con palabras sencillas y dejar que el asistente la haga en la línea de tiempo, donde puedes cambiarla o deshacerla.

Es gratis. No hay cuenta, ni marca de agua, ni subidas. Tu material se queda en tu disco, y las herramientas que usan aprendizaje automático se ejecutan en tu propio ordenador.

Bajo el editor, los códecs son nuestros. H.264, HEVC, AV1, ProRes, AAC, Opus y los demás están escritos en Rust en este repositorio. Cutlass no incluye FFmpeg.

<a name="download"></a>
## Descargar

| Sistema | Archivo |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64 bits | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Cada archivo tiene al lado una suma de comprobación `.sha256` en la página de [versiones](https://github.com/1mrnewton/cutlass/releases). Una vez instalado, Cutlass busca por sí mismo una versión más nueva: **Check for updates…** en el menú.

Las compilaciones aún no están firmadas con un certificado de desarrollador, así que el primer arranque necesita un paso más:

- **macOS:** abre la app una vez y luego permítela en System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** si SmartScreen detiene el instalador, elige **More info** › **Run anyway**.
- **Linux:** haz el archivo ejecutable con `chmod +x Cutlass-x86_64.AppImage` y luego ejecútalo.

<a name="what-you-can-do"></a>
## Lo que puedes hacer

**Corta en una línea de tiempo.** Pistas de vídeo, audio y texto. Divide, recorta, cierra huecos, congela un fotograma, invierte, refleja, gira y encuadra. Agrupa clips, o pliégalos en un clip compuesto y ábrelo después. Reemplaza un clip conservando sus ediciones, o copia los atributos de un clip a otro. El material pesado se puede cortar a través de un proxy.

**Velocidad y movimiento.** Fija una velocidad constante, o elige una curva de velocidad y dale forma arrastrando sus puntos. Anima un clip con fotogramas clave, y mueve o borra varios a la vez. Las animaciones de entrada, salida y combinadas están listas para aplicar.

**Color.** Ajustes básicos, HSL, curvas y ruedas de color. Carga una LUT `.cube` y mantén naturales los tonos de piel bajo ella. 200 filtros, y capas de ajuste en la línea de tiempo.

**Efectos y transiciones.** 200 efectos, 180 transiciones y 13 modos de fusión. Enmascara un clip con una línea, un espejo, un círculo, un rectángulo, un corazón o una estrella. Elimina una pantalla verde, o haz clic en un objeto para recortarlo.

**Texto, subtítulos y stickers.** Texto con fuentes, contorno, sombra, fondo, bocadillos y efectos, animado al entrar, al salir o en bucle. Los subtítulos se escuchan del clip en tu equipo (por ahora en inglés) y luego se editan como una lista: reescribe una línea, busca y reemplaza, elige una plantilla, da estilo a palabras sueltas y exporta el resultado como archivo SRT. Los stickers y las formas van en la línea de tiempo como cualquier otro clip.

**Herramientas de imagen que se ejecutan en tu ordenador.** Estabilizar, mejorar la calidad, reducir el ruido de imagen, flujo óptico para una cámara lenta suave, reiluminar, reencuadre automático y desenfoque de movimiento. No se envía nada a ningún sitio.

**Sonido.** Volumen y fundidos, extraer el audio de un vídeo, sincronizar vídeo y audio por su sonido, y marcar el ritmo.

**Exportar.** MP4 o MOV: toda la línea de tiempo, un rango o los clips seleccionados. También solo el sonido, un GIF o un WebP animado. En macOS y Windows, los códecs de hardware del sistema hacen el trabajo cuando pueden, y un interruptor en los ajustes los desactiva.

<a name="the-assistant"></a>
## El asistente

Di lo que quieres, y el asistente edita el proyecto abierto con las mismas peticiones que envía la línea de tiempo. Cada cambio que hace es un solo paso de deshacer.

- **Se ejecuta en tu equipo.** El asistente habla con un ejecutor de modelos en tu propio ordenador, como [Ollama](https://ollama.com), el servidor de llama.cpp o LM Studio. Indica la dirección y el modelo en Settings › Assistant. Por esta vía, el prompt no sale de tu ordenador.
- **Pregunta antes.** Todo lo que enviaría datos fuera del equipo espera a que tú lo permitas.
- **Tiene habilidades.** Subtítulos, quitar pausas, gráficos animados, una demo de producto y la configuración del proyecto vienen con la app, y puede llamar a herramientas de los servidores MCP que añadas.
- **Otros agentes también pueden manejar Cutlass.** `cutlass mcp /path/to/project` ofrece las mismas herramientas por stdio y sin ventana, así que Claude o cualquier cliente MCP puede leer y editar un proyecto.

<a name="formats"></a>
## Formatos

Los decodificadores y codificadores de abajo son código Rust en [`crates/codecs`](../../crates/codecs) y [`crates/containers`](../../crates/containers), probado contra FFmpeg como referencia externa. La mayoría de los crates guardan junto a su código un registro de lo que admiten.

| | Abrir | Exportar |
| --- | :---: | :---: |
| **Archivos** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, y solo el sonido |
| **Vídeo** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC mediante el codificador del sistema en macOS y Windows |
| **Sonido** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Imágenes** | JPEG, PNG, WebP, GIF, animados o fijos | GIF, WebP animado |

<a name="build-from-source"></a>
## Compilar desde el código fuente

Necesitas la cadena de herramientas estable de [Rust](https://rustup.rs) y un compilador de C. En Linux, también las cabeceras de GTK y ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Después:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

Las comprobaciones que ejecuta la CI (la última necesita [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## Lo que falta

Cutlass es joven. En la app, una función que está dibujada pero no construida aparece atenuada y marcada como **Later**, así siempre sabes qué funciona. Los huecos más grandes hoy:

- **Más contenedores.** Los archivos MKV, WebM, AVI y MPEG-TS todavía no se abren.
- **Herramientas de voz.** Reducción de ruido, mejora de la voz, normalización de sonoridad, aislamiento de la voz y texto a voz.
- **Herramientas generativas.** Eliminar, expandir y estilizar con IA.
- **Seguimiento.** Texto, stickers y máscaras que siguen a un sujeto.
- **Subtítulos en más idiomas.**
- **Un modelo alojado para el asistente.** Hoy usa un modelo local.
- **Móviles.** Las apps de iOS y Android están en [`apps`](../../apps) y comparten el mismo núcleo en Rust, pero el escritorio va primero.

Un informe de error con el archivo que lo causó es lo más útil que puedes enviar. [Abre una incidencia](https://github.com/1mrnewton/cutlass/issues).

## Cómo está organizado el código

| Carpeta | Qué contiene |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | El editor de escritorio, una ventana de egui. Compila el binario `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Las apps para móvil, en SwiftUI y Kotlin Compose. Llegan al núcleo en Rust a través de `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | El proyecto, las peticiones de edición con deshacer, y el comportamiento del editor que comparten todas las apps |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Convertir un proyecto y un instante en un fotograma y su sonido, para la vista previa y para la exportación |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Un crate por códec y por formato de archivo |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | Los códecs de hardware del sistema |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filtros, efectos, transiciones, máscaras, color, texto y subtítulos |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | Los modelos de imagen y de voz que se ejecutan en el dispositivo |
| [`crates/agent`](../../crates/agent) | El asistente: sus herramientas, sus habilidades y MCP |

## Contribuir

Usa Cutlass con material real y dinos dónde falla. Los [formularios de incidencias](https://github.com/1mrnewton/cutlass/issues/new/choose) cubren errores, funciones, rendimiento y archivos que no se abren.

¿Vas a enviar código? Ejecuta primero las tres comprobaciones de arriba. Los crates compartidos no dependen de ninguna plataforma concreta, así que la misma edición funciona en el escritorio y en el móvil.

<a name="license"></a>
## Licencia

Cutlass es gratuito bajo cualquiera de las dos licencias. Elige una.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

