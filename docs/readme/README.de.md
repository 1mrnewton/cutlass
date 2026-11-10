<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Ein kostenloser Open-Source-Videoeditor mit einem Assistenten, mit dem du sprechen kannst.</b><br>
  Für macOS, Windows und Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Download</a> ·
  <a href="#what-you-can-do">Funktionen</a> ·
  <a href="#the-assistant">Assistent</a> ·
  <a href="#formats">Formate</a> ·
  <a href="#build-from-source">Bauen</a> ·
  <a href="#not-here-yet">Was noch fehlt</a> ·
  <a href="https://cutlass.sh">Website</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · **Deutsch** · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass ist ein Videoeditor mit Timeline. Importiere Videomaterial, Ton und Bilder, schneide sie auf Spuren, füge Text, Effekte und Farbe hinzu und exportiere ein fertiges Video. Du kannst jeden Schritt von Hand machen oder den Schnitt in einfachen Worten beschreiben und den Assistenten ihn auf der Timeline ausführen lassen, wo du ihn ändern oder rückgängig machen kannst.

Es ist kostenlos. Es gibt kein Konto, kein Wasserzeichen und keinen Upload. Dein Material bleibt auf deiner Festplatte, und die Werkzeuge, die maschinelles Lernen nutzen, laufen auf deinem eigenen Computer.

Unter dem Editor sind die Codecs unsere eigenen. H.264, HEVC, AV1, ProRes, AAC, Opus und die übrigen sind in diesem Repository in Rust geschrieben. Cutlass liefert kein FFmpeg mit.

<a name="download"></a>
## Download

| System | Datei |
| --- | --- |
| macOS, Apple Silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64 Bit | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Zu jeder Datei liegt auf der [Releases](https://github.com/1mrnewton/cutlass/releases)-Seite eine `.sha256`-Prüfsumme. Nach der Installation sucht Cutlass selbst nach einer neueren Version: **Check for updates…** im Menü.

Die Builds sind noch nicht mit einem Entwicklerzertifikat signiert, deshalb braucht der erste Start einen Schritt mehr:

- **macOS:** öffne die App einmal und erlaube sie dann unter System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** wenn SmartScreen das Installationsprogramm anhält, wähle **More info** › **Run anyway**.
- **Linux:** mache die Datei mit `chmod +x Cutlass-x86_64.AppImage` ausführbar und starte sie dann.

<a name="what-you-can-do"></a>
## Was du tun kannst

**Auf einer Timeline schneiden.** Video-, Audio- und Textspuren. Teilen, trimmen, Lücken schließen, ein Bild einfrieren, rückwärts abspielen, spiegeln, drehen und zuschneiden. Gruppiere Clips oder falte sie zu einem zusammengesetzten Clip und öffne ihn später. Ersetze einen Clip und behalte seine Bearbeitungen, oder kopiere die Attribute eines Clips auf einen anderen. Schweres Material lässt sich über einen Proxy schneiden.

**Tempo und Bewegung.** Lege ein konstantes Tempo fest, oder wähle eine Tempokurve und forme sie, indem du ihre Punkte ziehst. Animiere einen Clip mit Keyframes, und verschiebe oder lösche mehrere Keyframes auf einmal. Animationen für Einstieg, Ausstieg und beides zusammen sind fertig zum Anwenden.

**Farbe.** Grundeinstellungen, HSL, Kurven und Farbräder. Lade eine `.cube`-LUT und halte Hauttöne darunter natürlich. 200 Filter und Einstellungsebenen auf der Timeline.

**Effekte und Übergänge.** 200 Effekte, 180 Übergänge und 13 Mischmodi. Maskiere einen Clip mit einer Linie, einem Spiegel, einem Kreis, einem Rechteck, einem Herz oder einem Stern. Stanze einen Greenscreen aus, oder klicke auf ein Objekt, um es freizustellen.

**Text, Untertitel und Sticker.** Text mit Schriften, Kontur, Schatten, Hintergrund, Sprechblasen und Effekten, animiert beim Einstieg, beim Ausstieg oder in einer Schleife. Untertitel werden auf deinem Rechner aus dem Clip herausgehört (vorerst nur Englisch) und dann als Liste bearbeitet: eine Zeile neu tippen, suchen und ersetzen, eine Vorlage wählen, einzelne Wörter gestalten und das Ergebnis als SRT-Datei exportieren. Sticker und Formen liegen wie jeder andere Clip auf der Timeline.

**Bildwerkzeuge, die auf deinem Computer laufen.** Stabilisieren, Qualität verbessern, Bildrauschen verringern, optischer Fluss für flüssige Zeitlupe, neu beleuchten, automatischer Bildausschnitt und Bewegungsunschärfe. Nichts wird irgendwohin gesendet.

**Ton.** Lautstärke und Blenden, den Ton aus einem Video herauslösen, Video und Audio anhand ihres Tons synchronisieren und den Beat markieren.

**Export.** MP4 oder MOV: die ganze Timeline, ein Bereich oder die ausgewählten Clips. Auch nur der Ton, ein GIF oder ein animiertes WebP. Unter macOS und Windows übernehmen die Hardware-Codecs des Systems die Arbeit, wenn sie können, und ein Schalter in den Einstellungen stellt sie ab.

<a name="the-assistant"></a>
## Der Assistent

Sag, was du willst, und der Assistent bearbeitet das geöffnete Projekt mit denselben Anfragen, die auch die Timeline sendet. Jede Änderung, die er macht, ist ein einzelner Rückgängig-Schritt.

- **Er läuft auf deinem Rechner.** Der Assistent spricht mit einem Modell-Runner auf deinem eigenen Computer, etwa [Ollama](https://ollama.com), dem llama.cpp-Server oder LM Studio. Adresse und Modell stellst du unter Settings › Assistant ein. Auf diesem Weg verlässt der Prompt deinen Computer nicht.
- **Er fragt zuerst.** Alles, was Daten vom Rechner wegschicken würde, wartet, bis du es erlaubst.
- **Er hat Skills.** Untertitel, Pausen entfernen, Motion Graphics, eine Produktdemo und die Projekteinrichtung werden mit der App geliefert, und er kann Werkzeuge der MCP-Server aufrufen, die du hinzufügst.
- **Auch andere Agenten können Cutlass steuern.** `cutlass mcp /path/to/project` stellt dieselben Werkzeuge über stdio ohne Fenster bereit, sodass Claude oder jeder MCP-Client ein Projekt lesen und bearbeiten kann.

<a name="formats"></a>
## Formate

Die Decoder und Encoder unten sind Rust-Code in [`crates/codecs`](../../crates/codecs) und [`crates/containers`](../../crates/containers), getestet gegen FFmpeg als äußere Referenz. Die meisten Crates führen neben ihrem Quellcode fest, was sie unterstützen.

| | Öffnen | Export |
| --- | :---: | :---: |
| **Dateien** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV und nur der Ton |
| **Video** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC über den Encoder des Systems unter macOS und Windows |
| **Ton** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Bilder** | JPEG, PNG, WebP, GIF, animiert oder als Standbild | GIF, animiertes WebP |

<a name="build-from-source"></a>
## Aus dem Quellcode bauen

Du brauchst die stabile [Rust](https://rustup.rs)-Toolchain und einen C-Compiler. Unter Linux außerdem die Header von GTK und ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Dann:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

Die Prüfungen, die die CI ausführt (die letzte braucht [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## Was noch fehlt

Cutlass ist jung. In der App ist eine Funktion, die gezeichnet, aber noch nicht gebaut ist, abgedunkelt und mit **Later** markiert, sodass du immer weißt, was funktioniert. Die größeren Lücken heute:

- **Mehr Container.** MKV-, WebM-, AVI- und MPEG-TS-Dateien lassen sich noch nicht öffnen.
- **Werkzeuge für die Stimme.** Rauschminderung, Stimmverbesserung, Lautheitsnormalisierung, Stimmisolierung und Text-zu-Sprache.
- **Generative Werkzeuge.** Entfernen, Erweitern und Stilisieren mit KI.
- **Tracking.** Text, Sticker und Masken, die einem Motiv folgen.
- **Untertitel in mehr Sprachen.**
- **Ein gehostetes Modell für den Assistenten.** Heute nutzt er ein lokales Modell.
- **Smartphones.** Die Apps für iOS und Android liegen in [`apps`](../../apps) und teilen denselben Rust-Kern, aber der Desktop kommt zuerst.

Ein Fehlerbericht mit der Datei, die ihn ausgelöst hat, ist das Nützlichste, was du schicken kannst. [Eröffne ein Issue](https://github.com/1mrnewton/cutlass/issues).

## Wie der Code aufgebaut ist

| Ordner | Was darin liegt |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | Der Desktop-Editor, ein egui-Fenster. Er baut das Binary `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Die Smartphone-Apps in SwiftUI und Kotlin Compose. Sie erreichen den Rust-Kern über `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | Das Projekt, die Bearbeitungsanfragen mit Rückgängig und das Editorverhalten, das alle Apps teilen |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Aus einem Projekt und einem Zeitpunkt ein Bild und seinen Ton machen, für die Vorschau und für den Export |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Ein Crate pro Codec und pro Dateiformat |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | Die Hardware-Codecs des Systems |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filter, Effekte, Übergänge, Masken, Farbe, Text und Untertitel |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | Die Modelle für Bild und Sprache, die auf dem Gerät laufen |
| [`crates/agent`](../../crates/agent) | Der Assistent: seine Werkzeuge, seine Skills und MCP |

## Mitwirken

Nutze Cutlass mit echtem Material und sag uns, wo es bricht. Die [Issue-Formulare](https://github.com/1mrnewton/cutlass/issues/new/choose) decken Fehler, Funktionen, Leistung und Dateien ab, die sich nicht öffnen lassen.

Du schickst Code? Führe zuerst die drei Prüfungen oben aus. Gemeinsame Crates bleiben frei von einer einzelnen Plattform, sodass derselbe Schnitt auf dem Desktop und auf Smartphones funktioniert.

<a name="license"></a>
## Lizenz

Cutlass ist unter jeder der beiden Lizenzen kostenlos. Wähle eine.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

