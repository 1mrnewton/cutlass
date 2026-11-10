<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Un editor video gratuito e open source, con un assistente a cui puoi parlare.</b><br>
  Per macOS, Windows e Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Scarica</a> ·
  <a href="#what-you-can-do">Funzioni</a> ·
  <a href="#the-assistant">Assistente</a> ·
  <a href="#formats">Formati</a> ·
  <a href="#build-from-source">Compila</a> ·
  <a href="#not-here-yet">Cosa manca</a> ·
  <a href="https://cutlass.sh">Sito web</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · **Italiano** · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass è un editor video con timeline. Importa video, suono e immagini, tagliali sulle tracce, aggiungi testo, effetti e colore, ed esporta un video finito. Puoi fare ogni passaggio a mano, oppure descrivere il montaggio con parole semplici e lasciare che l'assistente lo esegua sulla timeline, dove puoi modificarlo o annullarlo.

È gratuito. Niente account, niente filigrana e niente caricamenti. Il tuo materiale resta sul tuo disco, e gli strumenti che usano l'apprendimento automatico girano sul tuo computer.

Sotto l'editor, i codec sono nostri. H.264, HEVC, AV1, ProRes, AAC, Opus e gli altri sono scritti in Rust in questo repository. Cutlass non include FFmpeg.

<a name="download"></a>
## Scarica

| Sistema | File |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64 bit | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Ogni file ha accanto un checksum `.sha256` nella pagina delle [release](https://github.com/1mrnewton/cutlass/releases). Una volta installato, Cutlass controlla da solo se c'è una versione più recente: **Check for updates…** nel menu.

Le build non sono ancora firmate con un certificato da sviluppatore, quindi il primo avvio richiede un passaggio in più:

- **macOS:** apri l'app una volta, poi consentila in System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** se SmartScreen blocca l'installer, scegli **More info** › **Run anyway**.
- **Linux:** rendi il file eseguibile con `chmod +x Cutlass-x86_64.AppImage`, poi avvialo.

<a name="what-you-can-do"></a>
## Cosa puoi fare

**Taglia su una timeline.** Tracce video, audio e di testo. Dividi, rifila, chiudi gli spazi vuoti, blocca un fotogramma, inverti, specchia, ruota e ritaglia. Raggruppa le clip, oppure ripiegale in una clip composta da riaprire più tardi. Sostituisci una clip mantenendo le sue modifiche, o copia gli attributi di una clip su un'altra. Il materiale pesante si può tagliare attraverso un proxy.

**Velocità e movimento.** Imposta una velocità costante, oppure scegli una curva di velocità e cambiale forma trascinando i suoi punti. Anima una clip con i fotogrammi chiave, e spostane o eliminane più di uno alla volta. Le animazioni di entrata, di uscita e combinate sono pronte da applicare.

**Colore.** Regolazioni di base, HSL, curve e ruote dei colori. Carica una LUT `.cube` e mantieni naturali i toni della pelle sotto di essa. 200 filtri, e livelli di regolazione sulla timeline.

**Effetti e transizioni.** 200 effetti, 180 transizioni e 13 metodi di fusione. Maschera una clip con una linea, uno specchio, un cerchio, un rettangolo, un cuore o una stella. Elimina uno schermo verde, oppure fai clic su un oggetto per scontornarlo.

**Testo, sottotitoli e adesivi.** Testo con font, contorno, ombra, sfondo, fumetti ed effetti, animato in entrata, in uscita o in loop. I sottotitoli vengono ascoltati dalla clip sul tuo computer (per ora solo in inglese) e poi modificati come un elenco: riscrivi una riga, trova e sostituisci, scegli un modello, dai uno stile a singole parole ed esporta il risultato come file SRT. Adesivi e forme stanno sulla timeline come qualsiasi altra clip.

**Strumenti per l'immagine che girano sul tuo computer.** Stabilizzazione, miglioramento della qualità, riduzione del rumore dell'immagine, flusso ottico per un rallentatore fluido, nuova illuminazione, reinquadratura automatica e sfocatura di movimento. Non viene inviato nulla da nessuna parte.

**Suono.** Volume e dissolvenze, estrazione dell'audio da un video, sincronizzazione di video e audio in base al loro suono, e marcatura del ritmo.

**Esportazione.** MP4 o MOV: l'intera timeline, un intervallo o le clip selezionate. Anche solo il suono, una GIF o un WebP animato. Su macOS e Windows i codec hardware del sistema fanno il lavoro quando possono, e un interruttore nelle impostazioni li disattiva.

<a name="the-assistant"></a>
## L'assistente

Di' quello che vuoi, e l'assistente modifica il progetto aperto con le stesse richieste che invia la timeline. Ogni modifica che fa è un solo passo di annullamento.

- **Gira sul tuo computer.** L'assistente parla con un esecutore di modelli sul tuo computer, come [Ollama](https://ollama.com), il server di llama.cpp o LM Studio. Imposta l'indirizzo e il modello in Settings › Assistant. Per questa via il prompt non lascia il tuo computer.
- **Prima chiede.** Tutto ciò che invierebbe dati fuori dal computer aspetta il tuo permesso.
- **Ha delle abilità.** Sottotitoli, rimozione delle pause, motion graphics, una demo di prodotto e la configurazione del progetto sono incluse nell'app, e può chiamare gli strumenti dei server MCP che aggiungi.
- **Anche altri agenti possono guidare Cutlass.** `cutlass mcp /path/to/project` offre gli stessi strumenti su stdio, senza finestra, così Claude o qualsiasi client MCP può leggere e modificare un progetto.

<a name="formats"></a>
## Formati

I decoder e gli encoder qui sotto sono codice Rust in [`crates/codecs`](../../crates/codecs) e [`crates/containers`](../../crates/containers), testato rispetto a FFmpeg come riferimento esterno. La maggior parte dei crate tiene, accanto al codice, un registro di ciò che supporta.

| | Apertura | Esportazione |
| --- | :---: | :---: |
| **File** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, e solo il suono |
| **Video** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC tramite l'encoder di sistema su macOS e Windows |
| **Suono** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Immagini** | JPEG, PNG, WebP, GIF, animati o fissi | GIF, WebP animato |

<a name="build-from-source"></a>
## Compilare dai sorgenti

Ti servono la toolchain stabile di [Rust](https://rustup.rs) e un compilatore C. Su Linux, anche gli header di GTK e ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Poi:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

I controlli che esegue la CI (l'ultimo richiede [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## Cosa manca

Cutlass è giovane. Nell'app, una funzione disegnata ma non ancora costruita è attenuata e contrassegnata con **Later**, così sai sempre cosa funziona. Le lacune più grandi oggi:

- **Altri contenitori.** I file MKV, WebM, AVI e MPEG-TS non si aprono ancora.
- **Strumenti per la voce.** Riduzione del rumore, miglioramento della voce, normalizzazione del volume percepito, isolamento della voce e sintesi vocale.
- **Strumenti generativi.** Rimozione, espansione e stilizzazione con l'IA.
- **Tracciamento.** Testo, adesivi e maschere che seguono un soggetto.
- **Sottotitoli in più lingue.**
- **Un modello ospitato per l'assistente.** Oggi usa un modello locale.
- **Telefoni.** Le app per iOS e Android sono in [`apps`](../../apps) e condividono lo stesso nucleo in Rust, ma il desktop viene prima.

Una segnalazione di bug con il file che l'ha causato è la cosa più utile che puoi inviare. [Apri una issue](https://github.com/1mrnewton/cutlass/issues).

## Com'è organizzato il codice

| Cartella | Cosa contiene |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | L'editor desktop, una finestra egui. Compila il binario `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Le app per telefono, in SwiftUI e Kotlin Compose. Raggiungono il nucleo in Rust tramite `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | Il progetto, le richieste di modifica con annullamento, e il comportamento dell'editor condiviso da tutte le app |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Trasformare un progetto e un istante in un fotogramma e nel suo suono, per l'anteprima e per l'esportazione |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Un crate per ogni codec e per ogni formato di file |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | I codec hardware del sistema |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filtri, effetti, transizioni, maschere, colore, testo e sottotitoli |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | I modelli per l'immagine e per il parlato che girano sul dispositivo |
| [`crates/agent`](../../crates/agent) | L'assistente: i suoi strumenti, le sue abilità e MCP |

## Contribuire

Usa Cutlass su materiale reale e dicci dove si rompe. I [moduli per le issue](https://github.com/1mrnewton/cutlass/issues/new/choose) coprono bug, funzioni, prestazioni e file che non si aprono.

Vuoi inviare codice? Esegui prima i tre controlli qui sopra. I crate condivisi non dipendono da nessuna piattaforma in particolare, così la stessa modifica funziona sul desktop e sui telefoni.

<a name="license"></a>
## Licenza

Cutlass è gratuito con una qualsiasi delle due licenze. Scegline una.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

