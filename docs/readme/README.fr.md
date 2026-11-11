<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Un éditeur vidéo gratuit et open source, avec un assistant à qui vous pouvez parler.</b><br>
  Pour macOS, Windows et Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Télécharger</a> ·
  <a href="#what-you-can-do">Fonctions</a> ·
  <a href="#the-assistant">Assistant</a> ·
  <a href="#formats">Formats</a> ·
  <a href="#build-from-source">Compiler</a> ·
  <a href="#not-here-yet">Ce qui manque</a> ·
  <a href="https://cutlass.sh">Site web</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · **Français** · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass est un éditeur vidéo à timeline. Importez des vidéos, du son et des images, montez-les sur des pistes, ajoutez du texte, des effets et de la couleur, puis exportez une vidéo terminée. Vous pouvez faire chaque étape à la main, ou décrire le montage en langage courant et laisser l'assistant le réaliser sur la timeline, où vous pouvez le modifier ou l'annuler.

Il est gratuit. Pas de compte, pas de filigrane, pas d'envoi en ligne. Vos rushes restent sur votre disque, et les outils qui utilisent l'apprentissage automatique tournent sur votre propre ordinateur.

Sous l'éditeur, les codecs sont les nôtres. H.264, HEVC, AV1, ProRes, AAC, Opus et les autres sont écrits en Rust dans ce dépôt. Cutlass n'embarque pas FFmpeg.

<a name="download"></a>
## Télécharger

| Système | Fichier |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64 bits | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Chaque fichier est accompagné d'une somme de contrôle `.sha256` sur la page des [versions](https://github.com/1mrnewton/cutlass/releases). Une fois installé, Cutlass vérifie lui-même s'il existe une version plus récente : **Check for updates…** dans le menu.

Les builds ne sont pas encore signés avec un certificat de développeur, donc le premier lancement demande une étape de plus :

- **macOS:** ouvrez l'app une fois, puis autorisez-la dans System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** si SmartScreen bloque l'installateur, choisissez **More info** › **Run anyway**.
- **Linux:** rendez le fichier exécutable avec `chmod +x Cutlass-x86_64.AppImage`, puis lancez-le.

<a name="what-you-can-do"></a>
## Ce que vous pouvez faire

**Monter sur une timeline.** Pistes vidéo, audio et texte. Scindez, rognez, fermez les trous, figez une image, inversez, retournez en miroir, faites pivoter et recadrez. Groupez des clips, ou repliez-les dans un clip composé à rouvrir plus tard. Remplacez un clip en gardant ses réglages, ou copiez les attributs d'un clip sur un autre. Les rushes lourds peuvent être montés à travers un proxy.

**Vitesse et mouvement.** Réglez une vitesse constante, ou choisissez une courbe de vitesse et modifiez sa forme en faisant glisser ses points. Animez un clip avec des images clés, et déplacez ou supprimez-en plusieurs à la fois. Des animations d'entrée, de sortie et combinées sont prêtes à appliquer.

**Couleur.** Réglages de base, TSL (HSL), courbes et roues chromatiques. Chargez une LUT `.cube` et gardez des tons de peau naturels sous celle-ci. 200 filtres, et des calques de réglage sur la timeline.

**Effets et transitions.** 200 effets, 180 transitions et 13 modes de fusion. Masquez un clip avec une ligne, un miroir, un cercle, un rectangle, un cœur ou une étoile. Supprimez un fond vert, ou cliquez sur un objet pour le détourer.

**Texte, sous-titres et autocollants.** Du texte avec polices, contour, ombre, fond, bulles et effets, animé à l'entrée, à la sortie ou en boucle. Les sous-titres sont entendus à partir du clip sur votre machine (en anglais pour l'instant), puis modifiés sous forme de liste : retapez une ligne, recherchez et remplacez, choisissez un modèle, stylisez des mots isolés, et exportez le résultat en fichier SRT. Les autocollants et les formes se posent sur la timeline comme n'importe quel autre clip.

**Des outils d'image qui tournent sur votre ordinateur.** Stabiliser, améliorer la qualité, réduire le bruit de l'image, flux optique pour un ralenti fluide, rééclairer, recadrage automatique et flou de mouvement. Rien n'est envoyé nulle part.

**Son.** Volume et fondus, extraire l'audio d'une vidéo, synchroniser la vidéo et l'audio d'après leur son, et marquer le rythme.

**Exporter.** MP4 ou MOV : toute la timeline, une plage ou les clips sélectionnés. Le son seul, un GIF ou un WebP animé aussi. Sur macOS et Windows, les codecs matériels du système font le travail quand ils le peuvent, et un interrupteur dans les réglages les désactive.

<a name="the-assistant"></a>
## L'assistant

Dites ce que vous voulez, et l'assistant modifie le projet ouvert avec les mêmes requêtes que celles envoyées par la timeline. Chaque modification qu'il fait s'annule en une seule étape.

- **Il tourne sur votre machine.** L'assistant parle à un moteur de modèles sur votre propre ordinateur, comme [Ollama](https://ollama.com), le serveur de llama.cpp ou LM Studio. Indiquez l'adresse et le modèle dans Settings › Assistant. Par ce chemin, le prompt ne quitte pas votre ordinateur.
- **Il demande d'abord.** Tout ce qui enverrait des données hors de la machine attend votre accord.
- **Il a des compétences.** Sous-titres, suppression des pauses, motion design, démo de produit et configuration du projet sont livrés avec l'app, et il peut appeler les outils des serveurs MCP que vous ajoutez.
- **D'autres agents peuvent aussi piloter Cutlass.** `cutlass mcp /path/to/project` sert les mêmes outils sur stdio, sans fenêtre, si bien que Claude ou n'importe quel client MCP peut lire et modifier un projet.

<a name="formats"></a>
## Formats

Les décodeurs et encodeurs ci-dessous sont du code Rust dans [`crates/codecs`](../../crates/codecs) et [`crates/containers`](../../crates/containers), testé face à FFmpeg comme référence externe. Le README de chaque crate indique ce qu'il prend en charge et sa vitesse comparée à celle de FFmpeg.

| | Ouvrir | Exporter |
| --- | :---: | :---: |
| **Fichiers** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, et le son seul |
| **Vidéo** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC par l'encodeur du système sur macOS et Windows |
| **Son** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Images** | JPEG, PNG, WebP, GIF, animés ou fixes | GIF, WebP animé |

<a name="build-from-source"></a>
## Compiler depuis les sources

Il vous faut la chaîne d'outils stable de [Rust](https://rustup.rs) et un compilateur C. Sous Linux, aussi les en-têtes GTK et ALSA :

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Ensuite :

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

Les vérifications que lance la CI (la dernière nécessite [cargo-nextest](https://nexte.st)) :

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## Ce qui manque

Cutlass est jeune. Dans l'app, une fonction dessinée mais pas encore construite est grisée et marquée **Later**, donc vous savez toujours ce qui fonctionne. Les plus grands manques aujourd'hui :

- **Plus de conteneurs.** Les fichiers MKV, WebM, AVI et MPEG-TS ne s'ouvrent pas encore.
- **Outils pour la voix.** Réduction du bruit, amélioration de la voix, normalisation du volume sonore, isolation de la voix et synthèse vocale.
- **Outils génératifs.** Suppression, extension et stylisation par IA.
- **Suivi.** Du texte, des autocollants et des masques qui suivent un sujet.
- **Des sous-titres dans plus de langues.**
- **Un modèle hébergé pour l'assistant.** Il utilise aujourd'hui un modèle local.
- **Téléphones.** Les apps iOS et Android sont dans [`apps`](../../apps) et partagent le même cœur en Rust, mais le bureau passe en premier.

Un rapport de bug avec le fichier qui l'a provoqué est ce que vous pouvez envoyer de plus utile. [Ouvrez une issue](https://github.com/1mrnewton/cutlass/issues).

## Comment le code est organisé

| Dossier | Ce qu'il contient |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | L'éditeur de bureau, une fenêtre egui. Il produit le binaire `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Les apps pour téléphone, en SwiftUI et Kotlin Compose. Elles accèdent au cœur en Rust par `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | Le projet, les requêtes de montage avec annulation, et le comportement de l'éditeur partagé par toutes les apps |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Transformer un projet et un instant en une image et son audio, pour l'aperçu et pour l'export |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Un crate par codec et par format de fichier |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | Les codecs matériels du système |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filtres, effets, transitions, masques, couleur, texte et sous-titres |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | Les modèles d'image et de parole qui tournent sur l'appareil |
| [`crates/agent`](../../crates/agent) | L'assistant : ses outils, ses compétences et MCP |

## Contribuer

Utilisez Cutlass sur de vrais rushes et dites-nous où il casse. Les [formulaires d'issue](https://github.com/1mrnewton/cutlass/issues/new/choose) couvrent les bugs, les fonctions, les performances et les fichiers qui ne s'ouvrent pas.

Vous envoyez du code ? Lancez d'abord les trois vérifications ci-dessus. Les crates partagés ne dépendent d'aucune plateforme en particulier, donc le même montage fonctionne sur le bureau et sur téléphone.

<a name="license"></a>
## Licence

Cutlass est gratuit sous l'une ou l'autre licence. Choisissez-en une.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

