# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · **Français** · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

Cutlass est un éditeur vidéo à timeline. Le bureau, iOS et Android partagent un brouillon et une session. Un geste sur n’importe quelle interface devient la même requête. La lecture et l’export voient la même image pour le même brouillon et le même instant.

Le temps du document est un décompte signé de microsecondes. Un segment enregistre deux plages dans cette unité. La plage source est la partie du média qu’il utilise. La plage cible est sa place sur la timeline.

Quand cette traduction et l’arbre divergent, le [README anglais](README.md) fait référence.

## État

L’espace de travail en est au début.

| Pièce | État |
| --- | --- |
| `cutlass-model` | Le brouillon : pistes, segments, réserves de matériaux, temps en microsecondes |
| `cutlass-session` | Placer, déplacer et rogner en une seule étape d’annulation. Une requête en échec laisse le brouillon tel quel. Annuler et rétablir font avancer la révision |
| Les autres crates | Membres vides. Ils réservent le moteur, les conteneurs, les codecs, la lecture, l’export et l’agent |
| `apps/desktop` | Affiche `cutlass` et revient. La fenêtre vient ensuite |
| `apps/ios`, `apps/android` | Arrivent dans cet arbre plus tard |

Les conteneurs et les codecs sont du Rust natif, un crate par famille ou par codec. Sur Apple, Windows et Android, les frameworks média de la plateforme sont le chemin rapide. Linux utilise les codecs Rust et un aperçu logiciel.

## Disposition

```text
apps/desktop/          egui et eframe
crates/                un composant par crate
crates/containers/     une famille de conteneur par crate
crates/codecs/         un codec par crate
```

Les noms de crates portent le préfixe `cutlass-`. Le binaire de bureau lie les crates directement. Les interfaces du téléphone ne lient que `cutlass-ffi`.

## Compilation

Installez Rust stable. `rust-toolchain.toml` choisit le canal, Clippy et rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` lance Clippy sur l’espace de travail et refuse les avertissements. La CI Ubuntu lance ces contrôles et `cargo deny`.

## Licence

Cutlass est disponible, à votre choix, sous l’une de ces licences :

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) ou <https://www.apache.org/licenses/LICENSE-2.0>)
* Licence MIT ([LICENSE-MIT](LICENSE-MIT) ou <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
