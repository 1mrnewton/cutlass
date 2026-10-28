# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · **Bahasa Indonesia**

Cutlass adalah editor video berbasis linimasa. Desktop, iOS, dan Android memakai satu draf dan satu sesi. Gestur di kulit mana pun menjadi permintaan yang sama. Pemutaran dan ekspor melihat bingkai yang sama untuk draf dan waktu yang sama.

Waktu dalam dokumen adalah hitungan mikrodetik bertanda. Sebuah segmen menyimpan dua rentang dalam satuan itu. Rentang sumber adalah bagian media yang dipakai. Rentang target adalah tempatnya di linimasa.

Bila terjemahan ini dan pohon ini berbeda, [README bahasa Inggris](README.md) yang menjadi acuan.

## Status

Workspace ini masih awal.

| Bagian | Status |
| --- | --- |
| `cutlass-model` | Draf: trek, segmen, kumpulan materi, dan waktu mikrodetik |
| `cutlass-session` | Menempatkan, memindahkan, dan memangkas sebagai satu langkah pembatalan. Permintaan yang gagal membiarkan draf tetap. Batalkan dan ulangi menaikkan revisi |
| Krate lain | Anggota kosong. Mereka menyiapkan tempat untuk mesin, kontainer, kodek, pemutaran, ekspor, dan agen |
| `apps/desktop` | Mencetak `cutlass` lalu kembali. Jendela menyusul |
| `apps/ios`, `apps/android` | Masuk ke pohon ini nanti |

Kontainer dan kodek adalah Rust asli, satu krate per keluarga atau kodek. Di Apple, Windows, dan Android, kerangka media platform adalah jalur cepat. Linux memakai kodek Rust dan pratinjau perangkat lunak.

## Tata letak

```text
apps/desktop/          egui dan eframe
crates/                satu komponen per krate
crates/containers/     satu keluarga kontainer per krate
crates/codecs/         satu kodek per krate
```

Nama krate memakai awalan `cutlass-`. Biner desktop menautkan krate secara langsung. Kulit telepon hanya menautkan `cutlass-ffi`.

## Bangun

Pasang Rust stable. `rust-toolchain.toml` memilih kanal, Clippy, dan rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` menjalankan Clippy pada workspace dan menolak peringatan. CI Ubuntu menjalankan pemeriksaan ini dan `cargo deny`.

## Lisensi

Cutlass tersedia, menurut pilihan Anda, di bawah salah satu lisensi berikut:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) atau <https://www.apache.org/licenses/LICENSE-2.0>)
* Lisensi MIT ([LICENSE-MIT](LICENSE-MIT) atau <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
