<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>Editor video gratis dan sumber terbuka, dengan asisten yang bisa Anda ajak bicara.</b><br>
  Untuk macOS, Windows, dan Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">Unduh</a> ·
  <a href="#what-you-can-do">Fitur</a> ·
  <a href="#the-assistant">Asisten</a> ·
  <a href="#formats">Format</a> ·
  <a href="#build-from-source">Build</a> ·
  <a href="#not-here-yet">Yang belum ada</a> ·
  <a href="https://cutlass.sh">Situs web</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · [हिन्दी](README.hi.md) · **Bahasa Indonesia**

Cutlass adalah editor video dengan linimasa. Masukkan rekaman, suara, dan gambar, potong di atas trek, tambahkan teks, efek, dan warna, lalu ekspor video yang sudah jadi. Anda bisa mengerjakan setiap langkah sendiri, atau menjelaskan suntingannya dengan bahasa biasa dan membiarkan asisten mengerjakannya di linimasa, tempat Anda bisa mengubah atau membatalkannya.

Cutlass gratis. Tidak ada akun, tidak ada tanda air, dan tidak ada unggahan. Rekaman Anda tetap di disk Anda, dan alat yang memakai pembelajaran mesin berjalan di komputer Anda sendiri.

Di bawah editor, kodeknya buatan kami sendiri. H.264, HEVC, AV1, ProRes, AAC, Opus, dan yang lainnya ditulis dalam Rust di repositori ini. Cutlass tidak menyertakan FFmpeg.

<a name="download"></a>
## Unduh

| Sistem | Berkas |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64-bit | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

Setiap berkas punya checksum `.sha256` di sampingnya pada halaman [rilis](https://github.com/1mrnewton/cutlass/releases). Setelah terpasang, Cutlass memeriksa sendiri apakah ada rilis yang lebih baru: **Check for updates…** di menu.

Build ini belum ditandatangani dengan sertifikat pengembang, jadi saat pertama dibuka perlu satu langkah tambahan:

- **macOS:** buka aplikasinya sekali, lalu izinkan di System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** jika SmartScreen menghentikan pemasang, pilih **More info** › **Run anyway**.
- **Linux:** jadikan berkasnya bisa dijalankan dengan `chmod +x Cutlass-x86_64.AppImage`, lalu jalankan.

<a name="what-you-can-do"></a>
## Yang bisa Anda lakukan

**Memotong di linimasa.** Trek video, audio, dan teks. Belah, pangkas, tutup celah, bekukan satu bingkai, putar mundur, cerminkan, putar, dan potong tepi gambar. Kelompokkan klip, atau lipat menjadi klip gabungan dan buka lagi nanti. Ganti sebuah klip tanpa kehilangan suntingannya, atau salin atribut satu klip ke klip lain. Rekaman yang berat bisa dipotong melalui proksi.

**Kecepatan dan gerak.** Atur kecepatan tetap, atau pilih kurva kecepatan dan ubah bentuknya dengan menyeret titik-titiknya. Animasikan klip dengan keyframe, dan pindahkan atau hapus beberapa keyframe sekaligus. Animasi masuk, keluar, dan kombinasi siap dipakai.

**Warna.** Penyesuaian dasar, HSL, kurva, dan roda warna. Muat LUT `.cube` dan jaga warna kulit tetap alami di bawahnya. 200 filter, dan lapisan penyesuaian di linimasa.

**Efek dan transisi.** 200 efek, 180 transisi, dan 13 mode campuran. Beri klip mask berbentuk garis, cermin, lingkaran, persegi panjang, hati, atau bintang. Hilangkan layar hijau, atau klik sebuah objek untuk memotongnya dari latar.

**Teks, takarir, dan stiker.** Teks dengan fon, garis tepi, bayangan, latar, balon, dan efek, beranimasi saat masuk, saat keluar, atau berulang. Takarir didengarkan dari klip di mesin Anda (untuk saat ini bahasa Inggris), lalu disunting sebagai daftar: ketik ulang satu baris, cari dan ganti, pilih templat, beri gaya pada kata tertentu, dan ekspor hasilnya sebagai berkas SRT. Stiker dan bentuk berada di linimasa seperti klip lainnya.

**Alat gambar yang berjalan di komputer Anda.** Stabilkan, tingkatkan kualitas, kurangi derau gambar, optical flow untuk gerak lambat yang halus, atur ulang cahaya, pembingkaian ulang otomatis, dan blur gerak. Tidak ada yang dikirim ke mana pun.

**Suara.** Volume dan fade, ekstrak audio dari video, sinkronkan video dan audio berdasarkan suaranya, dan tandai ketukan.

**Ekspor.** MP4 atau MOV: seluruh linimasa, satu rentang, atau klip yang dipilih. Juga suara saja, GIF, atau WebP animasi. Di macOS dan Windows, kodek perangkat keras milik sistem mengerjakannya bila bisa, dan ada sakelar di pengaturan untuk mematikannya.

<a name="the-assistant"></a>
## Asisten

Katakan apa yang Anda inginkan, dan asisten menyunting proyek yang sedang terbuka dengan permintaan yang sama seperti yang dikirim linimasa. Setiap perubahan yang dibuatnya adalah satu langkah urungkan.

- **Berjalan di mesin Anda.** Asisten berbicara dengan penjalan model di komputer Anda sendiri, seperti [Ollama](https://ollama.com), server llama.cpp, atau LM Studio. Atur alamat dan modelnya di Settings › Assistant. Lewat jalur ini, prompt tidak meninggalkan komputer Anda.
- **Bertanya dulu.** Apa pun yang akan mengirim data keluar dari mesin menunggu sampai Anda mengizinkannya.
- **Punya keahlian.** Takarir, menghapus jeda, grafik gerak, demo produk, dan penyiapan proyek sudah disertakan dalam aplikasi, dan asisten bisa memanggil alat dari server MCP yang Anda tambahkan.
- **Agen lain juga bisa mengendalikan Cutlass.** `cutlass mcp /path/to/project` menyediakan alat yang sama lewat stdio tanpa jendela, sehingga Claude atau klien MCP mana pun bisa membaca dan menyunting proyek.

<a name="formats"></a>
## Format

Dekoder dan enkoder di bawah ini adalah kode Rust di [`crates/codecs`](../../crates/codecs) dan [`crates/containers`](../../crates/containers), yang diuji terhadap FFmpeg sebagai acuan luar. Sebagian besar crate menyimpan catatan tentang apa yang didukungnya di samping kode sumbernya.

| | Buka | Ekspor |
| --- | :---: | :---: |
| **Berkas** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, dan suara saja |
| **Video** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. HEVC melalui enkoder sistem di macOS dan Windows |
| **Suara** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **Gambar** | JPEG, PNG, WebP, GIF, animasi atau diam | GIF, WebP animasi |

<a name="build-from-source"></a>
## Build dari kode sumber

Anda memerlukan toolchain [Rust](https://rustup.rs) stabil dan sebuah kompiler C. Di Linux, juga header GTK dan ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

Lalu:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

Pemeriksaan yang dijalankan CI (yang terakhir memerlukan [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## Yang belum ada

Cutlass masih muda. Di aplikasi, fitur yang sudah digambar tetapi belum dibuat tampil redup dan ditandai **Later**, jadi Anda selalu tahu apa yang berfungsi. Kekurangan terbesar saat ini:

- **Lebih banyak kontainer.** Berkas MKV, WebM, AVI, dan MPEG-TS belum bisa dibuka.
- **Alat suara vokal.** Pengurangan derau, penyempurnaan suara, normalisasi kenyaringan, isolasi suara, dan teks ke ucapan.
- **Alat generatif.** Hapus, perluas, dan beri gaya dengan AI.
- **Pelacakan.** Teks, stiker, dan mask yang mengikuti subjek.
- **Takarir dalam lebih banyak bahasa.**
- **Model terhosting untuk asisten.** Saat ini asisten memakai model lokal.
- **Ponsel.** Aplikasi iOS dan Android ada di [`apps`](../../apps) dan memakai inti Rust yang sama, tetapi desktop didahulukan.

Laporan bug yang disertai berkas penyebabnya adalah hal paling berguna yang bisa Anda kirim. [Buka issue](https://github.com/1mrnewton/cutlass/issues).

## Susunan kode

| Folder | Isinya |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | Editor desktop, sebuah jendela egui. Menghasilkan biner `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | Aplikasi ponsel, dalam SwiftUI dan Kotlin Compose. Keduanya mencapai inti Rust melalui `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | Proyek, permintaan suntingan dengan urungkan, dan perilaku editor yang dipakai bersama oleh semua aplikasi |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | Mengubah sebuah proyek dan satu titik waktu menjadi satu bingkai dan suaranya, untuk pratinjau dan untuk ekspor |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | Satu crate untuk tiap kodek dan tiap format berkas |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | Kodek perangkat keras milik sistem |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | Filter, efek, transisi, mask, warna, teks, dan takarir |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | Model untuk gambar dan ucapan yang berjalan di perangkat |
| [`crates/agent`](../../crates/agent) | Asisten: alat, keahlian, dan MCP-nya |

## Berkontribusi

Pakai Cutlass pada rekaman sungguhan dan beri tahu kami di mana ia bermasalah. [Formulir issue](https://github.com/1mrnewton/cutlass/issues/new/choose) mencakup bug, fitur, kinerja, dan berkas yang tidak bisa dibuka.

Mau mengirim kode? Jalankan dulu tiga pemeriksaan di atas. Crate bersama tidak terikat pada satu platform, sehingga suntingan yang sama berfungsi di desktop dan di ponsel.

<a name="license"></a>
## Lisensi

Cutlass gratis di bawah salah satu dari dua lisensi ini. Pilih satu.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

