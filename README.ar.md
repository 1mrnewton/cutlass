# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · **العربية** · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

<div dir="rtl" lang="ar">

Cutlass محرر فيديو على خط زمني. سطح المكتب و iOS و Android تشارك مسودة واحدة وجلسة واحدة. إيماءة على أي واجهة تصبح الطلب نفسه. التشغيل والتصدير يريان الإطار نفسه للمسودة نفسها واللحظة نفسها.

الزمن في المستند عدد ذو إشارة من الميكروثانية. المقطع يخزن نطاقين بهذه الوحدة. نطاق المصدر هو جزء الوسائط الذي يستخدمه. نطاق الهدف هو موضعه على الخط الزمني.

إذا اختلفت هذه الترجمة عن الشجرة، فالمرجع هو [README الإنجليزي](README.md).

## الحالة

مساحة العمل في بدايتها.

| الجزء | الحالة |
| --- | --- |
| `cutlass-model` | المسودة: المسارات والمقاطع ومخازن المواد والزمن بالميكروثانية |
| `cutlass-session` | الوضع والنقل والقص كخطوة تراجع واحدة. الطلب الفاشل يُبقي المسودة كما كانت. التراجع والإعادة يرفعان رقم المراجعة |
| بقية الـ crates | أعضاء فارغة. تحجز المحرك والحاويات والكودكات والتشغيل والتصدير والوكيل |
| `apps/desktop` | يطبع `cutlass` ثم يعود. النافذة لاحقًا |
| `apps/ios` و `apps/android` | تصلان إلى هذه الشجرة لاحقًا |

الحاويات والكودكات Rust أصيل، crate لكل عائلة أو كودك. على Apple و Windows و Android أطر الوسائط في المنصة هي المسار السريع. Linux يستخدم كودكات Rust ومعاينة برمجية.

## الترتيب

```text
apps/desktop/          egui و eframe
crates/                مكوّن واحد لكل crate
crates/containers/     عائلة حاويات واحدة لكل crate
crates/codecs/         كودك واحد لكل crate
```

أسماء الـ crates تبدأ بالبادئة `cutlass-`. ثنائي سطح المكتب يربط الـ crates مباشرة. واجهات الهاتف تربط `cutlass-ffi` فقط.

## البناء

ثبّت Rust المستقر. `rust-toolchain.toml` يختار القناة و Clippy و rustfmt.

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` يشغّل Clippy على مساحة العمل ويرفض التحذيرات. تكامل Ubuntu المستمر يشغّل هذه الفحوص و `cargo deny`.

## الترخيص

يتوفر Cutlass، حسب اختيارك، تحت إحدى الرخصتين:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) أو <https://www.apache.org/licenses/LICENSE-2.0>)
* رخصة MIT ([LICENSE-MIT](LICENSE-MIT) أو <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo

</div>
