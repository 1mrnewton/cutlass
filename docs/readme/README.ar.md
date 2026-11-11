<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>محرر فيديو مجاني ومفتوح المصدر، مع مساعد يمكنك أن تكلمه.</b><br>
  لأنظمة macOS و Windows و Linux.
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">التنزيل</a> ·
  <a href="#what-you-can-do">الميزات</a> ·
  <a href="#the-assistant">المساعد</a> ·
  <a href="#formats">الصيغ</a> ·
  <a href="#build-from-source">البناء</a> ·
  <a href="#not-here-yet">ما لم يأتِ بعد</a> ·
  <a href="https://cutlass.sh">الموقع</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · **العربية** · [हिन्दी](README.hi.md) · [Bahasa Indonesia](README.id.md)

<div dir="rtl" lang="ar">

Cutlass محرر فيديو بخط زمني. أحضر اللقطات والصوت والصور، وقصّها على المسارات، وأضف النص والتأثيرات واللون، ثم صدّر فيديو جاهزًا. يمكنك أن تقوم بكل خطوة بيدك، أو أن تصف التحرير بكلام عادي وتترك المساعد ينفذه على الخط الزمني، حيث يمكنك تغييره أو التراجع عنه.

هو مجاني. لا حساب، ولا علامة مائية، ولا رفع للملفات. لقطاتك تبقى على قرصك، والأدوات التي تستخدم تعلّم الآلة تعمل على حاسوبك أنت.

تحت المحرر، برامج الترميز من صنعنا. H.264 و HEVC و AV1 و ProRes و AAC و Opus والبقية مكتوبة بلغة Rust في هذا المستودع. لا يتضمن Cutlass مكتبة FFmpeg.

<a name="download"></a>
## التنزيل

| النظام | الملف |
| --- | --- |
| macOS، معالجات Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows، 64 بت | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux، x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

بجانب كل ملف مجموع تحقق `.sha256` في صفحة [الإصدارات](https://github.com/1mrnewton/cutlass/releases). بعد التثبيت، يتحقق Cutlass بنفسه من وجود إصدار أحدث: **Check for updates…** في القائمة.

النسخ غير موقّعة بشهادة مطوّر بعد، لذلك يحتاج التشغيل الأول إلى خطوة إضافية:

- **macOS:** افتح التطبيق مرة واحدة، ثم اسمح به من System Settings › Privacy & Security › **Open Anyway**.
- **Windows:** إذا أوقف SmartScreen برنامج التثبيت، فاختر **More info** › **Run anyway**.
- **Linux:** اجعل الملف قابلًا للتنفيذ بالأمر `chmod +x Cutlass-x86_64.AppImage`، ثم شغّله.

<a name="what-you-can-do"></a>
## ما يمكنك فعله

**القص على خط زمني.** مسارات للفيديو والصوت والنص. قسّم، وشذّب، وأغلق الفراغات، وجمّد إطارًا، واعكس التشغيل، واقلب الصورة كالمرآة، ودوّر، واقتطع. اجمع المقاطع في مجموعة، أو اطوِها في مقطع مركّب وافتحه لاحقًا. استبدل مقطعًا مع الإبقاء على تعديلاته، أو انسخ خصائص مقطع إلى مقطع آخر. اللقطات الثقيلة يمكن قصها عبر ملف بديل (proxy).

**السرعة والحركة.** حدّد سرعة ثابتة، أو اختر منحنى سرعة وغيّر شكله بسحب نقاطه. حرّك المقطع بالإطارات المفتاحية، وانقل أو احذف عدة إطارات مفتاحية دفعة واحدة. حركات الدخول والخروج والحركات المركّبة جاهزة للتطبيق.

**اللون.** تعديلات أساسية، و HSL، ومنحنيات، وعجلات ألوان. حمّل ملف LUT بصيغة `.cube` وأبقِ ألوان البشرة طبيعية تحته. 200 فلتر، وطبقات تعديل على الخط الزمني.

**التأثيرات والانتقالات.** 200 تأثير، و 180 انتقالًا، و 13 نمط دمج. اقنع المقطع بخط أو مرآة أو دائرة أو مستطيل أو قلب أو نجمة. أزل الشاشة الخضراء، أو انقر على عنصر لتقصّه من خلفيته.

**النص والترجمة النصية والملصقات.** نص بخطوط وحدّ وظل وخلفية وفقاعات وتأثيرات، يتحرك عند الدخول أو الخروج أو في حلقة. تُستخرج الترجمة النصية بالاستماع إلى المقطع على جهازك (بالإنجليزية حاليًا)، ثم تُحرَّر كقائمة: أعد كتابة سطر، وابحث واستبدل، واختر قالبًا، ونسّق كلمات مفردة، وصدّر النتيجة كملف SRT. الملصقات والأشكال توضع على الخط الزمني مثل أي مقطع آخر.

**أدوات للصورة تعمل على حاسوبك.** تثبيت الاهتزاز، وتحسين الجودة، وتقليل ضجيج الصورة، والتدفق البصري لحركة بطيئة ناعمة، وإعادة الإضاءة، وإعادة التأطير تلقائيًا، وضبابية الحركة. لا يُرسَل شيء إلى أي مكان.

**الصوت.** مستوى الصوت والتلاشي، واستخراج الصوت من الفيديو، ومزامنة الفيديو والصوت بحسب صوتهما، ووضع علامات الإيقاع.

**التصدير.** MP4 أو MOV: الخط الزمني كله، أو نطاق منه، أو المقاطع المحددة. وكذلك الصوت وحده، أو GIF، أو WebP متحرك. على macOS و Windows تتولى برامج الترميز العتادية في النظام العمل متى استطاعت، ويوجد مفتاح في الإعدادات لإيقافها.

<a name="the-assistant"></a>
## المساعد

قل ما تريد، فيحرر المساعد المشروع المفتوح بالطلبات نفسها التي يرسلها الخط الزمني. كل تغيير يجريه هو خطوة تراجع واحدة.

- **يعمل على جهازك.** يتحدث المساعد إلى مشغّل نماذج على حاسوبك أنت، مثل [Ollama](https://ollama.com) أو خادم llama.cpp أو LM Studio. حدّد العنوان والنموذج في Settings › Assistant. في هذا المسار لا يغادر نص الطلب حاسوبك.
- **يسأل أولًا.** كل ما من شأنه أن يرسل بيانات خارج الجهاز ينتظر حتى تسمح به.
- **لديه مهارات.** الترجمة النصية، وإزالة فترات الصمت، والرسوم المتحركة، وعرض توضيحي لمنتج، وإعداد المشروع تأتي مع التطبيق، ويمكنه استدعاء أدوات من خوادم MCP التي تضيفها.
- **وكلاء آخرون يمكنهم تشغيل Cutlass أيضًا.** الأمر `cutlass mcp /path/to/project` يقدّم الأدوات نفسها عبر stdio بلا نافذة، فيستطيع Claude أو أي عميل MCP قراءة المشروع وتحريره.

<a name="formats"></a>
## الصيغ

برامج فك الترميز والترميز أدناه هي شيفرة Rust في [`crates/codecs`](../../crates/codecs) و [`crates/containers`](../../crates/containers)، وتُختبر بمقارنتها مع FFmpeg كمرجع خارجي. ملف README في كل حزمة يذكر ما تدعمه وسرعتها مقارنةً بـ FFmpeg.

| | الفتح | التصدير |
| --- | :---: | :---: |
| **الملفات** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4 و MOV، والصوت وحده |
| **الفيديو** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG. و HEVC عبر مرمّز النظام على macOS و Windows |
| **الصوت** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **الصور** | JPEG, PNG, WebP, GIF، متحركة أو ثابتة | GIF و WebP متحرك |

<a name="build-from-source"></a>
## البناء من المصدر

تحتاج إلى سلسلة أدوات [Rust](https://rustup.rs) المستقرة ومترجم للغة C. وعلى Linux تحتاج أيضًا إلى ملفات الترويسة الخاصة بـ GTK و ALSA:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

ثم:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

الفحوص التي تشغّلها CI (الأخير يحتاج إلى [cargo-nextest](https://nexte.st)):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## ما لم يأتِ بعد

Cutlass ما زال فتيًّا. في التطبيق، الميزة المرسومة التي لم تُبنَ بعد تظهر باهتة وعليها علامة **Later**، فتعرف دائمًا ما الذي يعمل. أكبر النواقص اليوم:

- **حاويات أكثر.** ملفات MKV و WebM و AVI و MPEG-TS لا تُفتح بعد.
- **أدوات الصوت البشري.** تقليل الضجيج، وتحسين الصوت، وتوحيد مستوى الجهارة، وعزل الصوت، وتحويل النص إلى كلام.
- **الأدوات التوليدية.** الإزالة والتوسيع والأسلبة بالذكاء الاصطناعي.
- **التتبع.** نصوص وملصقات وأقنعة تتبع عنصرًا في الصورة.
- **الترجمة النصية بلغات أكثر.**
- **نموذج مستضاف للمساعد.** يستخدم اليوم نموذجًا محليًا.
- **الهواتف.** تطبيقا iOS و Android موجودان في [`apps`](../../apps) ويتشاركان نواة Rust نفسها، لكن سطح المكتب يأتي أولًا.

بلاغ عن خلل مع الملف الذي سبّبه هو أنفع ما يمكنك إرساله. [افتح بلاغًا](https://github.com/1mrnewton/cutlass/issues).

## كيف نُظّمت الشيفرة

| المجلد | ما بداخله |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | محرر سطح المكتب، وهو نافذة egui. يبني الملف التنفيذي `cutlass` |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | تطبيقا الهاتف، بـ SwiftUI و Kotlin Compose. يصلان إلى نواة Rust عبر `crates/ffi` |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | المشروع، وطلبات التحرير مع التراجع، وسلوك المحرر المشترك بين كل التطبيقات |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | تحويل مشروع ولحظة زمنية إلى إطار وصوته، للمعاينة وللتصدير |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | حزمة واحدة لكل برنامج ترميز ولكل صيغة ملف |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | برامج الترميز العتادية في النظام |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | الفلاتر والتأثيرات والانتقالات والأقنعة واللون والنص والترجمة النصية |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | نماذج الصورة والكلام التي تعمل على الجهاز |
| [`crates/agent`](../../crates/agent) | المساعد: أدواته ومهاراته و MCP |

## المساهمة

استخدم Cutlass على لقطات حقيقية وأخبرنا أين يتعطل. [نماذج البلاغات](https://github.com/1mrnewton/cutlass/issues/new/choose) تغطي الأخطاء والميزات والأداء والملفات التي لا تُفتح.

سترسل شيفرة؟ شغّل الفحوص الثلاثة أعلاه أولًا. الحزم المشتركة لا ترتبط بمنصة بعينها، فيعمل التحرير نفسه على سطح المكتب وعلى الهواتف.

<a name="license"></a>
## الترخيص

Cutlass مجاني تحت أي من الرخصتين. اختر واحدة.

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

</div>
