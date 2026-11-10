<p align="center">
  <img src="../../assets/app-icon.PNG" alt="Cutlass" width="112">
</p>

<h1 align="center">Cutlass</h1>

<p align="center">
  <b>एक मुफ़्त, ओपन सोर्स वीडियो एडिटर, जिसके साथ एक असिस्टेंट है जिससे आप बात कर सकते हैं।</b><br>
  macOS, Windows और Linux के लिए।
</p>

<p align="center">
  <a href="https://github.com/1mrnewton/cutlass/releases/latest"><img alt="Latest release" src="https://img.shields.io/github/v/release/1mrnewton/cutlass?style=flat-square&label=release"></a>
  <a href="https://github.com/1mrnewton/cutlass/actions/workflows/ci.yml"><img alt="Build status" src="https://img.shields.io/github/actions/workflow/status/1mrnewton/cutlass/ci.yml?branch=main&style=flat-square&label=build"></a>
  <a href="#license"><img alt="License: MIT or Apache-2.0" src="https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue?style=flat-square"></a>
</p>

<p align="center">
  <a href="#download">डाउनलोड</a> ·
  <a href="#what-you-can-do">फ़ीचर</a> ·
  <a href="#the-assistant">असिस्टेंट</a> ·
  <a href="#formats">फ़ॉर्मैट</a> ·
  <a href="#build-from-source">बिल्ड</a> ·
  <a href="#not-here-yet">जो अभी नहीं है</a> ·
  <a href="https://cutlass.sh">वेबसाइट</a>
</p>

[English](../../README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · **हिन्दी** · [Bahasa Indonesia](README.id.md)

Cutlass एक टाइमलाइन वीडियो एडिटर है। फ़ुटेज, आवाज़ और तस्वीरें लाइए, उन्हें ट्रैक पर काटिए, टेक्स्ट, इफ़ेक्ट और रंग जोड़िए, और तैयार वीडियो एक्सपोर्ट कीजिए। आप हर कदम अपने हाथ से कर सकते हैं, या एडिट को सीधी भाषा में बताकर असिस्टेंट से टाइमलाइन पर करवा सकते हैं, जहाँ आप उसे बदल सकते हैं या वापस ले सकते हैं।

यह मुफ़्त है। न कोई अकाउंट, न वॉटरमार्क, न अपलोड। आपकी फ़ुटेज आपकी डिस्क पर रहती है, और मशीन लर्निंग वाले टूल आपके अपने कंप्यूटर पर चलते हैं।

एडिटर के नीचे के कोडेक हमारे अपने हैं। H.264, HEVC, AV1, ProRes, AAC, Opus और बाकी सब इसी रिपॉज़िटरी में Rust में लिखे गए हैं। Cutlass के साथ FFmpeg नहीं आता।

<a name="download"></a>
## डाउनलोड

| सिस्टम | फ़ाइल |
| --- | --- |
| macOS, Apple silicon | [Cutlass.dmg](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass.dmg) |
| Windows, 64-बिट | [Cutlass-setup.exe](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-setup.exe) |
| Linux, x86_64 | [Cutlass-x86_64.AppImage](https://github.com/1mrnewton/cutlass/releases/latest/download/Cutlass-x86_64.AppImage) |

[रिलीज़](https://github.com/1mrnewton/cutlass/releases) पेज पर हर फ़ाइल के साथ एक `.sha256` चेकसम रखा है। इंस्टॉल होने के बाद Cutlass नई रिलीज़ खुद जाँचता है: मेनू में **Check for updates…**।

ये बिल्ड अभी डेवलपर सर्टिफ़िकेट से साइन नहीं हैं, इसलिए पहली बार खोलने में एक कदम ज़्यादा लगता है:

- **macOS:** ऐप को एक बार खोलिए, फिर System Settings › Privacy & Security › **Open Anyway** से उसे अनुमति दीजिए।
- **Windows:** अगर SmartScreen इंस्टॉलर को रोक दे, तो **More info** › **Run anyway** चुनिए।
- **Linux:** `chmod +x Cutlass-x86_64.AppImage` से फ़ाइल को चलाने लायक बनाइए, फिर उसे चलाइए।

<a name="what-you-can-do"></a>
## आप क्या कर सकते हैं

**टाइमलाइन पर काटिए।** वीडियो, ऑडियो और टेक्स्ट ट्रैक। क्लिप को बाँटिए, ट्रिम कीजिए, खाली जगह बंद कीजिए, फ़्रेम फ़्रीज़ कीजिए, उल्टा चलाइए, मिरर कीजिए, घुमाइए और क्रॉप कीजिए। क्लिप का ग्रुप बनाइए, या उन्हें एक कंपाउंड क्लिप में समेटकर बाद में खोलिए। किसी क्लिप को बदलिए और उसके एडिट बनाए रखिए, या एक क्लिप की विशेषताएँ दूसरी पर कॉपी कीजिए। भारी फ़ुटेज को प्रॉक्सी के ज़रिए काटा जा सकता है।

**रफ़्तार और गति।** एक स्थिर रफ़्तार तय कीजिए, या रफ़्तार का कर्व चुनकर उसके बिंदु खींचते हुए उसका आकार बदलिए। कीफ़्रेम से क्लिप को ऐनिमेट कीजिए, और कई कीफ़्रेम एक साथ हटाइए या खिसकाइए। इन, आउट और कॉम्बो ऐनिमेशन लगाने के लिए तैयार हैं।

**रंग।** बुनियादी एडजस्टमेंट, HSL, कर्व और कलर व्हील। `.cube` LUT लोड कीजिए और उसके नीचे त्वचा के रंग स्वाभाविक रखिए। 200 फ़िल्टर, और टाइमलाइन पर एडजस्टमेंट लेयर।

**इफ़ेक्ट और ट्रांज़िशन।** 200 इफ़ेक्ट, 180 ट्रांज़िशन और 13 ब्लेंड मोड। क्लिप को रेखा, मिरर, गोले, आयत, दिल या तारे के आकार से मास्क कीजिए। ग्रीन स्क्रीन हटाइए, या किसी चीज़ पर क्लिक करके उसे काटकर अलग कीजिए।

**टेक्स्ट, कैप्शन और स्टिकर।** फ़ॉन्ट, स्ट्रोक, छाया, बैकग्राउंड, बबल और इफ़ेक्ट वाला टेक्स्ट, जो आते समय, जाते समय या लूप में ऐनिमेट होता है। कैप्शन आपकी मशीन पर क्लिप को सुनकर बनते हैं (अभी सिर्फ़ अंग्रेज़ी), फिर एक सूची की तरह एडिट होते हैं: कोई पंक्ति दोबारा लिखिए, खोजिए और बदलिए, टेम्पलेट चुनिए, अलग-अलग शब्दों को स्टाइल दीजिए, और नतीजा SRT फ़ाइल के रूप में एक्सपोर्ट कीजिए। स्टिकर और आकृतियाँ किसी भी दूसरी क्लिप की तरह टाइमलाइन पर रहती हैं।

**तस्वीर के टूल जो आपके कंप्यूटर पर चलते हैं।** स्टेबलाइज़, क्वालिटी बढ़ाना, तस्वीर का नॉइज़ घटाना, स्मूद स्लो मोशन के लिए ऑप्टिकल फ़्लो, रीलाइट, ऑटो रीफ़्रेम और मोशन ब्लर। कुछ भी कहीं नहीं भेजा जाता।

**आवाज़।** वॉल्यूम और फ़ेड, वीडियो से ऑडियो निकालना, वीडियो और ऑडियो को उनकी आवाज़ से सिंक करना, और बीट पर निशान लगाना।

**एक्सपोर्ट।** MP4 या MOV: पूरी टाइमलाइन, कोई रेंज, या चुनी हुई क्लिप। सिर्फ़ आवाज़, GIF या ऐनिमेटेड WebP भी। macOS और Windows पर, जब हो सके तब सिस्टम के हार्डवेयर कोडेक काम करते हैं, और सेटिंग्स का एक स्विच उन्हें बंद कर देता है।

<a name="the-assistant"></a>
## असिस्टेंट

जो चाहिए वह कहिए, और असिस्टेंट खुले हुए प्रोजेक्ट को उन्हीं रिक्वेस्ट से एडिट करता है जो टाइमलाइन भेजती है। उसका किया हर बदलाव एक अनडू कदम है।

- **यह आपकी मशीन पर चलता है।** असिस्टेंट आपके अपने कंप्यूटर पर चल रहे मॉडल रनर से बात करता है, जैसे [Ollama](https://ollama.com), llama.cpp सर्वर या LM Studio। पता और मॉडल Settings › Assistant में तय कीजिए। इस रास्ते पर प्रॉम्प्ट आपके कंप्यूटर से बाहर नहीं जाता।
- **यह पहले पूछता है।** जो भी चीज़ डेटा को मशीन से बाहर भेजेगी, वह आपकी अनुमति का इंतज़ार करती है।
- **इसके पास स्किल हैं।** कैप्शन, ठहराव हटाना, मोशन ग्राफ़िक्स, प्रोडक्ट डेमो और प्रोजेक्ट सेटअप ऐप के साथ आते हैं, और यह आपके जोड़े हुए MCP सर्वर के टूल भी चला सकता है।
- **दूसरे एजेंट भी Cutlass चला सकते हैं।** `cutlass mcp /path/to/project` बिना विंडो के stdio पर वही टूल देता है, इसलिए Claude या कोई भी MCP क्लाइंट प्रोजेक्ट को पढ़ और एडिट कर सकता है।

<a name="formats"></a>
## फ़ॉर्मैट

नीचे के डिकोडर और एनकोडर [`crates/codecs`](../../crates/codecs) और [`crates/containers`](../../crates/containers) में Rust कोड हैं, जिन्हें FFmpeg को बाहरी संदर्भ मानकर परखा जाता है। ज़्यादातर crate अपने सोर्स के पास यह रिकॉर्ड रखते हैं कि वे क्या सपोर्ट करते हैं।

| | खोलना | एक्सपोर्ट |
| --- | :---: | :---: |
| **फ़ाइलें** | MP4, MOV, M4V, M4A, WAV, AIFF, FLAC, Ogg, MP3 | MP4, MOV, और सिर्फ़ आवाज़ |
| **वीडियो** | H.264, HEVC, AV1, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG | H.264, ProRes, VP8, MPEG-4, MPEG-2, Motion JPEG। HEVC, macOS और Windows पर सिस्टम के एनकोडर से |
| **आवाज़** | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM | AAC, MP3, FLAC, ALAC, Opus, Vorbis, AC-3, E-AC-3, PCM |
| **तस्वीरें** | JPEG, PNG, WebP, GIF, ऐनिमेटेड या स्थिर | GIF, ऐनिमेटेड WebP |

<a name="build-from-source"></a>
## सोर्स से बिल्ड करना

आपको स्टेबल [Rust](https://rustup.rs) टूलचेन और एक C कंपाइलर चाहिए। Linux पर GTK और ALSA के हेडर भी चाहिए:

```sh
sudo apt-get install libgtk-3-dev libasound2-dev
```

फिर:

```sh
git clone https://github.com/1mrnewton/cutlass
cd cutlass
cargo run --release -p cutlass
```

वे जाँचें जो CI चलाता है (आख़िरी के लिए [cargo-nextest](https://nexte.st) चाहिए):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --workspace
```

<a name="not-here-yet"></a>
## जो अभी नहीं है

Cutlass अभी नया है। ऐप में जो फ़ीचर दिखता है पर अभी बना नहीं है, वह धुँधला रहता है और उस पर **Later** लिखा होता है, ताकि आपको हमेशा पता रहे कि क्या काम करता है। आज की बड़ी कमियाँ:

- **और कंटेनर।** MKV, WebM, AVI और MPEG-TS फ़ाइलें अभी नहीं खुलतीं।
- **आवाज़ के टूल।** नॉइज़ घटाना, आवाज़ निखारना, लाउडनेस नॉर्मलाइज़ करना, आवाज़ अलग करना और टेक्स्ट से बोली।
- **जनरेटिव टूल।** AI से हटाना, फैलाना और स्टाइल देना।
- **ट्रैकिंग।** टेक्स्ट, स्टिकर और मास्क जो किसी विषय के पीछे चलें।
- **और भाषाओं में कैप्शन।**
- **असिस्टेंट के लिए होस्ट किया हुआ मॉडल।** आज यह लोकल मॉडल इस्तेमाल करता है।
- **फ़ोन।** iOS और Android ऐप [`apps`](../../apps) में हैं और वही Rust कोर साझा करते हैं, पर डेस्कटॉप पहले आता है।

जिस फ़ाइल से गड़बड़ हुई, उसके साथ भेजी गई बग रिपोर्ट सबसे काम की चीज़ है जो आप भेज सकते हैं। [एक issue खोलिए](https://github.com/1mrnewton/cutlass/issues)।

## कोड कैसे बँटा है

| फ़ोल्डर | इसमें क्या है |
| --- | --- |
| [`apps/desktop`](../../apps/desktop) | डेस्कटॉप एडिटर, एक egui विंडो। यह `cutlass` बाइनरी बनाता है |
| [`apps/ios`](../../apps/ios), [`apps/android`](../../apps/android) | फ़ोन ऐप, SwiftUI और Kotlin Compose में। ये `crates/ffi` के ज़रिए Rust कोर तक पहुँचते हैं |
| [`crates/model`](../../crates/model), [`crates/session`](../../crates/session), [`crates/editor`](../../crates/editor) | प्रोजेक्ट, अनडू वाली एडिट रिक्वेस्ट, और एडिटर का वह व्यवहार जो हर ऐप में साझा है |
| [`crates/eval`](../../crates/eval), [`crates/engine`](../../crates/engine), [`crates/render`](../../crates/render), [`crates/playback`](../../crates/playback), [`crates/export`](../../crates/export) | प्रीव्यू और एक्सपोर्ट के लिए, एक प्रोजेक्ट और एक समय को एक फ़्रेम और उसकी आवाज़ में बदलना |
| [`crates/codecs`](../../crates/codecs), [`crates/containers`](../../crates/containers) | हर कोडेक और हर फ़ाइल फ़ॉर्मैट के लिए एक crate |
| [`crates/platform-apple`](../../crates/platform-apple), [`crates/platform-windows`](../../crates/platform-windows) | सिस्टम के हार्डवेयर कोडेक |
| [`crates/effects`](../../crates/effects), [`crates/text`](../../crates/text), [`crates/captions`](../../crates/captions) | फ़िल्टर, इफ़ेक्ट, ट्रांज़िशन, मास्क, रंग, टेक्स्ट और कैप्शन |
| [`crates/vision`](../../crates/vision), [`crates/transcription`](../../crates/transcription) | डिवाइस पर चलने वाले तस्वीर और बोली के मॉडल |
| [`crates/agent`](../../crates/agent) | असिस्टेंट: उसके टूल, स्किल और MCP |

## योगदान

Cutlass को असली फ़ुटेज पर इस्तेमाल कीजिए और हमें बताइए कि वह कहाँ टूटता है। [issue फ़ॉर्म](https://github.com/1mrnewton/cutlass/issues/new/choose) में बग, फ़ीचर, परफ़ॉर्मेंस और न खुलने वाली फ़ाइलों के लिए जगह है।

कोड भेज रहे हैं? पहले ऊपर की तीन जाँचें चलाइए। साझा crate किसी एक प्लैटफ़ॉर्म से बँधे नहीं रहते, इसलिए वही एडिट डेस्कटॉप और फ़ोन दोनों पर काम करता है।

<a name="license"></a>
## लाइसेंस

Cutlass दोनों में से किसी भी लाइसेंस के तहत मुफ़्त है। एक चुनिए।

* [Apache License 2.0](../../LICENSE-APACHE)
* [MIT License](../../LICENSE-MIT)

Copyright (c) 2026 1mrnewton

