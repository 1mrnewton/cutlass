# Cutlass

[English](README.md) · [简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Italiano](README.it.md) · [Русский](README.ru.md) · [العربية](README.ar.md) · **हिन्दी** · [Bahasa Indonesia](README.id.md)

Cutlass एक टाइमलाइन वीडियो संपादक है। डेस्कटॉप, iOS और Android एक ही ड्राफ्ट और एक ही सत्र साझा करते हैं। किसी भी शेल पर इशारा वही अनुरोध बनता है। प्लेबैक और एक्सपोर्ट एक ही ड्राफ्ट और एक ही समय पर एक ही फ़्रेम देखते हैं।

दस्तावेज़ में समय माइक्रोसेकंड की चिह्न-सहित गिनती है। एक सेगमेंट इस इकाई में दो अंतराल रखता है। स्रोत अंतराल मीडिया का वह भाग है जिसे वह उपयोग करता है। लक्ष्य अंतराल टाइमलाइन पर उसका स्थान है।

जब यह अनुवाद और यह ट्री अलग हों, तब [अंग्रेज़ी README](README.md) आधार है।

## स्थिति

वर्कस्पेस शुरुआती अवस्था में है।

| हिस्सा | स्थिति |
| --- | --- |
| `cutlass-model` | ड्राफ्ट: ट्रैक, सेगमेंट, सामग्री पूल, और माइक्रोसेकंड समय |
| `cutlass-session` | रखना, खिसकाना और ट्रिम एक ही पूर्ववत चरण के रूप में। असफल अनुरोध ड्राफ्ट को वैसा ही छोड़ता है। पूर्ववत और फिर से करना संशोधन आगे बढ़ाते हैं |
| बाकी क्रेट | खाली सदस्य। ये इंजन, कंटेनर, कोडेक, प्लेबैक, एक्सपोर्ट और एजेंट की जगह रखते हैं |
| `apps/desktop` | `cutlass` छापकर लौटता है। विंडो बाद में आती है |
| `apps/ios`, `apps/android` | इस ट्री में बाद में आते हैं |

कंटेनर और कोडेक मूल Rust हैं, हर परिवार या कोडेक के लिए एक क्रेट। Apple, Windows और Android पर प्लेटफ़ॉर्म के मीडिया फ़्रेमवर्क तेज़ मार्ग हैं। Linux Rust कोडेक और सॉफ़्टवेयर पूर्वावलोकन का उपयोग करता है।

## विन्यास

```text
apps/desktop/          egui और eframe
crates/                हर घटक के लिए एक क्रेट
crates/containers/     हर कंटेनर परिवार के लिए एक क्रेट
crates/codecs/         हर कोडेक के लिए एक क्रेट
```

क्रेट के नाम `cutlass-` उपसर्ग का उपयोग करते हैं। डेस्कटॉप बाइनरी क्रेट को सीधे जोड़ती है। फ़ोन शेल केवल `cutlass-ffi` जोड़ते हैं।

## बिल्ड

स्थिर Rust स्थापित करें। `rust-toolchain.toml` चैनल, Clippy और rustfmt चुनता है।

```sh
cargo test --workspace
cargo fmt --all -- --check
cargo lint
```

`cargo lint` वर्कस्पेस पर Clippy चलाता है और चेतावनियों को अस्वीकार करता है। Ubuntu CI ये जाँच और `cargo deny` चलाता है।

## लाइसेंस

Cutlass इनमें से किसी एक लाइसेंस के तहत, आपकी पसंद पर, उपलब्ध है:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) या <https://www.apache.org/licenses/LICENSE-2.0>)
* MIT लाइसेंस ([LICENSE-MIT](LICENSE-MIT) या <https://opensource.org/licenses/MIT>)

Copyright (c) 2026 Emmanuel Aidoo
