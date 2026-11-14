#!/usr/bin/env python3
"""Writes shaping cases over this Mac's system fonts, to check against HarfBuzz.

    python3 tools/text-oracles/shape_cases.py cases.txt          # Latin, Arabic, Hebrew
    python3 tools/text-oracles/shape_cases.py cases.txt indic    # Devanagari and its sisters
    CUTLASS_SHAPE_CASES=cases.txt cargo test -p cutlass-text --lib -- \
        --ignored harfbuzz_agrees

Each case is a font file, a face index, a script, a direction and a text,
tab separated. The test shapes each with `hb-shape` and with ours and
compares glyphs, clusters, advances and offsets. HarfBuzz shapes a font
that has Apple's own layout tables (Geeza Pro, Helvetica, Futura) through
those, which ours does not read, and sets one font's glyph classes aside
(Tahoma): those cases differ and the rest agree.
"""
import sys
SUP="/System/Library/Fonts/Supplemental/"; SYS="/System/Library/Fonts/"
def esc(t): return "".join(c if 32 <= ord(c) < 127 and c not in "\\" else "\\u{%X}" % ord(c) for c in t)
latin_fonts=[(SUP+"Arial.ttf",0),(SUP+"Times New Roman.ttf",0),(SUP+"Tahoma.ttf",0),(SUP+"Verdana.ttf",0),(SUP+"Georgia.ttf",0),(SUP+"Courier New.ttf",0),(SYS+"Avenir Next.ttc",0),(SYS+"Avenir Next.ttc",7),(SUP+"SnellRoundhand.ttc",0),(SYS+"SFNS.ttf",0),(SYS+"Helvetica.ttc",0),(SYS+"HelveticaNeue.ttc",0),(SUP+"Futura.ttc",0),(SYS+"Noteworthy.ttc",0),(SUP+"Arial Unicode.ttf",0)]
latin=["Hello, World!","AVATAR Type WAVE To. Yo, P.A.","office fjord fluffy affine fi fl ffi","café naïve éà Å","éàüñ ấ ǖ","Te V. 1/2 3-4 (x) [y] {z}","á̂̃ ộ","The quick brown fox jumps over the lazy dog 0123456789","АВТО Ударе́ние","Αθήνα άλφα","fi‌fi f‍i ­x","don’t “quote” — dash…","A B C D","naïve ﬁ Ĳ ſ"]
cases=[]
for f,i in latin_fonts:
    for t in latin: cases.append((f,i,"Latn","ltr",t))
    cases.append((f,i,"Latn","rtl","Hello (World)!"))
    cases.append((f,i,"Latn","rtl","fi <a> [b]"))
arabic_fonts=[(SUP+"Arial.ttf",0),(SUP+"Times New Roman.ttf",0),(SUP+"Tahoma.ttf",0),(SUP+"Courier New.ttf",0),(SYS+"SFArabic.ttf",0),(SYS+"NotoNastaliq.ttc",0),(SUP+"Arial Unicode.ttf",0),(SYS+"GeezaPro.ttc",0)]
arabic=["مرحبا بالعالم","السلام عليكم","بِسْمِ اللَّهِ الرَّحْمَٰنِ","لا لأ لإ لآ بلا","بــب ب‍ ‍ب ب‌ب","محمد ١٢٣ عربي","اردو کی ہے پاکستان","فارسی گچپژ","لَّ لَّ بُّ","(عربي) [ب]","شَدَّةٌ كِتَابٌ"]
for f,i in arabic_fonts:
    for t in arabic: cases.append((f,i,"Arab","rtl",t))
    cases.append((f,i,"Arab","ltr","١٢٣ 123"))
hebrew_fonts=[(SUP+"Arial.ttf",0),(SUP+"Times New Roman.ttf",0),(SUP+"Tahoma.ttf",0),(SUP+"Courier New.ttf",0),(SUP+"Arial Unicode.ttf",0),(SYS+"ArialHB.ttc",0)]
hebrew=["שלום עולם","שָׁלוֹם בְּרֵאשִׁית","(א) [ב] <ג>","אַבָּ כֹּל"]
for f,i in hebrew_fonts:
    for t in hebrew: cases.append((f,i,"Hebr","rtl",t))
if len(sys.argv)>2 and sys.argv[2]=="indic":
    cases=[]
    deva_fonts=[(SYS+"Kohinoor.ttc",0),(SUP+"ITFDevanagari.ttc",0),(SUP+"Arial Unicode.ttf",0),(SUP+"Shree714.ttc",0)]
    deva=["नमस्ते दुनिया","हिन्दी भाषा","क्षत्रिय ज्ञान","कर्म धर्म पूर्ण","प्रेम क्रम त्र","कि की कु कू के कै को कौ कं कः","स्त्री श्री द्वारा","क्‍ष क्‌ष र्‍क","हूँ में नहीं","क़ ख़ ड़ ढ़ ड़","उत्तर प्रदेश विद्यालय","शक्ति संस्कृत दृष्टि","रु रू हृ द्द ट्ट","अआइईउऊएऐओऔ १२३","र्कि र्त्य क्रि","िक ्क क्"]
    for f,i in deva_fonts:
        for t in deva: cases.append((f,i,"Deva","ltr",t))
    for f,i,s,ts in [(SYS+"KohinoorBangla.ttc",0,"Beng",["বাংলা ভাষা","ক্ষ স্ত্রী কি কে কো কৌ","প্রথম র্য য়"]),
                     (SYS+"KohinoorGujarati.ttc",0,"Gujr",["ગુજરાતી ભાષા","ક્ષ સ્ત્રી કિ ર્ક"]),
                     (SYS+"KohinoorTelugu.ttc",0,"Telu",["తెలుగు భాష","క్ష స్త్రీ కి"]),
                     (SUP+"Tamil Sangam MN.ttc",0,"Taml",["தமிழ் மொழி","கொ கோ கௌ கை ஸ்ரீ"]),
                     (SYS+"NotoSansKannada.ttc",0,"Knda",["ಕನ್ನಡ ಭಾಷೆ","ಕ್ಷ ಸ್ತ್ರೀ ಕಿ ಕೊ"]),
                     (SYS+"NotoSansOriya.ttc",0,"Orya",["ଓଡ଼ିଆ ଭାଷା","କ୍ଷ କି କୋ"]),
                     (SUP+"Gurmukhi MN.ttc",0,"Guru",["ਪੰਜਾਬੀ ਭਾਸ਼ਾ"])]:
        for t in ts: cases.append((f,i,s,"ltr",t))
with open(sys.argv[1],"w") as out:
    for f,i,s,d,t in cases: out.write(f"{f}\t{i}\t{s}\t{d}\t{esc(t)}\n")
print(len(cases))
