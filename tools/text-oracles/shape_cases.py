#!/usr/bin/env python3
"""Writes shaping cases over this Mac's system fonts, to check against HarfBuzz.

    python3 tools/text-oracles/shape_cases.py cases.txt          # Latin, Arabic, Hebrew
    python3 tools/text-oracles/shape_cases.py cases.txt indic    # Devanagari and its sisters
    python3 tools/text-oracles/shape_cases.py cases.txt khmer
    python3 tools/text-oracles/shape_cases.py cases.txt myanmar
    python3 tools/text-oracles/shape_cases.py cases.txt universal          # every script below
    python3 tools/text-oracles/shape_cases.py cases.txt universal Java     # one of them
    python3 tools/text-oracles/shape_cases.py cases.txt joined             # Mongolian, N'Ko and others
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
if len(sys.argv)>2 and sys.argv[2]=="khmer":
    # Words, then the hard clusters: vowels written before and in two
    # parts, stacks of subscripts with the subscript ro, shifters, a non
    # joiner, and clusters with no letter to stand on.
    cases=[]
    khmer=["សួស្តី","កម្ពុជា","ភាសាខ្មែរ","ខ្ញុំ","ស្រឡាញ់","ព្រះរាជាណាចក្រ","ក្រុង ស្ត្រី","សង្គ្រាម បង្អែម",
           "ញ៉ាំ ប៊ី ស៊ី អ៊ំ","សំឡេង កើត","កៀ កឿ កោ កៅ កេ កែ កៃ","កា់ កះ កំ កាំ","ង្រ្គ ង្គ្រ","ក្រ្ក ក្ក្រ ក្រ្រ",
           "េក ្ក ា ់ ៉","ក‌្ក ក្‌ក ក‍ ក្‍ក","១២៣ ៖ ។ ៛","◌ា ◌្ក ◌េ","កាា កេេ កុុ","ក្ក្ក្ក ស្ត្រ្យ",
           "ឥ ឧ ឱ្យ ឲ្យ ឬ","ប្រទេស កម្ពុជា រាជធានី ភ្នំពេញ","ខ្មែរ​អង់គ្លេស","abc កខគ 12"]
    for f,i in [(SUP+"Khmer MN.ttc",0),(SUP+"Khmer MN.ttc",1),(SUP+"Khmer Sangam MN.ttf",0)]:
        for t in khmer: cases.append((f,i,"Khmr","ltr",t))
if len(sys.argv)>2 and sys.argv[2]=="myanmar":
    # Burmese words, then Shan, Mon and Karen letters, then the hard
    # clusters: the kinzi, stacks, medials and vowels written before, an
    # anusvara after a vowel below, joiners, selectors, and clusters with
    # no letter. The MN faces shape by their own machines, as in HarfBuzz.
    cases=[]
    myanmar=["မြန်မာ","မင်္ဂလာပါ","ကျွန်တော်","ကြောင် ကြွေ","ပြည်ထောင်စု နိုင်ငံ","သမ္မတ ဗုဒ္ဓ တက္ကသိုလ်","အင်္ဂလိပ် သင်္ဘော",
             "ဦး ဩ ဪ ၎င်း ဣ ဤ","ကျေးဇူးတင်ပါတယ်","ရွှေ လှိုင် ငြိမ်း စွဲ ဆုံး ဉာဏ်","၁၂၃၄ ၊ ။","ေ ္က ် ြ ံ့",
             "ၽူႈ မႂ်ႇ ၵႂႃႇ လိၵ်ႈ","ၚဳ ဗွဲ ကၠံ ၜါ","က်‌က က်‍ က္‌က","ကုံ ကုံး ကိုံ့","က︀ ေ︀က ကေ︀ႄ","◌ေ ◌် ◌ျ",
             "ႇႇ ံံ ့့","က² ၁ာ က်²","င်္က္က င်္ကြ င်္ကျွေ","ကျြွှ ကြွှေါ်","ဿ ၏ ၍ ၌","abc ကခဂ 12"]
    for f,i in [(SYS+"NotoSansMyanmar.ttc",0),(SYS+"NotoSansMyanmar.ttc",4),(SYS+"NotoSerifMyanmar.ttc",0),
                (SUP+"Myanmar MN.ttc",0),(SUP+"Myanmar Sangam MN.ttc",0)]:
        for t in myanmar: cases.append((f,i,"Mymr","ltr",t))
if len(sys.argv)>2 and sys.argv[2] in ("universal","joined"):
    # The scripts shaped by the shared rules, each over the face this Mac
    # has for it. The texts are made from the script's own characters:
    # each letter, each mark on a letter, marks in pairs, letters stacked
    # with each virama and each stacker, marks and viramas with no letter,
    # joiners and non-joiners, and the script beside Latin and digits. The
    # Tibetan and Sinhala faces here shape by their own machines, and the
    # Tai Tham and Syloti Nagri faces name only a default script, so those
    # are shaped as before; they are here to show nothing moved.
    import unicodedata
    cases=[]
    N="NotoSans"; R="-Regular.ttf"; O="-Regular.otf"
    scripts=[("Tibt",SUP+"Kailasa.ttc",[(0xF00,0xFFF)]),("Tibt",SUP+"Kokonor.ttf",[(0xF00,0xFFF)]),
        ("Sinh",SUP+"Sinhala MN.ttc",[(0xD80,0xDFF)]),("Sinh",SUP+"Sinhala Sangam MN.ttc",[(0xD80,0xDFF)]),
        ("Batk",SUP+N+"Batak"+R,[(0x1BC0,0x1BFF)]),("Bhks",SUP+N+"Bhaiksuki"+R,[(0x11C00,0x11C6F)]),
        ("Bugi",SUP+N+"Buginese"+R,[(0x1A00,0x1A1F)]),("Buhd",SUP+N+"Buhid"+R,[(0x1740,0x175F)]),
        ("Cakm",SUP+N+"Chakma"+R,[(0x11100,0x1114F)]),("Cham",SUP+N+"Cham"+R,[(0xAA00,0xAA5F)]),
        ("Dupl",SUP+N+"Duployan"+R,[(0x1BC00,0x1BC9F)]),("Egyp",SUP+N+"EgyptianHieroglyphs"+R,[(0x13000,0x1345F)]),
        ("Gong",SUP+N+"GunjalaGondi"+O,[(0x11D60,0x11DAF)]),("Gonm",SUP+N+"MasaramGondi"+O,[(0x11D00,0x11D5F)]),
        ("Hano",SUP+N+"Hanunoo"+R,[(0x1720,0x173F)]),("Java",SUP+N+"Javanese"+O,[(0xA980,0xA9DF)]),
        ("Kthi",SUP+N+"Kaithi"+R,[(0x11080,0x110CF)]),("Kali",SUP+N+"KayahLi"+R,[(0xA900,0xA92F)]),
        ("Khar",SUP+N+"Kharoshthi"+R,[(0x10A00,0x10A5F)]),("Khoj",SUP+N+"Khojki"+R,[(0x11200,0x1124F)]),
        ("Sind",SUP+N+"Khudawadi"+R,[(0x112B0,0x112FF)]),("Lepc",SUP+N+"Lepcha"+R,[(0x1C00,0x1C4F)]),
        ("Limb",SUP+N+"Limbu"+R,[(0x1900,0x194F)]),("Mahj",SUP+N+"Mahajani"+R,[(0x11150,0x1117F)]),
        ("Marc",SUP+N+"Marchen"+R,[(0x11C70,0x11CBF)]),("Mtei",SUP+N+"MeeteiMayek"+R,[(0xABC0,0xABFF),(0xAAE0,0xAAFF)]),
        ("Plrd",SUP+N+"Miao"+R,[(0x16F00,0x16F9F)]),("Modi",SUP+N+"Modi"+R,[(0x11600,0x1165F)]),
        ("Mult",SUP+N+"Multani"+R,[(0x11280,0x112AF)]),("Nagm",SUP+N+"NagMundari"+R,[(0x1E4D0,0x1E4FF)]),
        ("Newa",SUP+N+"Newa"+R,[(0x11400,0x1147F)]),("Hmng",SUP+N+"PahawhHmong"+R,[(0x16B00,0x16B8F)]),
        ("Rjng",SUP+N+"Rejang"+R,[(0xA930,0xA95F)]),("Saur",SUP+N+"Saurashtra"+R,[(0xA880,0xA8DF)]),
        ("Shrd",SUP+N+"Sharada"+R,[(0x11180,0x111DF)]),("Sidd",SUP+N+"Siddham"+O,[(0x11580,0x115FF)]),
        ("Sund",SUP+N+"Sundanese"+R,[(0x1B80,0x1BBF),(0x1CC0,0x1CCF)]),("Sunu",SUP+N+"Sunuwar"+O,[(0x11BC0,0x11BFF)]),
        ("Sylo",SUP+N+"SylotiNagri"+R,[(0xA800,0xA82F)]),("Tglg",SUP+N+"Tagalog"+R,[(0x1700,0x171F)]),
        ("Tagb",SUP+N+"Tagbanwa"+R,[(0x1760,0x177F)]),("Tale",SUP+N+"TaiLe"+R,[(0x1950,0x197F)]),
        ("Lana",SUP+N+"TaiTham"+R,[(0x1A20,0x1AAF)]),("Tavt",SUP+N+"TaiViet"+R,[(0xAA80,0xAADF)]),
        ("Takr",SUP+N+"Takri"+R,[(0x11680,0x116CF)]),("Tfng",SUP+N+"Tifinagh"+O,[(0x2D30,0x2D7F)]),
        ("Tirh",SUP+N+"Tirhuta"+R,[(0x11480,0x114DF)]),("Wcho",SUP+N+"Wancho"+R,[(0x1E2C0,0x1E2FF)]),
        ("Yezi",SUP+"NotoSerifYezidi"+O,[(0x10E80,0x10EBF)]),("Ahom",SUP+"NotoSerifAhom"+R,[(0x11700,0x1174F)]),
        ("Bali",SUP+"NotoSerifBalinese"+R,[(0x1B00,0x1B7F)]),
        ("Hmnp",SUP+"NotoSerifNyiakengPuachueHmong"+R,[(0x1E100,0x1E14F)])]
    # The scripts HarfBuzz shapes by the same rules and joins as Arabic
    # besides. Ours joins them and does not yet cut them into clusters,
    # so these show how far that is from HarfBuzz.
    if sys.argv[2]=="joined":
        scripts=[("Mong",SUP+N+"Mongolian"+R,[(0x1800,0x18AF)]),("Phag",SUP+N+"PhagsPa"+R,[(0xA840,0xA87F)]),
            ("Nkoo",SUP+N+"NKo"+R,[(0x7C0,0x7FF)]),("Mand",SUP+N+"Mandaic"+R,[(0x840,0x85F)]),
            ("Mani",SUP+N+"Manichaean"+R,[(0x10AC0,0x10AFF)]),("Phlp",SUP+N+"PsalterPahlavi"+R,[(0x10B80,0x10BAF)]),
            ("Adlm",SUP+N+"Adlam"+R,[(0x1E900,0x1E95F)]),("Rohg",SUP+N+"HanifiRohingya"+R,[(0x10D00,0x10D3F)])]
    def rows(words,size=6): return [" ".join(words[i:i+size]) for i in range(0,len(words),size)]
    for code,face,blocks in scripts:
        if len(sys.argv)>3 and sys.argv[3]!=code: continue
        every=[chr(u) for a,b in blocks for u in range(a,b+1)]
        kind={c:unicodedata.category(c) for c in every}
        marks=[c for c in every if kind[c] in ("Mn","Mc","Me")]
        # A virama, or a sign that stacks the next letter under this one.
        stackers=[c for c in marks if unicodedata.combining(c)==9]
        marks=[c for c in marks if c not in stackers]
        letters=[c for c in every if kind[c] in ("Lo","Lm")]
        others=[c for c in every if kind[c] in ("Nd","No","Nl","Po","So")]
        if not letters: letters=[c for c in every if kind[c]=="Cn"][:24]
        texts=["".join(letters[i:i+8]) for i in range(0,min(len(letters),64),8)]
        texts+=["".join(others[i:i+8]) for i in range(0,min(len(others),32),8)]
        if not letters: continue
        a=letters[0]; b=letters[min(1,len(letters)-1)]; c=letters[min(4,len(letters)-1)]
        texts+=rows([a+m for m in marks])+rows([c+m for m in marks])
        texts+=rows([m for m in marks])+rows([m+a for m in marks])
        near=[(m,n) for i,m in enumerate(marks) for n in marks[i+1:i+4]]
        far=[(m,n) for i,m in enumerate(marks) for n in marks[i+7::9]]
        texts+=rows([a+m+n for m,n in near])+rows([a+n+m for m,n in near])+rows([a+m+n for m,n in far])
        texts+=rows([a+m+m for m in marks])
        # A letter before another: some are written as a hook over it.
        texts+=rows([l+a for l in letters[:48]])+rows([l+a+m for l in letters[:48] for m in marks[:1]])
        for h in stackers:
            texts+=rows([a+h+l for l in letters[:48]])+rows([l+h+a for l in letters[:48]])
            texts+=rows([a+h+b+m for m in marks])+rows([a+m+h for m in marks])
            texts+=[a+h, h, h+a, a+h+h+b, a+h+b+h+c, a+h+b+h+c+h+a, a+h+" "+b]
            texts+=[a+h+"\u200D"+b, a+h+"\u200C"+b, a+"\u200D"+h+b, a+"\u200C"+h+b, a+h+"\u200D", a+h+"\u200C"]
            texts+=rows([a+h+b+m+n for m,n in near[:36]])
        if marks:
            m=marks[0]; n=marks[-1]
            texts+=[a+"\u200C"+m, a+"\u200D"+m, a+m+"\u200C"+n, a+m+"\u200D"+n, "\u200C"+m, "\u25CC"+m+" \u25CC"+n,
                    "\u00A0"+m+" \u00A0"+n, a+"\uFE00"+m, a+m+"\u034F"+n, "x"+m+" 1"+n]
        texts+=["\u200C", a+"\u200C", a+"\u200C"+b, a+"\u200D"+b, "abc "+a+b+c+" 12", a+" "+b+"."+c+","+a+"-"+b,
                (a+b+c)*4]
        way="rtl" if code in ("Khar","Yezi","Nkoo","Mand","Mani","Phlp","Adlm","Rohg") else "ltr"
        for t in texts:
            if t: cases.append((face,0,code,way,t))
with open(sys.argv[1],"w") as out:
    for f,i,s,d,t in cases: out.write(f"{f}\t{i}\t{s}\t{d}\t{esc(t)}\n")
print(len(cases))
