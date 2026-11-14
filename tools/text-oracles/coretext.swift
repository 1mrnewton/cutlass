// Prints how Core Text sets a text in one face of a font file, to check
// shaping against on a Mac: each glyph as `id@x,y+advance` in font units,
// in the order it is drawn, and the face each run was set in.
//
//     swiftc -O tools/text-oracles/coretext.swift -o coretext
//     ./coretext "/System/Library/Fonts/Supplemental/Tamil MN.ttc" 0 U+0BA3,U+0B95
//     CUTLASS_CORETEXT=./coretext CUTLASS_SHAPE_CASES=cases.txt cargo test \
//         -p cutlass-text --lib -- --ignored coretext_agrees
//
// Core Text is the reference for faces made for Apple's layout engine,
// with `morx`, `kern` version 1 and `kerx` tables. HarfBuzz 14 reads the
// class pairs of `kerx` format 2 another way than these faces are built,
// so there the two differ and Core Text is right. For OpenType faces
// HarfBuzz is the reference: Core Text shapes those its own way.
import Foundation
import CoreText
let args = CommandLine.arguments
let url = URL(fileURLWithPath: args[1]) as CFURL
let index = Int(args[2]) ?? 0
guard let descs = CTFontManagerCreateFontDescriptorsFromURL(url) as? [CTFontDescriptor], index < descs.count else { print("no font"); exit(1) }
let probe = CTFontCreateWithFontDescriptor(descs[index], 100, nil)
let upem = CGFloat(CTFontGetUnitsPerEm(probe))
let font = CTFontCreateWithFontDescriptor(descs[index], upem, nil)
var text = ""
for part in args[3].split(separator: ",") {
    let hex = part.replacingOccurrences(of: "U+", with: "")
    if let v = UInt32(hex, radix: 16), let s = Unicode.Scalar(v) { text.unicodeScalars.append(s) }
}
let attr = NSAttributedString(string: text, attributes: [kCTFontAttributeName as NSAttributedString.Key: font, kCTLigatureAttributeName as NSAttributedString.Key: 1])
let line = CTLineCreateWithAttributedString(attr)
var out: [String] = []
for run in CTLineGetGlyphRuns(line) as! [CTRun] {
    let n = CTRunGetGlyphCount(run)
    var glyphs = [CGGlyph](repeating: 0, count: n)
    var pos = [CGPoint](repeating: .zero, count: n)
    var adv = [CGSize](repeating: .zero, count: n)
    CTRunGetGlyphs(run, CFRange(location: 0, length: 0), &glyphs)
    CTRunGetPositions(run, CFRange(location: 0, length: 0), &pos)
    CTRunGetAdvances(run, CFRange(location: 0, length: 0), &adv)
    let attrs = CTRunGetAttributes(run) as NSDictionary
    let used = attrs[kCTFontAttributeName as String] as! CTFont
    let name = CTFontCopyPostScriptName(used) as String
    for i in 0..<n { out.append("\(glyphs[i])@\(Int(pos[i].x.rounded())),\(Int(pos[i].y.rounded()))+\(Int(adv[i].width.rounded()))") }
    out.append("{\(name)}")
}
print(out.joined(separator: "|"))
