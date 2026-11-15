import CoreGraphics
import CoreText
import Foundation
import Metal

enum LabelMaker {
    /// Renders the attractor's name (letter-spaced small caps) and its equations
    /// (small italic serif) into a premultiplied RGBA texture.
    /// Returns the texture and its size in pixels.
    static func make(device: MTLDevice, attractor: Attractor, screenHeight: Int,
                     accent: SIMD3<Float>) -> (MTLTexture, Int, Int)? {
        let unit = CGFloat(screenHeight) / 1000   // 1 unit = 0.1% of screen height
        let nameSize = max(9, 15 * unit)
        let eqSize = max(7, 11.5 * unit)

        func font(_ names: [String], _ size: CGFloat) -> CTFont {
            for n in names {
                let f = CTFontCreateWithName(n as CFString, size, nil)
                if (CTFontCopyPostScriptName(f) as String) == n { return f }
            }
            return CTFontCreateWithName("HelveticaNeue-Light" as CFString, size, nil)
        }
        let nameFont = font(["AvenirNext-Regular", "HelveticaNeue"], nameSize)
        let eqFont = font(["Baskerville-Italic", "Didot-Italic", "Georgia-Italic"], eqSize)

        let white = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.78)
        let soft = CGColor(srgbRed: CGFloat(0.55 + 0.45 * accent.x), green: CGFloat(0.55 + 0.45 * accent.y),
                           blue: CGFloat(0.55 + 0.45 * accent.z), alpha: 0.62)
        let dim = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.32)

        func line(_ s: String, _ f: CTFont, _ c: CGColor, kern: CGFloat) -> CTLine {
            let attrs: [NSAttributedString.Key: Any] = [
                NSAttributedString.Key(kCTFontAttributeName as String): f,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): c,
                NSAttributedString.Key(kCTKernAttributeName as String): kern,
            ]
            return CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs))
        }

        let title = line(attractor.name.uppercased(), nameFont, white, kern: nameSize * 0.42)
        let sub = line("ATTRACTOR", nameFont, dim, kern: nameSize * 0.42)
        let eqs = attractor.equations.map { line($0, eqFont, soft, kern: eqSize * 0.04) }

        func width(_ l: CTLine) -> CGFloat { CGFloat(CTLineGetTypographicBounds(l, nil, nil, nil)) }
        let titleW = width(title)
        let subW = width(sub)
        let eqW = eqs.map(width).max() ?? 0
        let nameGap = nameSize * 1.0
        let eqLead = eqSize * 1.55
        let ruleGap = nameSize * 1.25

        let pad = 4 * unit
        let contentW = max(titleW + nameGap + subW, eqW)
        let w = Int((contentW + 2 * pad).rounded(.up))
        let h = Int((nameSize * 1.3 + ruleGap + CGFloat(eqs.count) * eqLead + 2 * pad).rounded(.up))

        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.setAllowsFontSmoothing(false)
        ctx.setShouldAntialias(true)

        // CoreGraphics origin is bottom-left. Equations at the bottom, name on top.
        var y = pad + eqSize * 0.3
        for l in eqs.reversed() {
            ctx.textPosition = CGPoint(x: pad, y: y)
            CTLineDraw(l, ctx)
            y += eqLead
        }
        // Thin rule between equations and name
        y += ruleGap * 0.1
        ctx.setFillColor(CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.18))
        ctx.fill(CGRect(x: pad, y: y, width: contentW * 0.18, height: max(1, unit)))
        y += ruleGap * 0.9
        ctx.textPosition = CGPoint(x: pad, y: y)
        CTLineDraw(title, ctx)
        ctx.textPosition = CGPoint(x: pad + titleW + nameGap, y: y)
        CTLineDraw(sub, ctx)

        guard let data = ctx.data else { return nil }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = .shaderRead
        guard let tex = device.makeTexture(descriptor: d) else { return nil }
        // CGContext memory is top row first, which matches Metal's texture origin.
        tex.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: data, bytesPerRow: w * 4)
        return (tex, w, h)
    }

    /// NDC rectangle for a label of `w`x`h` pixels in the bottom-left corner.
    static func rect(w: Int, h: Int, screenW: Int, screenH: Int) -> SIMD4<Float> {
        let margin = Float(screenH) * 0.045
        let x0 = margin, y0 = margin
        let toX = { (px: Float) in px / Float(screenW) * 2 - 1 }
        let toY = { (px: Float) in px / Float(screenH) * 2 - 1 }
        return SIMD4(toX(x0), toY(y0), toX(x0 + Float(w)), toY(y0 + Float(h)))
    }
}
