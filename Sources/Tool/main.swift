// attractor-render: offline harness for tuning cameras/looks and benchmarking.
//
//   attractor-render --attractor lorenz --out out.png [--size 1600x1000] [--particles 1048576]
//                    [--palette ember] [--seconds 4] [--yaw 30 --pitch 10]
//                    [--sheet yaw|pitch] [--bench]
import CoreGraphics
import Foundation
import ImageIO
import Metal
import UniformTypeIdentifiers

var args = [String: String]()
var flags = Set<String>()
do {
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        guard a.hasPrefix("--") else { continue }
        let k = String(a.dropFirst(2))
        if ["bench", "nolabel", "points", "nodraw", "nosim", "throughput"].contains(k) { flags.insert(k) } else { args[k] = it.next() ?? "" }
    }
}

let device = MTLCreateSystemDefaultDevice()!
let size = (args["size"] ?? "1600x1000").split(separator: "x").map { Int($0)! }
let (W, H) = (size[0], size[1])
let particleCount = Int(args["particles"] ?? "") ?? 49_152
let paletteName = args["palette"] ?? "ember"
let palette = Palette.all.first { $0.name.lowercased() == paletteName.lowercased() } ?? Palette.all[0]
let seconds = Float(args["seconds"] ?? "") ?? 3

func writePNG(_ tex: MTLTexture, _ path: String) {
    let w = tex.width, h = tex.height
    var bytes = [UInt8](repeating: 0, count: w * h * 4)
    tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
    // BGRA -> RGBA
    for i in stride(from: 0, to: bytes.count, by: 4) { bytes.swapAt(i, i + 2) }
    writeRGBA(bytes, w, h, path)
}

func writeRGBA(_ bytes: [UInt8], _ w: Int, _ h: Int, _ path: String) {
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let provider = CGDataProvider(data: Data(bytes) as CFData)!
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: cs,
                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                      decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

func makeTarget(_ w: Int, _ h: Int) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: w, height: h, mipmapped: false)
    d.usage = [.renderTarget, .shaderRead]
    d.storageMode = .shared
    return device.makeTexture(descriptor: d)!
}

func render(_ r: AttractorRenderer, target: MTLTexture, frames: Int, label: Bool) {
    let aspect = Float(target.width) / Float(target.height)
    r.resize(width: target.width, height: target.height)
    r.calibrateIfNeeded(aspect: aspect)
    if label, let (tex, lw, lh) = LabelMaker.make(device: device, attractor: r.attractor,
                                                   screenHeight: target.height, accent: r.palette.stops[3]) {
        r.labelTexture = tex
        r.labelRectNDC = LabelMaker.rect(w: lw, h: lh, screenW: target.width, screenH: target.height)
        r.labelAlpha = 1
    } else {
        r.labelAlpha = 0
    }
    for i in 0..<frames {
        let cb = r.queue.makeCommandBuffer()!
        r.encodeSimulation(cb, dt: 1 / 60, aspect: aspect)
        if i == frames - 1 { r.encodeComposite(cb, target: target, fade: 1) }
        cb.commit()
        if i == frames - 1 { cb.waitUntilCompleted() }
    }
}

let names = (args["attractor"] ?? "lorenz") == "all" ? Attractors.all.map(\.name) : (args["attractor"] ?? "lorenz").split(separator: ",").map(String.init)
let outBase = args["out"] ?? "out.png"

for name in names {
    guard var att = Attractors.named(name) else { print("unknown attractor \(name)"); exit(1) }
    if let z = args["zoom"].flatMap(Float.init) { att.zoom = z }
    if let d = args["decay"].flatMap(Float.init) { att.decay = d }
    let out = names.count > 1 ? outBase.replacingOccurrences(of: ".png", with: "-\(att.name.lowercased().replacingOccurrences(of: "–", with: "-")).png") : outBase
    let t0 = Date()
    let r = try AttractorRenderer(device: device, outputFormat: .bgra8Unorm, particleCount: particleCount,
                                  attractor: att, palette: palette)
    if let s = args["accumscale"].flatMap(Float.init) { r.accumScale = s }
    r.usePoints = flags.contains("points")
    if args["grow"] == nil {
        // Show the settled attractor: warm up every particle, no restart from a cloud.
        r.growFromCloud = false
        try r.load(attractor: att, palette: palette)
    }
    if let h = args["head"].flatMap(Float.init) { r.headSize = h }
    if let b = args["bright"].flatMap(Float.init) { r.particleBrightness = b }
    r.debugSkipDraw = flags.contains("nodraw")
    r.debugSkipSim = flags.contains("nosim")
    if let s = args["bloom"].flatMap(Float.init) { r.bloomStrength = s }
    let loadTime = Date().timeIntervalSince(t0)

    if let sheet = args["sheet"] {
        // Contact sheet of camera angles.
        let values: [Float] = sheet == "yaw" ? stride(from: 0, to: 360, by: 30).map(Float.init)
                                              : [-60, -40, -20, 0, 20, 40, 60, 80, 90, -90, 10, 30]
        let fixed = Float(args[sheet == "yaw" ? "pitch" : "yaw"] ?? "") ?? (sheet == "yaw" ? att.pitch : att.yaw)
        let cols = 4, rows = 3
        let cw = W / cols, ch = H / rows
        let cell = makeTarget(cw, ch)
        var big = [UInt8](repeating: 0, count: W * H * 4)
        for (i, v) in values.enumerated() {
            r.cameraOverride = sheet == "yaw" ? (v, fixed) : (fixed, v)
            render(r, target: cell, frames: 40, label: false)
            var bytes = [UInt8](repeating: 0, count: cw * ch * 4)
            cell.getBytes(&bytes, bytesPerRow: cw * 4, from: MTLRegionMake2D(0, 0, cw, ch), mipmapLevel: 0)
            let ox = (i % cols) * cw, oy = (i / cols) * ch
            for y in 0..<ch {
                for x in 0..<cw {
                    let s = (y * cw + x) * 4, d = ((oy + y) * W + ox + x) * 4
                    // grid lines + BGRA->RGBA
                    let edge = x == 0 || y == 0
                    big[d] = edge ? 60 : bytes[s + 2]; big[d + 1] = edge ? 60 : bytes[s + 1]; big[d + 2] = edge ? 60 : bytes[s]; big[d + 3] = 255
                }
            }
        }
        writeRGBA(big, W, H, out)
        print("\(att.name) sheet \(sheet) [\(values.map { Int($0) })] fixed=\(fixed) -> \(out)")
        continue
    }

    if let y = args["yaw"].flatMap(Float.init) { r.cameraOverride = (y, Float(args["pitch"] ?? "") ?? att.pitch) }
    let target = makeTarget(W, H)

    if flags.contains("throughput") {
        // Saturate the GPU (so clocks stay high and steady) and measure average cost per frame.
        render(r, target: target, frames: 30, label: true)
        let n = 600
        let t0 = Date()
        var last: MTLCommandBuffer?
        for _ in 0..<n {
            let cb = r.queue.makeCommandBuffer()!
            r.encodeSimulation(cb, dt: 1 / 60, aspect: Float(W) / Float(H))
            r.encodeComposite(cb, target: target, fade: 1)
            cb.commit()
            last = cb
        }
        last?.waitUntilCompleted()
        let ms = Date().timeIntervalSince(t0) / Double(n) * 1000
        print(String(format: "%@ %dx%d: %.3f ms/frame at full clocks", att.name, W, H, ms))
        continue
    }
    if flags.contains("bench") {
        render(r, target: target, frames: 30, label: true)
        var gpu: [Double] = []
        let n = 300
        let sem = DispatchSemaphore(value: 0)
        let lock = NSLock()
        for _ in 0..<n {
            let cb = r.queue.makeCommandBuffer()!
            r.encodeSimulation(cb, dt: 1 / 60, aspect: Float(W) / Float(H))
            r.encodeComposite(cb, target: target, fade: 1)
            cb.addCompletedHandler { cb in
                lock.lock(); gpu.append(cb.gpuEndTime - cb.gpuStartTime); lock.unlock()
                sem.signal()
            }
            cb.commit()
            cb.waitUntilCompleted()
        }
        for _ in 0..<n { sem.wait() }
        gpu.sort()
        let med = gpu[gpu.count / 2] * 1000, p95 = gpu[Int(Double(gpu.count) * 0.95)] * 1000
        print(String(format: "%@ %dx%d particles=%d: GPU median %.2f ms, p95 %.2f ms  => ~%.0f%% of a 60 fps frame budget (load %.2fs)",
                     att.name, W, H, particleCount, med, p95, med / 16.667 * 100, loadTime))
        continue
    }

    render(r, target: target, frames: Int(seconds * 60), label: !flags.contains("nolabel"))
    writePNG(target, out)
    print(r.debugDescription + String(format: " load=%.2fs -> %@", loadTime, out))
}
