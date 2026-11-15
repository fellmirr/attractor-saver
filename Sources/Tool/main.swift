// attractor-render: offline harness for tuning cameras/looks, making videos and benchmarking.
//
//   attractor-render --attractor lorenz[,thomas|all] --out out.png
//       [--size 1600x1000] [--particles N] [--palette dusk] [--seconds 4]
//       [--grow]                  start from a cloud (as the screensaver does) instead of settled
//       [--yaw 30 --pitch 10]     fixed camera      [--sheet yaw|pitch]  contact sheet of angles
//       [--zoom Z] [--decay D] [--head PX] [--bright B] [--bloom S] [--accumscale S] [--nolabel]
//       [--video out.mp4 --skip S --crf 20]   record --seconds of video after skipping S seconds
//       [--bench | --throughput]  GPU timing (per frame / at saturated clocks)
//       [--nodraw] [--nosim]      profiling: skip trail drawing / simulation
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
        if ["bench", "nolabel", "grow", "nodraw", "nosim", "throughput"].contains(k) { flags.insert(k) } else { args[k] = it.next() ?? "" }
    }
}

let device = MTLCreateSystemDefaultDevice()!
let size = (args["size"] ?? "1600x1000").split(separator: "x").compactMap { Int($0) }
guard size.count == 2, size[0] > 0, size[1] > 0 else { print("--size must look like 1600x1000"); exit(1) }
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
    let out: String = {
        guard names.count > 1 else { return outBase }
        let slug = Attractors.key(att.name).lowercased()
        let base = (outBase as NSString).deletingPathExtension, ext = (outBase as NSString).pathExtension
        return "\(base)-\(slug).\(ext.isEmpty ? "png" : ext)"
    }()
    let t0 = Date()
    // Without --grow, show the settled attractor: every particle warmed up, no restart.
    let r = try AttractorRenderer(device: device, outputFormat: .bgra8Unorm, particleCount: particleCount,
                                  attractor: att, palette: palette, growFromCloud: flags.contains("grow"))
    if let s = args["accumscale"].flatMap(Float.init) { r.accumScale = s }
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

    if let videoPath = args["video"] {
        // Render a video: skip `--skip` seconds (e.g. while the shape forms), then record
        // `--seconds` at 60 fps, piping raw frames into ffmpeg. Fades in/out at the ends.
        let skip = Float(args["skip"] ?? "") ?? 0
        let aspect = Float(W) / Float(H)
        r.resize(width: W, height: H)
        r.calibrateIfNeeded(aspect: aspect)
        if let (tex, lw, lh) = LabelMaker.make(device: device, attractor: r.attractor, screenHeight: H,
                                               accent: r.palette.stops[3]) {
            r.labelTexture = tex
            r.labelRectNDC = LabelMaker.rect(w: lw, h: lh, screenW: W, screenH: H)
        }
        for _ in 0..<Int(skip * 60) {
            let cb = r.queue.makeCommandBuffer()!
            r.encodeSimulation(cb, dt: 1 / 60, aspect: aspect)
            cb.commit()
        }
        let ffmpeg = Process()
        ffmpeg.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        ffmpeg.arguments = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
                            "-s", "\(W)x\(H)", "-r", "60", "-i", "-",
                            "-c:v", "libx264", "-preset", "slow", "-crf", args["crf"] ?? "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", videoPath]
        let pipe = Pipe()
        ffmpeg.standardInput = pipe
        signal(SIGPIPE, SIG_IGN)   // report ffmpeg failures instead of dying on a broken pipe
        do { try ffmpeg.run() } catch { print("could not start ffmpeg: \(error)"); exit(1) }
        let total = Int(seconds * 60)
        var bytes = [UInt8](repeating: 0, count: W * H * 4)
        for i in 0..<total {
            let t = Float(i) / 60
            let f = max(0, min(1, t / 1.0, (seconds - t) / 1.0))
            let fade = f * f * (3 - 2 * f)   // same smoothstep fade as the screensaver
            let labelT = max(0, min(1, (t - 1.5) / 2))
            r.labelAlpha = labelT * labelT * (3 - 2 * labelT)
            let cb = r.queue.makeCommandBuffer()!
            r.encodeSimulation(cb, dt: 1 / 60, aspect: aspect)
            r.encodeComposite(cb, target: target, fade: fade)
            cb.commit()
            cb.waitUntilCompleted()
            target.getBytes(&bytes, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
            pipe.fileHandleForWriting.write(Data(bytes))
        }
        try pipe.fileHandleForWriting.close()
        ffmpeg.waitUntilExit()
        guard ffmpeg.terminationStatus == 0 else { print("ffmpeg failed (\(ffmpeg.terminationStatus))"); exit(1) }
        print("\(att.name): wrote \(total) frames -> \(videoPath)")
        continue
    }
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
