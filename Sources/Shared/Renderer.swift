import Foundation
import Metal
import simd

struct Uniforms {
    var mvp = matrix_identity_float4x4
    var sim = SIMD4<Float>(repeating: 0)
    var misc = SIMD4<Float>(repeating: 0)
    var cam = SIMD4<Float>(repeating: 0)
    var seed = SIMD4<Float>(repeating: 0)
    var pal0 = SIMD4<Float>(repeating: 0)
    var pal1 = SIMD4<Float>(repeating: 0)
    var pal2 = SIMD4<Float>(repeating: 0)
    var pal3 = SIMD4<Float>(repeating: 0)
    var pal4 = SIMD4<Float>(repeating: 0)
    var bg = SIMD4<Float>(repeating: 0)
    var label = SIMD4<Float>(repeating: 0)
    var extra = SIMD4<Float>(repeating: 0)
    var head = SIMD4<Float>(repeating: 0)
    var view = SIMD4<Float>(repeating: 0)
}

enum RendererError: Error { case setup(String) }

final class AttractorRenderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let outputFormat: MTLPixelFormat
    let particleCount: Int
    let particles: MTLBuffer
    /// Intermediate points along each particle's path this frame (trailSegments + 1 per particle).
    let pathPoints: MTLBuffer
    static let trailSegments = 3
    /// Bytes per particle: one float4 (xyz position, w speed).
    static let particleStride = 16
    /// Trail/bloom buffers: packed 32-bit float colour. Half the memory traffic of rgba16Float
    /// with no visible difference (the black point hides the precision loss in faint tails).
    static let hdrFormat: MTLPixelFormat = .rg11b10Float

    let attractor: Attractor
    let palette: Palette
    private var seedPSO: MTLComputePipelineState!
    private var stepPSO: MTLComputePipelineState!
    private var linePSO: MTLRenderPipelineState!
    private var headPSO: MTLRenderPipelineState!
    /// Head sprite diameter in pixels at the attractor's centre depth (scaled per display).
    var headSize: Float = 3.2
    var headBrightness: Float = 0.9
    // Harness-only switches for profiling (attractor-render --nodraw / --nosim).
    var debugSkipDraw = false
    var debugSkipSim = false
    private var fadePSO: MTLRenderPipelineState!
    private var downPSO: MTLRenderPipelineState!
    private var compositePSO: MTLRenderPipelineState!
    private var labelPSO: MTLRenderPipelineState!

    private var accum: MTLTexture?
    private var bloom: [MTLTexture] = []
    var labelTexture: MTLTexture?
    var labelRectNDC = SIMD4<Float>(repeating: 0)
    var labelAlpha: Float = 0
    /// Output size the targets and label were last configured for.
    var configuredSize: (width: Int, height: Int)?

    /// Fraction of the output resolution used for the particle accumulation buffer.
    var accumScale: Float = 1.0
    /// Palette positions per second that colours slowly drift through.
    var colorDrift: Float = 0.005
    var bloomStrength: Float = 0.12

    // Calibrated per attractor
    private var center = SIMD3<Float>(repeating: 0)
    private var scale: Float = 1
    private var medianSpeed: Float = 1
    private var simTimePerFrame: Float = 0.01
    private var lineIntensity: Float = 1

    var time: Float = 0
    private var lastMVP = matrix_identity_float4x4
    private var lastCamDist: Float = 3
    private var frame: UInt32 = 0
    /// Normalised world units per second that a median particle travels.
    var flowSpeed: Float = 0.08
    /// Trail line width in pixels (at 1800 px tall; scales up on larger displays).
    var lineWidth: Float = 1.15
    /// Default camera orbit, degrees per second (a full turn every ~2.4 minutes).
    var orbitSpeed: Float = 2.5
    /// Camera overrides for exploring angles in the harness.
    var cameraOverride: (yaw: Float, pitch: Float)?

    init(device: MTLDevice, outputFormat: MTLPixelFormat, particleCount: Int,
         attractor: Attractor, palette: Palette, growFromCloud: Bool = true) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw RendererError.setup("queue") }
        queue = q
        self.outputFormat = outputFormat
        self.particleCount = particleCount
        self.growFromCloud = growFromCloud
        guard let b = device.makeBuffer(length: particleCount * Self.particleStride, options: .storageModePrivate) else {
            throw RendererError.setup("particle buffer")
        }
        particles = b
        guard let pb = device.makeBuffer(length: particleCount * (Self.trailSegments + 1) * 16,
                                         options: .storageModePrivate) else {
            throw RendererError.setup("path buffer")
        }
        pathPoints = pb
        self.attractor = attractor
        self.palette = palette
        try load()
    }

    // MARK: - Setup

    /// Compiles this attractor's pipelines, then warms it up to measure its size and speed.
    private func load() throws {
        let src = shaderTemplate
            .replacingOccurrences(of: "DERIV_BODY", with: attractor.deriv)
            .replacingOccurrences(of: "TRAIL_SEGS_VALUE", with: String(Self.trailSegments))
        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        let lib = try device.makeLibrary(source: src, options: opts)
        func fn(_ n: String) throws -> MTLFunction {
            guard let f = lib.makeFunction(name: n) else { throw RendererError.setup("missing \(n)") }
            return f
        }
        seedPSO = try device.makeComputePipelineState(function: try fn("seedParticles"))
        stepPSO = try device.makeComputePipelineState(function: try fn("stepParticles"))

        let hdr = Self.hdrFormat
        func pipe(_ v: String, _ f: String, _ fmt: MTLPixelFormat,
                  _ blend: ((MTLRenderPipelineColorAttachmentDescriptor) -> Void)? = nil) throws -> MTLRenderPipelineState {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = try fn(v)
            d.fragmentFunction = try fn(f)
            d.colorAttachments[0].pixelFormat = fmt
            if let blend { d.colorAttachments[0].isBlendingEnabled = true; blend(d.colorAttachments[0]) }
            return try device.makeRenderPipelineState(descriptor: d)
        }
        linePSO = try pipe("lineVertex", "lineFragment", hdr) {
            $0.rgbBlendOperation = .add; $0.alphaBlendOperation = .add
            $0.sourceRGBBlendFactor = .one; $0.destinationRGBBlendFactor = .one
            $0.sourceAlphaBlendFactor = .one; $0.destinationAlphaBlendFactor = .one
        }
        headPSO = try pipe("headVertex", "headFragment", outputFormat) {
            $0.sourceRGBBlendFactor = .one; $0.destinationRGBBlendFactor = .one
            $0.sourceAlphaBlendFactor = .zero; $0.destinationAlphaBlendFactor = .one
        }
        fadePSO = try pipe("fullscreenVertex", "fadeFragment", hdr) {
            $0.sourceRGBBlendFactor = .zero; $0.destinationRGBBlendFactor = .sourceAlpha
            $0.sourceAlphaBlendFactor = .zero; $0.destinationAlphaBlendFactor = .sourceAlpha
        }
        downPSO = try pipe("fullscreenVertex", "downsampleFragment", hdr)
        compositePSO = try pipe("fullscreenVertex", "compositeFragment", outputFormat)
        labelPSO = try pipe("labelVertex", "labelFragment", outputFormat) {
            $0.sourceRGBBlendFactor = .one; $0.destinationRGBBlendFactor = .oneMinusSourceAlpha
            $0.sourceAlphaBlendFactor = .one; $0.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }

        prewarmAndNormalise()
        pendingRestart = growFromCloud
    }

    private var needsCalibration = true
    private var pendingRestart = false
    /// After calibrating, restart from a random cloud so the viewer sees the shape emerge.
    /// When false (offline renders), every particle is warmed up and the settled shape is shown.
    let growFromCloud: Bool
    var particleBrightness: Float = 0.42

    private func baseUniforms() -> Uniforms {
        var u = Uniforms()
        u.cam = SIMD4(0, Float(particleCount), 0, Float(frame))
        u.seed = SIMD4(attractor.seed, 0)
        func lin(_ c: SIMD3<Float>) -> SIMD4<Float> { SIMD4(pow(c, SIMD3(repeating: 2.2)), 1) }
        u.pal0 = lin(palette.stops[0]); u.pal1 = lin(palette.stops[1]); u.pal2 = lin(palette.stops[2])
        u.pal3 = lin(palette.stops[3]); u.pal4 = lin(palette.stops[4])
        u.bg = lin(palette.background)
        return u
    }

    private func dispatchStep(_ enc: MTLComputeCommandEncoder, _ u: inout Uniforms, count: Int? = nil) {
        enc.setComputePipelineState(stepPSO)
        enc.setBuffer(particles, offset: 0, index: 0)
        enc.setBuffer(pathPoints, offset: 0, index: 2)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        let w = stepPSO.threadExecutionWidth * 4
        enc.dispatchThreads(MTLSize(width: count ?? particleCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
    }

    private func seed(at point: SIMD3<Float>, spread: Float) {
        var u = baseUniforms()
        u.seed = SIMD4(point, 0)
        u.cam.z = spread
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(seedPSO)
        enc.setBuffer(particles, offset: 0, index: 0)
        enc.setBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.dispatchThreads(MTLSize(width: particleCount, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
    }

    /// Scatter particles in a loose cloud around the attractor so the shape forms from t = 0.
    func restartFromCloud() {
        seed(at: center, spread: 0.45 / scale)
        clearAccum()
        time = 0
        sinceRestart = 0
    }

    /// Real seconds since particles were scattered; drives the time-lapse while the shape forms.
    private var sinceRestart: Float = 1000
    /// Simulation speed multiplier right after a restart, easing back to 1×.
    var formationBoost: Float = 4.5

    private func prewarmAndNormalise() {
        var u = baseUniforms()
        seed(at: attractor.seed, spread: attractor.spread)
        // Integrate for prewarmTime, capped at a fixed step budget.
        let maxSteps = 6000
        let steps = min(maxSteps, Int((attractor.prewarmTime / attractor.maxDt).rounded(.up)))
        let h = attractor.prewarmTime / Float(steps)
        // When the shape will grow from a fresh cloud anyway, the warm-up only needs enough
        // particles to measure the attractor, so it stays light on the GPU.
        let warmCount = growFromCloud ? min(particleCount, 8192) : particleCount
        // Diverging particles respawn next to another *warmed* particle, not a cold seed.
        u.cam.y = Float(warmCount)
        // Submit one small chunk at a time and wait for it: this may run in the background
        // while another attractor is on screen, and must never hold up its frames.
        let chunk = 100
        var done = 0
        while done < steps {
            let n = min(chunk, steps - done)
            let cb = queue.makeCommandBuffer()!
            let enc = cb.makeComputeCommandEncoder()!
            u.sim = SIMD4(h, Float(n), 0, 0)
            frame &+= 1
            u.cam.w = Float(frame)
            dispatchStep(enc, &u, count: warmCount)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            done += n
        }

        // Read back a sample to find the attractor's centre, size and typical speed.
        // One bulk copy (tens of thousands of tiny copies would stall the GPU for a frame or two).
        let bytes = warmCount * Self.particleStride
        let readback = device.makeBuffer(length: bytes, options: .storageModeShared)!
        let cb = queue.makeCommandBuffer()!
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: particles, sourceOffset: 0, to: readback, destinationOffset: 0, size: bytes)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        let sampleCount = min(warmCount, 65536)
        let stride = warmCount / sampleCount
        let all = readback.contents().bindMemory(to: SIMD4<Float>.self, capacity: warmCount)
        var xs: [Float] = [], ys: [Float] = [], zs: [Float] = [], sp: [Float] = []
        xs.reserveCapacity(sampleCount); ys.reserveCapacity(sampleCount); zs.reserveCapacity(sampleCount)
        for i in 0..<sampleCount {
            let p = all[i * stride]
            guard p.x.isFinite, p.y.isFinite, p.z.isFinite else { continue }
            xs.append(p.x); ys.append(p.y); zs.append(p.z); sp.append(p.w)
        }
        // Percentiles of an already-sorted array.
        func pct(_ s: [Float], _ q: Float) -> Float {
            s.isEmpty ? 0 : s[min(s.count - 1, Int(Float(s.count) * q))]
        }
        let sx = xs.sorted(), sy = ys.sorted(), sz = zs.sorted()
        center = SIMD3(0.5 * (pct(sx, 0.005) + pct(sx, 0.995)),
                       0.5 * (pct(sy, 0.005) + pct(sy, 0.995)),
                       0.5 * (pct(sz, 0.005) + pct(sz, 0.995)))
        var dists: [Float] = []
        dists.reserveCapacity(xs.count)
        for i in 0..<xs.count { dists.append(simd_length(SIMD3(xs[i], ys[i], zs[i]) - center)) }
        let radius = max(pct(dists.sorted(), 0.997), 1e-6)
        scale = 1 / radius
        medianSpeed = max(pct(sp.sorted(), 0.5), 1e-6)

        // Choose simulated time per frame so the median particle moves at `flowSpeed`.
        let simPerSecond = flowSpeed * attractor.speed / (medianSpeed * scale)
        simTimePerFrame = simPerSecond / 60
    }

    var debugDescription: String {
        String(format: "%@: center=(%.2f, %.2f, %.2f) radius=%.2f medSpeed=%.2f simT/frame=%.4f",
               attractor.name, center.x, center.y, center.z, 1 / scale, medianSpeed, simTimePerFrame)
    }

    // MARK: - Targets

    func resize(width: Int, height: Int) {
        let w = max(1, Int(Float(width) * accumScale)), h = max(1, Int(Float(height) * accumScale))
        if let a = accum, a.width == w, a.height == h { return }
        func tex(_ w: Int, _ h: Int) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: Self.hdrFormat, width: w, height: h, mipmapped: false)
            d.usage = [.renderTarget, .shaderRead]
            d.storageMode = .private
            return device.makeTexture(descriptor: d)!
        }
        accum = tex(w, h)
        bloom = [tex(max(1, w / 2), max(1, h / 2)), tex(max(1, w / 4), max(1, h / 4)), tex(max(1, w / 8), max(1, h / 8))]
        clearAccum()
        needsCalibration = true
    }

    func releaseTargets() {
        accum = nil
        bloom = []
        configuredSize = nil
    }

    /// Clears the trails (e.g. between contact-sheet cells in the harness).
    func clearTrails() { clearAccum() }

    private func clearAccum() {
        guard let accum else { return }
        let cb = queue.makeCommandBuffer()!
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = accum
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        rp.colorAttachments[0].storeAction = .store
        cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()
        cb.commit()
    }

    // MARK: - Camera

    private func cameraMatrices(aspect: Float) -> (simd_float4x4, Float) {
        let a = attractor
        // Calm continuous orbit (attractor-specific spin overrides the default rate).
        var yaw = a.yaw + (a.spin != 0 ? a.spin : orbitSpeed) * time
        var pitch = a.pitch + a.pitchSway * sin(time * 2 * .pi / 97)
        if let o = cameraOverride { yaw = o.yaw; pitch = o.pitch }
        let fovY: Float = 32 * .pi / 180
        // Fit the unit sphere to the smaller screen dimension.
        let fitHalf = aspect >= 1 ? tan(fovY / 2) : tan(fovY / 2) * aspect
        let dist = 1.08 / sin(atan(fitHalf)) / a.zoom * (1 + 0.045 * sin(time * 2 * .pi / 53))
        let yr = yaw * .pi / 180, pr = pitch * .pi / 180
        let eye = SIMD3<Float>(dist * cos(pr) * sin(yr), dist * sin(pr), dist * cos(pr) * cos(yr))
        let view = lookAt(eye: eye, target: .zero, up: SIMD3(0, 1, 0))
        let proj = perspective(fovY: fovY, aspect: aspect, near: 0.05, far: 50)

        // Model: centre, scale, then orient. `up` (if set) takes precedence over `axes`/`flip`.
        var orient = simd_float4x4(0)
        if let up = a.up {
            // Rotate so the attractor's `up` direction points along world Y.
            let y = simd_normalize(up)
            let helper: SIMD3<Float> = abs(y.x) < 0.9 ? [1, 0, 0] : [0, 0, 1]
            let x = simd_normalize(simd_cross(helper, y))
            let z = simd_cross(x, y)
            // Rows of the rotation are the new basis vectors.
            orient = simd_float4x4(rows: [SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(0, 0, 0, 1)])
        } else {
            // Map attractor axes to world axes (world Y is up on screen).
            let ax = [a.axes.0, a.axes.1, a.axes.2]
            for row in 0..<3 { orient[ax[row]][row] = a.flip[row] }
            orient[3][3] = 1
        }
        let m = orient * scaleMatrix(scale) * translate(-center)
        return (proj * view * m, dist)
    }

    // MARK: - Frame

    /// Advance the simulation and accumulate one frame of trails.
    func encodeSimulation(_ cb: MTLCommandBuffer, dt: Float, aspect: Float) {
        guard let accum else { return }
        let step = min(max(dt, 0), 1 / 15)
        time += step
        frame &+= 1
        var u = baseUniforms()
        sinceRestart += step
        let boost = 1 + (formationBoost - 1) * exp(-sinceRestart / 3)
        let rate = step * 60
        let frameSimTime = simTimePerFrame * rate * boost
        let n = max(Self.trailSegments, Int((frameSimTime / attractor.maxDt).rounded(.up)))
        let h = frameSimTime / Float(n)
        u.sim = SIMD4(h, Float(n), 1 / medianSpeed, lineIntensity)
        let (mvp, dist) = cameraMatrices(aspect: aspect)
        u.mvp = mvp
        u.cam.x = dist
        lastMVP = mvp
        lastCamDist = dist
        u.misc = SIMD4(time, pow(attractor.decay, rate), 0, 1)
        u.extra.z = colorDrift

        if !debugSkipSim {
            let ce = cb.makeComputeCommandEncoder()!
            dispatchStep(ce, &u)
            ce.endEncoding()
        }

        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = accum
        rp.colorAttachments[0].loadAction = .load
        rp.colorAttachments[0].storeAction = .store
        let re = cb.makeRenderCommandEncoder(descriptor: rp)!
        re.setRenderPipelineState(fadePSO)
        re.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        // Line width scales gently with resolution so it reads the same on any display.
        u.view = SIMD4(Float(accum.width), Float(accum.height),
                       lineWidth * 0.5 * max(1, Float(accum.height) / 1800), 0)
        re.setRenderPipelineState(linePSO)
        re.setVertexBuffer(particles, offset: 0, index: 0)
        re.setVertexBuffer(pathPoints, offset: 0, index: 2)
        re.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        re.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        if !debugSkipDraw {
            // Each trail segment is a 6-vertex quad.
            re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: particleCount * 6 * Self.trailSegments)
        }
        re.endEncoding()
    }

    /// Bloom + tonemap + label into `target`.
    func encodeComposite(_ cb: MTLCommandBuffer, target: MTLTexture, fade: Float) {
        guard let accum, bloom.count == 3 else { return }
        var u = baseUniforms()
        u.misc = SIMD4(time, 0, attractor.exposure, fade)
        u.label = labelRectNDC
        u.extra = SIMD4(labelAlpha, bloomStrength, colorDrift, 0)

        var src = accum
        for dst in bloom {
            let rp = MTLRenderPassDescriptor()
            rp.colorAttachments[0].texture = dst
            rp.colorAttachments[0].loadAction = .dontCare
            rp.colorAttachments[0].storeAction = .store
            let re = cb.makeRenderCommandEncoder(descriptor: rp)!
            re.setRenderPipelineState(downPSO)
            re.setFragmentTexture(src, index: 0)
            re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            re.endEncoding()
            src = dst
        }

        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = target
        rp.colorAttachments[0].loadAction = .dontCare
        rp.colorAttachments[0].storeAction = .store
        let re = cb.makeRenderCommandEncoder(descriptor: rp)!
        re.setRenderPipelineState(compositePSO)
        re.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        re.setFragmentTexture(accum, index: 0)
        re.setFragmentTexture(bloom[1], index: 1)
        re.setFragmentTexture(bloom[2], index: 2)
        re.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        if headSize > 0 {
            var hu = u
            hu.mvp = lastMVP
            hu.cam.x = lastCamDist
            hu.sim.w = lineIntensity
            let pxScale = Float(target.height) / 1800
            hu.head = SIMD4(headSize * pxScale, headBrightness, 0, 0)
            re.setRenderPipelineState(headPSO)
            re.setVertexBuffer(particles, offset: 0, index: 0)
            re.setVertexBytes(&hu, length: MemoryLayout<Uniforms>.stride, index: 1)
            re.drawPrimitives(type: .point, vertexStart: 0, vertexCount: particleCount)
        }
        if let labelTexture, labelAlpha > 0 {
            re.setRenderPipelineState(labelPSO)
            re.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            re.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
            re.setFragmentTexture(labelTexture, index: 0)
            re.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        re.endEncoding()
    }

    /// Sets per-particle energy for the current target size and, the first time after a load,
    /// restarts the particles from a cloud.
    func calibrateIfNeeded(aspect: Float) {
        guard needsCalibration, let accum else { return }
        needsCalibration = false
        // Fixed per-particle energy (scaled with resolution so streaks look the same on any
        // display): each particle reads as a distinct glowing streak, and only genuinely dense
        // regions build up into bright cores.
        lineIntensity = particleBrightness * sqrt(Float(accum.width * accum.height) / (2880 * 1800))
        if pendingRestart {
            pendingRestart = false
            restartFromCloud()
        }
    }
}

// MARK: - Matrix helpers

func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
    let y = 1 / tan(fovY / 2), x = y / aspect, z = far / (near - far)
    return simd_float4x4(columns: (SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, z, -1), SIMD4(0, 0, z * near, 0)))
}

func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let f = simd_normalize(target - eye)
    let s = simd_normalize(simd_cross(f, up))
    let u = simd_cross(s, f)
    return simd_float4x4(columns: (SIMD4(s.x, u.x, -f.x, 0), SIMD4(s.y, u.y, -f.y, 0),
                                   SIMD4(s.z, u.z, -f.z, 0),
                                   SIMD4(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)))
}

func translate(_ t: SIMD3<Float>) -> simd_float4x4 {
    var m = matrix_identity_float4x4
    m[3] = SIMD4(t, 1)
    return m
}

func scaleMatrix(_ s: Float) -> simd_float4x4 {
    simd_float4x4(diagonal: SIMD4(s, s, s, 1))
}
