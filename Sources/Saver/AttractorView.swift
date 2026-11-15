import AppKit
import Metal
import QuartzCore
import ScreenSaver

@objc(AttractorSaverView)
final class AttractorSaverView: ScreenSaverView {
    /// Seconds each attractor is shown before fading to the next.
    private static let cycleSeconds: Double =
        ProcessInfo.processInfo.environment["ATTRACTOR_CYCLE"].flatMap(Double.init) ?? 240
    private static let fadeOutSeconds: Double = 2.0
    private static let fadeInSeconds: Double = 3.5

    /// Shared initial pick, so every display starts on the same attractor when the screensaver
    /// starts (each display then cycles independently).
    private static var sharedPick: (attractor: Int, palette: Int, at: Date)?

    private var metalLayer: CAMetalLayer?
    private var device: MTLDevice?
    private var renderer: AttractorRenderer?
    private var attractorIndex = 0
    private var paletteIndex = 0

    // Timing. Everything is driven by the display's refresh (CADisplayLink), so frames land
    // exactly once per refresh and the simulation advances by exact refresh intervals.
    private var link: CADisplayLink?
    private var lastTarget: CFTimeInterval = 0
    private var clock: CFTimeInterval = 0          // seconds of animation shown
    private var shownSince: CFTimeInterval = 0     // in `clock` time
    private var fadingOutSince: CFTimeInterval?
    private let inFlight = DispatchSemaphore(value: 2)
    private var stopped = true          // until startAnimation
    private var willStopObserver: NSObjectProtocol?

    // Attractors are prepared off the main thread (shader compile, warm-up, calibration) and
    // swapped in when ready, so switching never blocks rendering.
    private static let loadQueue = DispatchQueue(label: "attractor.load", qos: .userInitiated)
    private var loading = false
    private var pending: AttractorRenderer?
    private var swapAsSoonAsReady = false

    // Frame pacing statistics. Only collected when debugging (ATTRACTOR_DEBUG) or once the
    // viewer has asked for them (--stats), so the real screensaver never accumulates them.
    private let statsLock = NSLock()
    private var lastPresented: CFTimeInterval = 0
    private var presentIntervals: [Double] = []
    private var skippedTicks = 0
    private var framesSinceSwap = 0
    private var lastFade: Float = 0
    private let debugTiming = ProcessInfo.processInfo.environment["ATTRACTOR_DEBUG"] != nil
    private var statsRequested = false

    override init?(frame: NSRect, isPreview: Bool) {
        super.init(frame: frame, isPreview: isPreview)
        // The display link does the real work; this only matters as a fallback.
        animationTimeInterval = 1.0 / 60.0
        wantsLayer = true
        // macOS 14+ keeps legacyScreenSaver alive after the screensaver is dismissed and
        // doesn't always call stopAnimation, so listen for the "will stop" broadcast too.
        // (Not for the System Settings preview, which that broadcast doesn't concern.)
        if !isPreview {
            willStopObserver = DistributedNotificationCenter.default().addObserver(
                forName: NSNotification.Name("com.apple.screensaver.willstop"), object: nil, queue: .main
            ) { [weak self] _ in self?.halt() }
        }
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        wantsLayer = true
    }

    deinit {
        if let o = willStopObserver { DistributedNotificationCenter.default().removeObserver(o) }
        link?.invalidate()
    }

    override func makeBackingLayer() -> CALayer {
        let l = CAMetalLayer()
        let dev = MTLCreateSystemDefaultDevice()
        device = dev
        l.device = dev
        l.pixelFormat = .bgra8Unorm
        l.framebufferOnly = true
        l.isOpaque = true
        l.colorspace = CGColorSpace(name: CGColorSpace.sRGB)
        l.maximumDrawableCount = 3
        l.backgroundColor = CGColor(gray: 0, alpha: 1)
        metalLayer = l
        return l
    }

    override var isOpaque: Bool { true }

    override func draw(_ rect: NSRect) {
        NSColor.black.setFill()
        rect.fill()
    }

    // MARK: - Lifecycle

    override func startAnimation() {
        super.startAnimation()
        stopped = false
        lastTarget = 0
        startLinkIfPossible()
        if renderer == nil {
            choose(fresh: true)
            requestLoad(swapImmediately: true)
        } else if pending != nil {
            swapInPending()
        } else {
            fadeInFromNow()
        }
    }

    override func stopAnimation() {
        super.stopAnimation()
        halt()
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // A display link is bound to the window's screen; recreate it when that changes.
        link?.invalidate()
        link = nil
        lastTarget = 0
        startLinkIfPossible()
    }

    private func startLinkIfPossible() {
        guard !stopped, link == nil, window != nil else { return }
        let l = displayLink(target: self, selector: #selector(tick(_:)))
        l.preferredFrameRateRange = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)
        l.add(to: .main, forMode: .common)
        link = l
    }

    private func halt() {
        stopped = true
        link?.invalidate()
        link = nil
        renderer?.releaseTargets()
    }

    private func fadeInFromNow() {
        shownSince = clock
        fadingOutSince = nil
    }

    // MARK: - Choosing and loading

    private func choose(fresh: Bool) {
        let atts = Attractors.all, pals = Palette.all
        if fresh, let p = Self.sharedPick, Date().timeIntervalSince(p.at) < 10 {
            attractorIndex = p.attractor
            paletteIndex = p.palette
        } else if fresh {
            attractorIndex = Int.random(in: 0..<atts.count)
            paletteIndex = Int.random(in: 0..<pals.count)
        } else {
            // Next one: avoid repeating the current attractor or palette.
            attractorIndex = (attractorIndex + Int.random(in: 1..<atts.count)) % atts.count
            paletteIndex = (paletteIndex + Int.random(in: 1..<pals.count)) % pals.count
        }
        Self.sharedPick = (attractorIndex, paletteIndex, Date())
    }

    private func pixelSize() -> (Int, Int) {
        let scale = window?.backingScaleFactor ?? metalLayer?.contentsScale ?? 2
        return (Int(bounds.width * scale), Int(bounds.height * scale))
    }

    /// Prepares the currently chosen attractor on a background queue.
    private func requestLoad(swapImmediately: Bool) {
        swapAsSoonAsReady = swapAsSoonAsReady || swapImmediately
        guard !loading, let device else { return }
        loading = true
        pending = nil
        // What's on screen now, to fall back to if this load fails.
        let previous = renderer.map { r in
            (Attractors.all.firstIndex { $0.name == r.attractor.name } ?? 0,
             Palette.all.firstIndex { $0.name == r.palette.name } ?? 0)
        }
        let att = Attractors.all[attractorIndex], pal = Palette.all[paletteIndex]
        let count = isPreview ? (1 << 14) : 49_152
        let (w, h) = pixelSize()
        let preview = isPreview
        Self.loadQueue.async { [weak self] in
            var r: AttractorRenderer?
            do {
                r = try AttractorRenderer(device: device, outputFormat: .bgra8Unorm, particleCount: count,
                                          attractor: att, palette: pal)
                if let r, w > 0, h > 0 {
                    Self.configure(r, width: w, height: h, preview: preview)
                    Self.warmUp(r, width: w, height: h)
                }
            } catch {
                NSLog("AttractorSaver: failed to load \(att.name): \(error)")
            }
            DispatchQueue.main.async {
                guard let self else { return }
                self.loading = false
                self.pending = r
                if r == nil {
                    self.swapAsSoonAsReady = false
                    if let (a, p) = previous {
                        // Keep showing the current attractor (fading it back in if a cycle had
                        // started) and keep the description truthful.
                        self.attractorIndex = a
                        self.paletteIndex = p
                        if self.fadingOutSince != nil { self.fadeInFromNow() }
                    } else {
                        // Nothing on screen yet: try another attractor shortly.
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
                            guard let self, self.renderer == nil, !self.stopped else { return }
                            self.choose(fresh: false)
                            self.requestLoad(swapImmediately: true)
                        }
                    }
                    return
                }
                // If the choice changed while this was loading, load again.
                if let r, r.attractor.name != Attractors.all[self.attractorIndex].name
                    || r.palette.name != Palette.all[self.paletteIndex].name {
                    self.pending = nil
                    self.requestLoad(swapImmediately: false)
                }
            }
        }
    }

    /// Sizes targets, calibrates and builds the label. Safe to call off the main thread for a
    /// renderer that isn't on screen yet.
    private static func configure(_ r: AttractorRenderer, width w: Int, height h: Int, preview: Bool) {
        // Keep the accumulation buffer at most ~4K worth of pixels on very large displays.
        let maxPixels: Float = 3840 * 2160
        r.accumScale = min(1, sqrt(maxPixels / Float(w * h)))
        r.resize(width: w, height: h)
        r.calibrateIfNeeded(aspect: Float(w) / Float(h))
        if !preview, let (tex, lw, lh) = LabelMaker.make(device: r.device, attractor: r.attractor,
                                                         screenHeight: h, accent: r.palette.stops[3]) {
            r.labelTexture = tex
            r.labelRectNDC = LabelMaker.rect(w: lw, h: lh, screenW: w, screenH: h)
        }
        r.configuredSize = (w, h)
    }

    /// Renders one throwaway frame so first-use costs (fresh GPU allocations, first pipeline use)
    /// are paid in the background rather than on the first visible frame.
    private static func warmUp(_ r: AttractorRenderer, width w: Int, height h: Int) {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: 64, height: 64, mipmapped: false)
        d.usage = .renderTarget
        d.storageMode = .private
        guard let scratch = r.device.makeTexture(descriptor: d), let cb = r.queue.makeCommandBuffer() else { return }
        r.encodeSimulation(cb, dt: 0, aspect: Float(w) / Float(h))
        r.encodeComposite(cb, target: scratch, fade: 0)
        cb.commit()
        cb.waitUntilCompleted()
    }

    private func swapInPending() {
        guard let p = pending else { return }
        pending = nil
        swapAsSoonAsReady = false
        let old = renderer
        renderer = p
        // Free the old renderer's GPU memory off the main thread.
        if let old { Self.loadQueue.async { withExtendedLifetime(old) {} } }
        framesSinceSwap = 0
        fadeInFromNow()
    }

    // MARK: - Frame

    override func animateOneFrame() {
        // Fallback only: normally the display link drives frames.
        if !stopped, link == nil {
            startLinkIfPossible()
            if link == nil { renderFrame(dt: 1.0 / 60.0) }
        }
    }

    @objc private func tick(_ l: CADisplayLink) {
        let target = l.targetTimestamp
        let dt = lastTarget == 0 ? (l.targetTimestamp - l.timestamp) : target - lastTarget
        lastTarget = target
        renderFrame(dt: dt)
    }

    private func renderFrame(dt rawDt: CFTimeInterval) {
        guard !stopped, let layer = metalLayer else { return }
        let dt = min(max(rawDt, 0), 0.1)

        if pending != nil, swapAsSoonAsReady || renderer == nil { swapInPending() }
        guard renderer != nil else { return }
        clock += dt

        let (w, h) = pixelSize()
        guard w > 0, h > 0 else { return }
        let scale = window?.backingScaleFactor ?? layer.contentsScale
        if layer.contentsScale != scale { layer.contentsScale = scale }
        if layer.drawableSize.width != CGFloat(w) || layer.drawableSize.height != CGFloat(h) {
            layer.drawableSize = CGSize(width: w, height: h)
        }
        // (Re)configure when the size changed, after a stop released the GPU targets, or when a
        // renderer prepared at an older size was swapped in.
        if let r = renderer, r.configuredSize == nil || r.configuredSize! != (w, h) {
            Self.configure(r, width: w, height: h, preview: isPreview)
        }

        // Cycle attractors with a fade through black. The next one starts loading as the fade
        // begins, so it's normally ready by the time the screen is dark.
        if fadingOutSince == nil, clock - shownSince > Self.cycleSeconds {
            fadingOutSince = clock
            choose(fresh: false)
            requestLoad(swapImmediately: false)
        }
        if let t0 = fadingOutSince, clock - t0 >= Self.fadeOutSeconds, pending != nil {
            swapInPending()     // new attractor starts at fade 0 this very frame
        }
        // Brightness is computed after any swap, so the incoming attractor never shows at the
        // outgoing one's brightness (that caused a one-frame flash at every switch).
        var fade = Float(min(1, (clock - shownSince) / Self.fadeInSeconds))
        if let t0 = fadingOutSince {
            fade = min(fade, Float(max(0, 1 - (clock - t0) / Self.fadeOutSeconds)))
        }
        fade = fade * fade * (3 - 2 * fade)
        if debugTiming, abs(fade - lastFade) > 0.05 {
            print(String(format: "  brightness jump %.2f -> %.2f", lastFade, fade))
        }
        lastFade = fade

        guard let renderer else { return }
        // Label drifts in a little after the attractor appears.
        let labelT = Float(max(0, min(1, (clock - shownSince - 2.5) / 3)))
        renderer.labelAlpha = isPreview ? 0 : labelT * labelT * (3 - 2 * labelT)

        // Never queue more than two frames on the GPU; skip a tick instead of blocking.
        guard inFlight.wait(timeout: .now()) == .success else { skippedTicks += 1; return }
        guard let drawable = layer.nextDrawable(), let cb = renderer.queue.makeCommandBuffer() else {
            inFlight.signal()
            return
        }
        renderer.encodeSimulation(cb, dt: Float(dt), aspect: Float(w) / Float(h))
        renderer.encodeComposite(cb, target: drawable.texture, fade: fade)
        let sem = inFlight
        cb.addCompletedHandler { _ in sem.signal() }
        framesSinceSwap += 1
        let phaseLoading = loading, phaseSinceSwap = framesSinceSwap
        if debugTiming || statsRequested { drawable.addPresentedHandler { [weak self] d in
            guard let self, d.presentedTime > 0 else { return }
            self.statsLock.lock()
            if self.lastPresented > 0 {
                let iv = d.presentedTime - self.lastPresented
                if self.statsRequested { self.presentIntervals.append(iv) }
                if self.debugTiming, iv > 0.025 {
                    print(String(format: "  slow present %.1f ms  loading=%d framesSinceSwap=%d", iv * 1000,
                                 phaseLoading ? 1 : 0, phaseSinceSwap))
                }
            }
            self.lastPresented = d.presentedTime
            self.statsLock.unlock()
        } }
        cb.present(drawable)
        cb.commit()
    }

    // MARK: - Manual control (used by the viewer app)

    @objc func showNextAttractor() { jump(attractorBy: 1, paletteBy: Int.random(in: 1..<Palette.all.count)) }
    @objc func showPreviousAttractor() { jump(attractorBy: -1, paletteBy: Int.random(in: 1..<Palette.all.count)) }
    @objc func restartAttractor() {
        if pending != nil { swapInPending() } else { renderer?.restartFromCloud(); fadeInFromNow() }
    }
    @objc func showNextPalette() { jump(attractorBy: 0, paletteBy: 1) }
    @objc func showRandom() { choose(fresh: false); requestLoad(swapImmediately: true) }

    private func jump(attractorBy da: Int, paletteBy dp: Int) {
        let na = Attractors.all.count, np = Palette.all.count
        attractorIndex = ((attractorIndex + da) % na + na) % na
        paletteIndex = ((paletteIndex + dp) % np + np) % np
        requestLoad(swapImmediately: true)
    }

    @objc var currentDescription: String {
        "\(Attractors.all[attractorIndex].name) · \(Palette.all[paletteIndex].name)"
    }

    /// Summary of presentation intervals since the last call: frames, hitches (> 1.5 refresh
    /// periods between presents), worst interval, ticks skipped because the GPU was busy.
    @objc var frameStats: String {
        statsRequested = true
        statsLock.lock()
        let iv = presentIntervals
        presentIntervals.removeAll()
        statsLock.unlock()
        let period = 1.0 / 60.0   // the display link is capped at 60 fps
        let hitches = iv.filter { $0 > period * 1.5 }.count
        let worst = (iv.max() ?? 0) * 1000
        let s = String(format: "frames %d  hitches %d  worst %.1f ms  gpu-busy skips %d",
                       iv.count, hitches, worst, skippedTicks)
        skippedTicks = 0
        return s
    }

    override var hasConfigureSheet: Bool { false }
    override var configureSheet: NSWindow? { nil }
}
