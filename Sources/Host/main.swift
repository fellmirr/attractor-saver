// Attractor Viewer: runs the real .saver bundle in a window so you can watch, cycle and profile it.
//
//   → / space   next attractor        ←   previous attractor
//   p           next palette          r   random attractor + palette
//   s           restart (watch it form)   f   toggle full screen    q / esc   quit
//
// CLI: AttractorViewer [path/to/Attractor.saver] [--seconds N] [--size WxH] [--next-every S]
import AppKit
import ScreenSaver

let argv = CommandLine.arguments
func arg(_ k: String) -> String? { argv.firstIndex(of: k).flatMap { $0 + 1 < argv.count ? argv[$0 + 1] : nil } }

// Use an explicit path, else the saver embedded in our app bundle, else one next to the binary.
let candidates = [argv.dropFirst().first { $0.hasSuffix(".saver") },
                  Bundle.main.path(forResource: "Attractor", ofType: "saver"),
                  (argv[0] as NSString).deletingLastPathComponent + "/Attractor.saver"].compactMap { $0 }
guard let path = candidates.first(where: { FileManager.default.fileExists(atPath: $0) }),
      let bundle = Bundle(path: path), bundle.load(),
      let cls = bundle.principalClass as? ScreenSaverView.Type else {
    fputs("Could not load Attractor.saver (looked in: \(candidates))\n", stderr)
    exit(1)
}
let seconds = Double(arg("--seconds") ?? "") ?? 0
let nextEvery = Double(arg("--next-every") ?? "") ?? 0
let size = (arg("--size") ?? "1440x900").split(separator: "x").compactMap { Double($0) }

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let screenIndex = Int(arg("--screen") ?? "") ?? 0
let targetScreen = NSScreen.screens.indices.contains(screenIndex) ? NSScreen.screens[screenIndex] : NSScreen.main!
let fullscreen = argv.contains("--fullscreen")
let rect = fullscreen ? NSRect(origin: .zero, size: targetScreen.frame.size)
                      : NSRect(x: 0, y: 0, width: size[0], height: size[1])
let window = NSWindow(contentRect: rect,
                      styleMask: fullscreen ? [.borderless] : [.titled, .closable, .resizable, .miniaturizable],
                      backing: .buffered, defer: false, screen: targetScreen)
if fullscreen {
    window.setFrame(targetScreen.frame, display: true)
    window.level = .screenSaver
}
window.collectionBehavior = [.fullScreenPrimary]
if !fullscreen { window.center() }
let view = cls.init(frame: rect, isPreview: false)!
view.autoresizingMask = [.width, .height]
window.contentView = view

func updateTitle() {
    let d = (view.value(forKey: "currentDescription") as? String) ?? ""
    window.title = "\(d)    —    →/space next · ← prev · p palette · r random · s restart · f fullscreen"
}
func send(_ sel: String) {
    view.perform(NSSelectorFromString(sel))
    updateTitle()
}

NSEvent.addLocalMonitorForEvents(matching: .keyDown) { e in
    switch e.keyCode {
    case 124, 49: send("showNextAttractor")       // → or space
    case 123: send("showPreviousAttractor")       // ←
    case 53: app.terminate(nil)                    // esc
    default:
        switch e.charactersIgnoringModifiers?.lowercased() {
        case "p": send("showNextPalette")
        case "r": send("showRandom")
        case "s": send("restartAttractor")
        case "f": window.toggleFullScreen(nil)
        case "q": app.terminate(nil)
        default: return e
        }
    }
    return nil
}

final class Delegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
let delegate = Delegate()
app.delegate = delegate

window.makeKeyAndOrderFront(nil)
app.activate(ignoringOtherApps: true)
view.startAnimation()
updateTitle()
// ScreenSaverView.startAnimation() drives animateOneFrame itself, exactly as in the real screensaver.
if nextEvery > 0 {
    RunLoop.main.add(Timer(timeInterval: nextEvery, repeats: true) { _ in send("showNextAttractor") }, forMode: .common)
}
if argv.contains("--stats") {
    RunLoop.main.add(Timer(timeInterval: 5, repeats: true) { _ in
        print((view.value(forKey: "currentDescription") as? String ?? ""), "|", view.value(forKey: "frameStats") as? String ?? "")
        fflush(stdout)
    }, forMode: .common)
}
if seconds > 0 {
    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { view.stopAnimation(); exit(0) }
}
app.run()
