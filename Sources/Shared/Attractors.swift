import Foundation
import simd

/// A strange attractor: its ODE (as Metal source), how to seed it, and how to look at it.
struct Attractor {
    let name: String
    /// Body of `float3 deriv(float3 p)` in Metal. Constants are inlined, so each
    /// attractor gets its own specialised compute pipeline.
    let deriv: String
    /// Pretty equations for the on-screen label.
    let equations: [String]
    let seed: SIMD3<Float>
    let spread: Float
    /// Largest stable RK4 step in the attractor's own time units.
    let maxDt: Float
    /// Simulated time used to let particles settle onto the attractor and mix.
    let prewarmTime: Float
    /// Maps attractor axes to world axes. World Y is up on screen.
    let axes: (Int, Int, Int)
    let flip: SIMD3<Float>
    /// Optional: attractor-space direction to point up on screen (overrides axes).
    var up: SIMD3<Float>? = nil
    // Camera
    var yaw: Float = 0          // degrees
    var pitch: Float = 15       // degrees
    var pitchSway: Float = 6    // degrees of slow pitch oscillation
    var spin: Float = 0         // orbit rate in degrees/second (0 = renderer's default orbit)
    var zoom: Float = 1.0
    var speed: Float = 1.0      // multiplier on the flow speed
    var exposure: Float = 1.0
    var decay: Float = 0.9      // per-frame trail persistence at 60 fps
}

enum Attractors {
    static let all: [Attractor] = [
        Attractor(
            name: "Lorenz",
            deriv: "const float s=10.0, r=28.0, b=8.0/3.0; return float3(s*(p.y-p.x), p.x*(r-p.z)-p.y, p.x*p.y-b*p.z);",
            equations: ["ẋ = σ(y − x)", "ẏ = x(ρ − z) − y", "ż = xy − βz"],
            seed: [1, 1, 20], spread: 6, maxDt: 0.004, prewarmTime: 25,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 140, pitch: 5),
        Attractor(
            name: "Aizawa",
            deriv: """
            const float a=0.95, b=0.7, c=0.6, d=3.5, e=0.25, f=0.1;
            return float3((p.z-b)*p.x - d*p.y, d*p.x + (p.z-b)*p.y,
                          c + a*p.z - p.z*p.z*p.z/3.0 - (p.x*p.x+p.y*p.y)*(1.0+e*p.z) + f*p.z*p.x*p.x*p.x);
            """,
            equations: ["ẋ = (z − b)x − dy", "ẏ = dx + (z − b)y", "ż = c + az − z³/3 − (x² + y²)(1 + ez) + fzx³"],
            seed: [0.1, 0, 0], spread: 0.5, maxDt: 0.01, prewarmTime: 60,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 0, pitch: 18, pitchSway: 10, spin: 3),
        Attractor(
            name: "Halvorsen",
            deriv: "const float a=1.4; return float3(-a*p.x-4.0*p.y-4.0*p.z-p.y*p.y, -a*p.y-4.0*p.z-4.0*p.x-p.z*p.z, -a*p.z-4.0*p.x-4.0*p.y-p.x*p.x);",
            equations: ["ẋ = −ax − 4y − 4z − y²", "ẏ = −ay − 4z − 4x − z²", "ż = −az − 4x − 4y − x²"],
            seed: [-1.48, -1.51, 2.04], spread: 1, maxDt: 0.004, prewarmTime: 40,
            axes: (0, 1, 2), flip: [1, 1, 1], up: [1, 1, 1],
            yaw: 0, pitch: 65, pitchSway: 12, spin: 3),
        Attractor(
            name: "Thomas",
            deriv: "const float b=0.208186; return float3(sin(p.y)-b*p.x, sin(p.z)-b*p.y, sin(p.x)-b*p.z);",
            equations: ["ẋ = sin y − bx", "ẏ = sin z − by", "ż = sin x − bz"],
            seed: [1, 0.5, 0], spread: 2, maxDt: 0.05, prewarmTime: 400,
            axes: (0, 1, 2), flip: [1, 1, 1], up: [1, 1, 1],
            yaw: 0, pitch: 55, pitchSway: 25, spin: 2.5),
        Attractor(
            name: "Chen",
            deriv: "const float a=35.0, b=3.0, c=28.0; return float3(a*(p.y-p.x), (c-a)*p.x - p.x*p.z + c*p.y, p.x*p.y - b*p.z);",
            equations: ["ẋ = a(y − x)", "ẏ = (c − a)x − xz + cy", "ż = xy − bz"],
            seed: [-10, 0, 37], spread: 4, maxDt: 0.002, prewarmTime: 15,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 155, pitch: 5),
        Attractor(
            name: "Dadras",
            deriv: "const float a=3.0, b=2.7, c=1.7, d=2.0, e=9.0; return float3(p.y - a*p.x + b*p.y*p.z, c*p.y - p.x*p.z + p.z, d*p.x*p.y - e*p.z);",
            equations: ["ẋ = y − ax + byz", "ẏ = cy − xz + z", "ż = dxy − ez"],
            seed: [1, 1, 0], spread: 1, maxDt: 0.005, prewarmTime: 60,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 120, pitch: 15, zoom: 0.85),
        Attractor(
            name: "Rössler",
            deriv: "const float a=0.2, b=0.2, c=5.7; return float3(-p.y-p.z, p.x+a*p.y, b+p.z*(p.x-c));",
            equations: ["ẋ = −y − z", "ẏ = x + ay", "ż = b + z(x − c)"],
            seed: [1, 1, 0], spread: 0.5, maxDt: 0.02, prewarmTime: 200,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 15, pitch: 25, exposure: 2.2),
        Attractor(
            name: "Four-Wing",
            deriv: "const float a=0.2, b=0.01, c=-0.4; return float3(a*p.x + p.y*p.z, b*p.x + c*p.y - p.x*p.z, -p.z - p.x*p.y);",
            equations: ["ẋ = ax + yz", "ẏ = bx + cy − xz", "ż = −z − xy"],
            seed: [1.3, -0.18, 0.01], spread: 0.5, maxDt: 0.01, prewarmTime: 150,
            axes: (0, 1, 2), flip: [1, 1, 1],
            yaw: 0, pitch: 15, zoom: 0.85),
        Attractor(
            name: "Burke–Shaw",
            deriv: "const float s=10.0, v=4.272; return float3(-s*(p.x+p.y), -p.y - s*p.x*p.z, s*p.x*p.y + v);",
            equations: ["ẋ = −s(x + y)", "ẏ = −y − sxz", "ż = sxy + v"],
            seed: [0.6, 0, 0], spread: 0.2, maxDt: 0.003, prewarmTime: 40,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 75, pitch: 10),
        Attractor(
            name: "Arneodo",
            deriv: "return float3(p.y, p.z, 5.5*p.x - 3.5*p.y - p.z - p.x*p.x*p.x);",
            equations: ["ẋ = y", "ẏ = z", "ż = −ax − by − z + dx³"],
            seed: [0.2, 0.2, 0], spread: 0.3, maxDt: 0.005, prewarmTime: 80,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 90, pitch: 15),
        Attractor(
            name: "Rucklidge",
            deriv: "const float k=2.0, a=6.7; return float3(-k*p.x + a*p.y - p.y*p.z, p.x, -p.z + p.y*p.y);",
            equations: ["ẋ = −κx + λy − yz", "ẏ = x", "ż = −z + y²"],
            seed: [1, 0, 4.5], spread: 0.5, maxDt: 0.005, prewarmTime: 80,
            axes: (0, 2, 1), flip: [1, 1, 1],
            yaw: 15, pitch: 10),
    ]

    static func key(_ s: String) -> String {
        s.folding(options: [.diacriticInsensitive, .caseInsensitive], locale: nil)
            .filter { $0.isLetter }
    }

    static func named(_ n: String) -> Attractor? {
        let k = key(n)
        return all.first { key($0.name).hasPrefix(k) }
    }
}

struct Palette {
    let name: String
    let stops: [SIMD3<Float>]   // 5 sRGB stops, slow -> fast
    let background: SIMD3<Float>

    static func hex(_ h: UInt32) -> SIMD3<Float> {
        SIMD3(Float((h >> 16) & 0xff), Float((h >> 8) & 0xff), Float(h & 0xff)) / 255
    }

    // Each palette: two or three related hues on a steady dark -> light ramp.
    static let all: [Palette] = [
        Palette(name: "Tide", stops: [hex(0x06324a), hex(0x0f7f8f), hex(0x5fb8c9), hex(0xe58fb8), hex(0xfbd3e4)], background: hex(0x02060a)),
        Palette(name: "Dusk", stops: [hex(0x1d1050), hex(0x5b2a9e), hex(0xb54a9a), hex(0xf08a7e), hex(0xffd2a6)], background: hex(0x05030c)),
        Palette(name: "Opal", stops: [hex(0x0f1f5c), hex(0x2f5fb8), hex(0x6fb6e0), hex(0xd8b4e6), hex(0xf6ecfb)], background: hex(0x03040b)),
        Palette(name: "Orchid", stops: [hex(0x2a0f4f), hex(0x6a2c91), hex(0xb15fc4), hex(0xe5a3dc), hex(0xfbe3f4)], background: hex(0x05030b)),
        Palette(name: "Lagoon", stops: [hex(0x062a4a), hex(0x0b5f8a), hex(0x18a3a8), hex(0x7fd9c4), hex(0xdff7ee)], background: hex(0x020609)),
        Palette(name: "Ember", stops: [hex(0x2a0c3c), hex(0x7a1f5c), hex(0xd2486a), hex(0xf59a6b), hex(0xffd9b0)], background: hex(0x07040a)),
        Palette(name: "Aurora", stops: [hex(0x0b2a5b), hex(0x14659a), hex(0x1fa69a), hex(0x8fd9a8), hex(0xe2f8dc)], background: hex(0x02060b)),
        Palette(name: "Glacier", stops: [hex(0x0a2440), hex(0x15599c), hex(0x4aa3e6), hex(0x9fd7f5), hex(0xe6f6ff)], background: hex(0x02050a)),
        Palette(name: "Gilded", stops: [hex(0x1c1230), hex(0x5a3a5e), hex(0xb07a5a), hex(0xe2b878), hex(0xfaecc8)], background: hex(0x060409)),
    ]
}
