// Metal source, compiled at runtime (no offline Metal toolchain needed).
// `DERIV_BODY` is replaced by the attractor's ODE so each attractor gets a
// specialised, branch-free integrator; `TRAIL_SEGS_VALUE` by AttractorRenderer.trailSegments.
// `struct Uniforms` must match the Swift struct in Renderer.swift.
let shaderTemplate = #"""
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;
    float4 sim;    // x: RK4 step h, y: RK4 steps this frame, z: 1/medianSpeed, w: line intensity
    float4 misc;   // x: time, y: trail decay (sim pass), z: exposure (composite), w: global fade
    float4 cam;    // x: camera distance, y: respawn pool size (normally particle count), z: seed spread, w: frame index
    float4 seed;   // xyz: seed point
    float4 pal[5];
    float4 bg;     // rgb background (linear)
    float4 label;  // label rect in NDC: x0, y0, x1, y1
    float4 extra;  // x: label alpha, y: bloom strength, z: colour drift
    float4 head;   // x: head size px, y: head brightness
    float4 view;   // xy: accumulation target size px, z: line half-width px
};

struct Particle {
    float4 pos;   // xyz position, w: speed
};

static inline float3 deriv(float3 p) {
    DERIV_BODY
}

static inline float3 rk4(float3 p, float h) {
    float3 k1 = deriv(p);
    float3 k2 = deriv(p + 0.5f * h * k1);
    float3 k3 = deriv(p + 0.5f * h * k2);
    float3 k4 = deriv(p + h * k3);
    return p + (h / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
}

static inline uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}
static inline float rand01(uint x) { return float(hash(x) & 0xffffffu) / 16777216.0f; }

static inline float3 randGauss3(uint s) {
    float u1 = max(rand01(s * 4u + 1u), 1e-7f), u2 = rand01(s * 4u + 2u);
    float u3 = max(rand01(s * 4u + 3u), 1e-7f), u4 = rand01(s * 4u + 4u);
    float r1 = sqrt(-2.0f * log(u1)), r2 = sqrt(-2.0f * log(u3));
    return float3(r1 * cos(6.2831853f * u2), r1 * sin(6.2831853f * u2), r2 * cos(6.2831853f * u4));
}

kernel void seedParticles(device Particle* ps [[buffer(0)]],
                          constant Uniforms& u [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    float3 p = u.seed.xyz + u.cam.z * randGauss3(id + 12345u);
    ps[id].pos = float4(p, 0.0f);
}

// Each frame the integrator records TRAIL_SEGS+1 points along the true path, so the streak
// drawn for this frame follows the curve instead of cutting a straight chord across it.
#define TRAIL_SEGS TRAIL_SEGS_VALUE   // injected from AttractorRenderer.trailSegments

kernel void stepParticles(device Particle* ps [[buffer(0)]],
                          constant Uniforms& u [[buffer(1)]],
                          device float4* path [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    float3 p = ps[id].pos.xyz;
    float h = u.sim.x;
    int n = int(u.sim.y);
    device float4* pts = path + id * (TRAIL_SEGS + 1);
    pts[0] = float4(p, 0.0f);
    int done = 0;
    for (int seg = 1; seg <= TRAIL_SEGS; seg++) {
        int target = (n * seg) / TRAIL_SEGS;
        for (; done < target; done++) p = rk4(p, h);
        pts[seg] = float4(p, 0.0f);
    }

    bool bad = any(isnan(p)) || any(isinf(p)) || length_squared(p) > 1e8f;
    if (bad) {
        // Respawn next to a random other particle; chaos separates them quickly.
        uint count = uint(u.cam.y);
        uint other = hash(id ^ (uint(u.cam.w) * 2654435761u)) % count;
        float3 q = ps[other].pos.xyz;
        if (any(isnan(q)) || any(isinf(q)) || length_squared(q) > 1e8f) q = u.seed.xyz;
        p = q + 1e-3f * randGauss3(id + uint(u.cam.w));
        for (int seg = 0; seg <= TRAIL_SEGS; seg++) pts[seg] = float4(p, 0.0f);
    }
    ps[id].pos = float4(p, length(deriv(p)));
}

// ---------------------------------------------------------------- drawing

struct LineOut {
    float4 position [[position]];
    half4 color;
    float across [[center_no_perspective]];  // signed pixel distance from the line's centre
    float along [[center_no_perspective]];   // pixel distance along the segment from its start
    float segLen [[flat]];                   // segment length in pixels
};

static inline float3 palette(constant Uniforms& u, float t) {
    float x = saturate(t) * 4.0f;
    int i = min(int(x), 3);
    return mix(u.pal[i].rgb, u.pal[i + 1].rgb, x - float(i));
}

static inline half4 particleColor(constant Uniforms& u, Particle P, uint pid, float clipW);

// Each path segment is expanded into a screen-space quad slightly wider than the line, and the
// fragment shader fades the edges by pixel coverage: smooth, anti-aliased sub-pixel lines.
vertex LineOut lineVertex(uint vid [[vertex_id]],
                          device const Particle* ps [[buffer(0)]],
                          constant Uniforms& u [[buffer(1)]],
                          device const float4* path [[buffer(2)]]) {
    uint seg = vid / 6u, corner = vid % 6u;
    uint pid = seg / TRAIL_SEGS, k = seg % TRAIL_SEGS;
    device const float4* pts = path + pid * (TRAIL_SEGS + 1);
    float4 ca = u.mvp * float4(pts[k].xyz, 1.0f);
    float4 cb = u.mvp * float4(pts[k + 1].xyz, 1.0f);
    float2 half_vp = 0.5f * u.view.xy;
    float2 sa = ca.xy / ca.w * half_vp, sb = cb.xy / cb.w * half_vp;
    float2 d = sb - sa;
    float len = length(d);
    float2 dir = len > 1e-5f ? d / len : float2(1.0f, 0.0f);
    float2 nrm = float2(-dir.y, dir.x);

    // Two triangles: (a,-) (b,-) (a,+) / (a,+) (b,-) (b,+)
    bool useB = corner == 1u || corner == 4u || corner == 5u;
    float side = (corner == 2u || corner == 3u || corner == 5u) ? 1.0f : -1.0f;
    float extent = u.view.z + 0.5f;              // half-width plus the box filter's half-pixel reach
    float4 c = useB ? cb : ca;
    // Also extend a pixel past each end so very short (sub-pixel) segments are box-filtered
    // along their length instead of randomly hitting or missing pixel centres.
    float2 sp = (useB ? sb : sa) + nrm * side * extent + dir * (useB ? 0.5f : -0.5f);

    LineOut o;
    o.position = float4(sp / half_vp * c.w, c.z, c.w);
    o.across = side * extent;
    o.along = useB ? len + 0.5f : -0.5f;
    o.segLen = len;
    o.color = particleColor(u, ps[pid], pid, c.w);
    return o;
}

struct PointOut {
    float4 position [[position]];
    half4 color;
    float size [[point_size]];
};


// Particle heads: soft round sprites drawn straight onto the final image (after tonemapping),
// so they stay crisp instead of smearing in the trail buffer. Size grows as they approach.
vertex PointOut headVertex(uint vid [[vertex_id]],
                           device const Particle* ps [[buffer(0)]],
                           constant Uniforms& u [[buffer(1)]]) {
    Particle P = ps[vid];
    PointOut o;
    o.position = u.mvp * float4(P.pos.xyz, 1.0f);
    half4 c = particleColor(u, P, vid, o.position.w);
    float near = u.cam.x / max(o.position.w, 0.1f);
    o.size = u.head.x * near * near;
    // Gamma-ish lift so heads read as bright cores of the palette colour.
    float3 lin = float3(c.rgb) * u.head.y / max(u.sim.w, 1e-6f);
    o.color = half4(half3(pow(saturate(lin), float3(0.6f)) * u.misc.w), 1.0h);
    return o;
}

fragment half4 headFragment(PointOut in [[stage_in]], float2 pc [[point_coord]]) {
    float r2 = length_squared(pc - 0.5f) * 4.0f;
    float a = exp(-r2 * 3.0f) * step(r2, 1.0f);
    return in.color * half(a);
}

static inline half4 particleColor(constant Uniforms& u, Particle P, uint pid, float clipW) {

    // Colour by speed relative to the median.
    float s = P.pos.w * u.sim.z;
    float t = 0.42f * log2(max(s, 1e-4f)) + 0.32f;
    // A little per-particle jitter so neighbouring particles read as individuals,
    // plus a slow drift through the palette. Mirrored so it never jumps.
    t += (rand01(pid) - 0.5f) * 0.12f + u.misc.x * u.extra.z;
    t = 1.0f - abs(1.0f - fract(t * 0.5f) * 2.0f);
    float3 c = palette(u, t);

    // Depth cue: the far side of the attractor glows a little less.
    float d = u.cam.x;
    float depth = saturate((clipW - (d - 1.0f)) * 0.5f);
    c *= mix(1.25f, 0.45f, depth);

    // Per-particle brightness variation keeps individual streaks legible in dense sheets.
    float r = rand01(pid * 7u + 3u);
    c *= 0.3f + 1.7f * r * r;

    return half4(half3(c * u.sim.w), 1.0h);
}

fragment half4 lineFragment(LineOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    // Coverage of a box filter over a line of width 2*halfWidth.
    float hw = u.view.z;
    float x = abs(in.across);
    float cov = saturate(hw + 0.5f - x) * min(1.0f, 2.0f * hw);
    // Overlap of this pixel's footprint [t-0.5, t+0.5] with the segment [0, len].
    float t = in.along;
    cov *= saturate(min(t + 0.5f, in.segLen) - max(t - 0.5f, 0.0f));
    return in.color * half(cov);
}

struct QuadOut {
    float4 position [[position]];
    float2 uv;
};

vertex QuadOut fullscreenVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    QuadOut o;
    o.position = float4(p * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = float2(p.x, 1.0f - p.y);
    return o;
}

// Output alpha = decay; blending is (src * 0 + dst * srcAlpha) so this fades the trails.
fragment half4 fadeFragment(QuadOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    return half4(0.0h, 0.0h, 0.0h, half(u.misc.y));
}

// Simple downsample with a 4-tap box (used to build the bloom chain).
fragment half4 downsampleFragment(QuadOut in [[stage_in]],
                                  texture2d<half> src [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 t = 1.0f / float2(src.get_width(), src.get_height());
    half4 c = src.sample(s, in.uv + t * float2(-1, -1))
            + src.sample(s, in.uv + t * float2( 1, -1))
            + src.sample(s, in.uv + t * float2(-1,  1))
            + src.sample(s, in.uv + t * float2( 1,  1));
    return c * 0.25h;
}

static inline float ign(float2 p, float t) {
    // Interleaved gradient noise for dithering, animated.
    p += t * 5.588238f;
    return fract(52.9829189f * fract(dot(p, float2(0.06711056f, 0.00583715f))));
}

fragment half4 compositeFragment(QuadOut in [[stage_in]],
                                 constant Uniforms& u [[buffer(1)]],
                                 texture2d<half> accum [[texture(0)]],
                                 texture2d<half> bloom1 [[texture(1)]],
                                 texture2d<half> bloom2 [[texture(2)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float3 c = float3(accum.sample(s, in.uv).rgb);
    float3 b = float3(bloom1.sample(s, in.uv).rgb) * 0.6f + float3(bloom2.sample(s, in.uv).rgb) * 0.4f;
    c += b * u.extra.y;
    c *= u.misc.z;

    // Mostly hue-preserving shoulder, so dense regions stay coloured and only the
    // very hottest cores drift toward white.
    float L = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    float3 huePreserving = c * ((1.0f - exp(-L)) / max(L, 1e-5f));
    float3 perChannel = 1.0f - exp(-c);
    float3 mapped = mix(huePreserving, perChannel, 0.3f);
    mapped = min(mapped, 1.0f);
    // Black point: the faint exponential tail of old trails would otherwise be lifted into a
    // visible haze by display gamma. Cutting it keeps tails short and the background clean.
    mapped = max(mapped - 0.006f, 0.0f) * (1.0f / 0.994f);

    float2 q = in.uv - 0.5f;
    float vignette = 1.0f - dot(q, q) * 0.9f;
    float3 bg = u.bg.rgb * (1.0f - dot(q, q) * 1.6f);
    float3 col = bg + mapped * vignette;
    col *= u.misc.w;

    col = pow(max(col, 0.0f), float3(1.0f / 2.2f));
    col += (ign(in.position.xy, fmod(u.misc.x * 60.0f, 64.0f)) - 0.5f) / 255.0f;
    return half4(half3(col), 1.0h);
}

// Label: premultiplied RGBA texture drawn into a rect.
vertex QuadOut labelVertex(uint vid [[vertex_id]], constant Uniforms& u [[buffer(1)]]) {
    float2 corners[4] = { float2(0, 0), float2(1, 0), float2(0, 1), float2(1, 1) };
    float2 p = corners[vid];
    QuadOut o;
    o.position = float4(mix(u.label.x, u.label.z, p.x), mix(u.label.y, u.label.w, p.y), 0.0f, 1.0f);
    o.uv = float2(p.x, 1.0f - p.y);
    return o;
}

fragment half4 labelFragment(QuadOut in [[stage_in]],
                             constant Uniforms& u [[buffer(1)]],
                             texture2d<half> tex [[texture(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return tex.sample(s, in.uv) * half(u.extra.x * u.misc.w);
}
"""#
