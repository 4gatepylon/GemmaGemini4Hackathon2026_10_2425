// GPU timestamps around regions of GPU work, for any Metal program.
//
// Apple GPUs sample counters only at stage boundaries (M3 Pro, verified:
// supportsCounterSampling(.atStageBoundary) is true, .atDispatchBoundary is
// false), so a region is one compute encoder. makeEncoder() returns an encoder
// whose start and end get timestamped. Everything dispatched in it counts
// toward that region.
//
//   let prof = RegionProfiler(device: device)
//   let enc = prof.makeEncoder(cb, name: "qkv", stack: ["token 0", "layer 3"], bytes: b, flops: f)
//   ...dispatch...
//   prof.end(enc)                      // instead of enc.endEncoding()
//   ...commit, wait for completion...
//   let events = prof.collect().events
//
// Serialization: Apple GPUs run independent encoders concurrently, even
// across command buffers (measured: two encoders with no shared buffers
// started 40 us apart and overlapped). Metal only orders encoders that touch
// the same resources. With `serialize` on (the default), each region waits on
// a fence that the previous region updates, so regions never overlap and each
// one's time is its own.
//
// Limits (measured on M3 Pro): one counter sample buffer holds at most 32 KB
// (4096 timestamps = 2048 regions), and only ~32 can exist at once. So at most
// 65536 regions can be in flight between collect() calls.

import Metal

final class RegionProfiler {
    static let regionsPerBuffer = 2048
    static let maxBuffers = 32

    struct Region {
        var name: String
        var kernel: String?
        var stack: [String]
        var track: String
        var bytes: Double?
        var flops: Double?
        var variant: String?
        var args: [String: String]?
        var buffer: Int
        var slot: Int
    }

    let device: MTLDevice
    private let counterSet: MTLCounterSet
    private var buffers: [MTLCounterSampleBuffer] = []
    private var used = 0
    private(set) var regions: [Region] = []
    private var calib: (cpu: MTLTimestamp, gpu: MTLTimestamp)
    private let fence: MTLFence
    /// Make each region wait for the previous one (see header comment).
    var serialize = true

    init(device: MTLDevice) {
        guard device.supportsCounterSampling(.atStageBoundary),
              let ts = device.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else {
            fatalError("RegionProfiler: this GPU can't sample timestamps at encoder boundaries")
        }
        self.device = device
        counterSet = ts
        calib = device.sampleTimestamps()
        fence = device.makeFence()!
        fence.label = "RegionProfiler"
    }

    private func newBuffer() {
        precondition(buffers.count < Self.maxBuffers,
                     "RegionProfiler: more than \(Self.maxBuffers * Self.regionsPerBuffer) regions in flight; call collect() more often")
        let d = MTLCounterSampleBufferDescriptor()
        d.counterSet = counterSet
        d.storageMode = .shared
        d.sampleCount = 2 * Self.regionsPerBuffer
        d.label = "RegionProfiler \(buffers.count)"
        do { buffers.append(try device.makeCounterSampleBuffer(descriptor: d)) } catch {
            fatalError("RegionProfiler: cannot allocate counter sample buffer: \(error)")
        }
        used = 0
    }

    /// A new compute encoder on `cb` whose GPU start/end times are recorded.
    func makeEncoder(_ cb: MTLCommandBuffer, name: String, kernel: String? = nil, stack: [String] = [],
                     track: String = "GPU", bytes: Double? = nil, flops: Double? = nil,
                     variant: String? = nil, args: [String: String]? = nil) -> MTLComputeCommandEncoder {
        if buffers.isEmpty || used == Self.regionsPerBuffer { newBuffer() }
        let slot = used
        used += 1
        let pd = MTLComputePassDescriptor()
        pd.dispatchType = .serial
        let a = pd.sampleBufferAttachments[0]!
        a.sampleBuffer = buffers[buffers.count - 1]
        a.startOfEncoderSampleIndex = 2 * slot
        a.endOfEncoderSampleIndex = 2 * slot + 1
        regions.append(Region(name: name, kernel: kernel, stack: stack, track: track, bytes: bytes, flops: flops,
                              variant: variant, args: args, buffer: buffers.count - 1, slot: slot))
        let enc = cb.makeComputeCommandEncoder(descriptor: pd)!
        enc.label = name
        if serialize { enc.waitForFence(fence) }
        return enc
    }

    /// End a region's encoder (use this instead of `enc.endEncoding()`).
    func end(_ enc: MTLComputeCommandEncoder) {
        if serialize { enc.updateFence(fence) }
        enc.endEncoding()
    }

    /// Resolve every region recorded so far into events (ns, CLOCK_UPTIME_RAW
    /// domain) and reset. Call only after all command buffers using these
    /// encoders have completed. `invalid` counts regions the GPU didn't time.
    func collect() -> (events: [KProf.Event], invalid: Int) {
        let now = device.sampleTimestamps()
        // GPU ticks -> CPU ns. On Apple silicon both are already ns, so scale is 1.
        let dg = Double(now.gpu &- calib.gpu), dc = Double(now.cpu &- calib.cpu)
        var scale = dg > 1e7 ? dc / dg : 1
        if abs(scale - 1) < 1e-3 { scale = 1 }
        let (c0, g0) = calib
        func toNs(_ g: UInt64) -> UInt64 { UInt64(Double(c0) + (Double(g) - Double(g0)) * scale) }

        var stamps: [[UInt64]] = []
        for (i, b) in buffers.enumerated() {
            let n = i == buffers.count - 1 ? used : Self.regionsPerBuffer
            guard n > 0, let data = try? b.resolveCounterRange(0..<(2 * n)) else { stamps.append([]); continue }
            stamps.append(data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) })
        }
        var events: [KProf.Event] = []
        var invalid = 0
        for r in regions {
            let s = stamps[r.buffer]
            guard 2 * r.slot + 1 < s.count else { invalid += 1; continue }
            let a = s[2 * r.slot], b = s[2 * r.slot + 1]
            if a == MTLCounterErrorValue || b == MTLCounterErrorValue || a == 0 || b < a { invalid += 1; continue }
            events.append(KProf.Event(name: r.name, kernel: r.kernel, stack: r.stack, track: r.track,
                                      startNs: toNs(a), durNs: toNs(b) - toNs(a), bytes: r.bytes, flops: r.flops,
                                      variant: r.variant, args: r.args))
        }
        buffers = []
        regions = []
        used = 0
        calib = device.sampleTimestamps()
        return (events, invalid)
    }
}
