// kbench: time Metal kernels described by a JSON spec (see ../../README.md).
//
//   kbench SPEC.json [--set NAME=VALUE]... [--reps N] [--warmup N] [--out PATH]
//                    [--peaks PATH] [--reset-peaks] [--idle-ms MS] [--no-noise] [--list]
//
// Each repetition is one command buffer. Each step is its own timestamped
// encoder, with the step's dispatch repeated `repeat` times. Timed encoders are
// chained with a fence so they never overlap (Apple GPUs otherwise run
// independent encoders concurrently). Repetitions are queued back to back
// (pipelined), so the GPU stays busy and clocked up.
// Results: a table on stdout plus a kprof JSON file (default
// $DATA_DIR/prof/kbench/<name>-<time>.json, $DATA_DIR = <repo>/.context by default).

import Foundation
import Metal

let profilingDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
let repoRoot = profilingDir.deletingLastPathComponent().deletingLastPathComponent()
// Outputs go to $DATA_DIR/prof ($DATA_DIR defaults to <repo>/.context, as in surfaces/paths.py).
let dataDir = ProcessInfo.processInfo.environment["DATA_DIR"].map { URL(fileURLWithPath: $0) }
    ?? repoRoot.appendingPathComponent(".context")
let profDir = dataDir.appendingPathComponent("prof")

func die(_ m: String) -> Never {
    FileHandle.standardError.write("kbench: \(m)\n".data(using: .utf8)!)
    exit(1)
}

struct Options {
    var spec = ""
    var sets: [(String, String)] = []
    var reps: Int?
    var warmup: Int?
    var out: String?
    var peaks = profDir.appendingPathComponent("peaks.json").path
    var idleMs = 1000
    var noise = true
    var list = false
    var resetPeaks = false
}

let usage = """
usage: kbench SPEC.json [--set NAME=VALUE]... [--reps N] [--warmup N] [--out PATH]
                        [--peaks PATH] [--reset-peaks] [--idle-ms MS] [--no-noise] [--list]
  --set NAME=VALUE  override a define or var of the spec (repeatable)
  --out PATH        kprof JSON output (default $DATA_DIR/prof/kbench/<name>-<time>.json)
  --peaks PATH      measured roofline peaks (default $DATA_DIR/prof/peaks.json)
  --reset-peaks     let a roofline spec overwrite peaks even if lower (default: keep the best)
  --idle-ms MS      GPU-utilization baseline window before the run (default 1000)
  --no-noise        skip noise recording
  --list            print the resolved spec and exit
"""

func parseOptions() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    func next(_ f: String) -> String { guard let v = it.next() else { die("missing value for \(f)") }; return v }
    while let a = it.next() {
        switch a {
        case "--set":
            let kv = next(a).split(separator: "=", maxSplits: 1).map(String.init)
            guard kv.count == 2 else { die("--set expects NAME=VALUE") }
            o.sets.append((kv[0], kv[1]))
        case "--reps": o.reps = Int(next(a))
        case "--warmup": o.warmup = Int(next(a))
        case "--out": o.out = next(a)
        case "--peaks": o.peaks = next(a)
        case "--idle-ms": o.idleMs = Int(next(a)) ?? 1000
        case "--no-noise": o.noise = false
        case "--list": o.list = true
        case "--reset-peaks": o.resetPeaks = true
        case "-h", "--help": print(usage); exit(0)
        default:
            if a.hasPrefix("-") || !o.spec.isEmpty { die("unexpected argument \(a)\n\(usage)") }
            o.spec = a
        }
    }
    if o.spec.isEmpty { die(usage) }
    return o
}

// MARK: - peaks (measured roofline)

struct Peaks: Codable {
    var bandwidthGbps: Double?
    var bandwidthFrom: String?
    var flopsGflops: Double?
    var flopsFrom: String?
    var measured: String?
}

func loadPeaks(_ path: String) -> Peaks? {
    guard let d = FileManager.default.contents(atPath: path) else { return nil }
    let dec = JSONDecoder()
    dec.keyDecodingStrategy = .convertFromSnakeCase
    return try? dec.decode(Peaks.self, from: d)
}

func savePeaks(_ p: Peaks, _ path: String) {
    let enc = JSONEncoder()
    enc.keyEncodingStrategy = .convertToSnakeCase
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try! enc.encode(p).write(to: url)
}

// MARK: - main

let opts = parseOptions()
guard let device = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
let spec: ResolvedSpec
do { spec = try BenchSpec(path: opts.spec).resolve(overrides: opts.sets) } catch { die("\(error)") }
let reps = opts.reps ?? spec.reps
let warmup = opts.warmup ?? spec.warmup

if opts.list {
    print("\(spec.name) [\(spec.variant)]  source: \(spec.sourcePath)")
    print("defines: \(spec.defines.map { "\($0.0)=\($0.1)" }.joined(separator: " "))")
    print("vars: \(spec.vars.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))")
    for b in spec.buffers { print("buffer \(b.name): \(b.type) x \(b.count) (\(b.bytes >> 10) KiB) x\(b.rotate) \(b.initial)") }
    for s in spec.steps { print("step \(s.name): \(s.kernel) \(s.dispatch) repeat \(s.repeatCount) bytes \(s.bytes ?? 0) flops \(s.flops ?? 0)") }
    exit(0)
}

// Noise: snapshot and idle GPU baseline before we touch the GPU.
let monitor = NoiseMonitor()
let startSnap = NoiseMonitor.snapshot(topProcesses: opts.noise)
let idleSamples = opts.noise ? monitor.sampleIdle(ms: opts.idleMs) : []

// Compile: tunables are injected as #defines in front of the source (as the kernels in ../../kernels/ do).
guard var src = try? String(contentsOfFile: spec.sourcePath, encoding: .utf8) else { die("cannot read \(spec.sourcePath)") }
src = spec.defines.map { "#define \($0.0) \($0.1)\n" }.joined() + src
let copts = MTLCompileOptions()
copts.mathMode = spec.fastMath ? .fast : .safe
let library: MTLLibrary
do { library = try device.makeLibrary(source: src, options: copts) } catch { die("kernel compile failed:\n\(error)") }
var pipelines: [String: MTLComputePipelineState] = [:]
for s in spec.steps where pipelines[s.kernel] == nil {
    guard let f = library.makeFunction(name: s.kernel) else {
        die("no kernel \(s.kernel) in \(spec.sourcePath) (has: \(library.functionNames.sorted().joined(separator: ", ")))")
    }
    do { pipelines[s.kernel] = try device.makeComputePipelineState(function: f) } catch { die("pipeline \(s.kernel): \(error)") }
}

// Buffers (rotated copies get identical contents).
var buffers: [String: [MTLBuffer]] = [:]
for (bi, b) in spec.buffers.enumerated() {
    guard let first = device.makeBuffer(length: max(b.bytes, 16), options: .storageModeShared) else { die("cannot allocate \(b.name) (\(b.bytes) B)") }
    first.label = b.name
    fill(first, b, seed: UInt64(bi + 1) &* 0x9E37_79B9)
    var copies = [first]
    for _ in 1..<max(b.rotate, 1) {
        let c = device.makeBuffer(bytes: first.contents(), length: max(b.bytes, 16), options: .storageModeShared)!
        c.label = b.name
        copies.append(c)
    }
    buffers[b.name] = copies
}

let queue = device.makeCommandQueue()!
let prof = RegionProfiler(device: device)
var dispatchIndex = [Int](repeating: 0, count: spec.steps.count)

func encode(_ enc: MTLComputeCommandEncoder, step si: Int) {
    let s = spec.steps[si]
    enc.setComputePipelineState(pipelines[s.kernel]!)
    for (i, n) in s.threadgroupMemory.enumerated() { enc.setThreadgroupMemoryLength(n, index: i) }
    for _ in 0..<s.repeatCount {
        let k = dispatchIndex[si]
        dispatchIndex[si] += 1
        for (i, b) in s.binds.enumerated() {
            switch b {
            case .buffer(let name, let offset):
                let copies = buffers[name]!
                enc.setBuffer(copies[k % copies.count], offset: offset, index: i)
            case .bytes(let bytes):
                bytes.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: bytes.count, index: i) }
            }
        }
        switch s.dispatch {
        case .threadgroups(let g, let t): enc.dispatchThreadgroups(g, threadsPerThreadgroup: t)
        case .threads(let g, let t): enc.dispatchThreads(g, threadsPerThreadgroup: t)
        }
    }
}

enum Pass { case warmup, steps, whole }
let loopCount = spec.loop?.count ?? 1
func stepBytes(_ s: StepSpec) -> Double? { s.bytes.map { $0 * Double(s.repeatCount) } }
func stepFlops(_ s: StepSpec) -> Double? { s.flops.map { $0 * Double(s.repeatCount) } }
let wholeBytes = spec.steps.allSatisfy { $0.bytes != nil } ? spec.steps.map { stepBytes($0)! }.reduce(0, +) * Double(loopCount) : nil
let wholeFlops = spec.steps.contains { $0.flops != nil } ? spec.steps.map { stepFlops($0) ?? 0 }.reduce(0, +) * Double(loopCount) : nil

/// `.steps`: every step is its own timed region (nested rep > loop iteration > group > step).
/// `.whole`: the entire rep is one timed encoder, like a real engine encodes a token.
func runReps(_ n: Int, _ pass: Pass) {
    var last: MTLCommandBuffer?
    for rep in 0..<n {
        let cb = queue.makeCommandBuffer()!
        if pass == .whole {
            let enc = prof.makeEncoder(cb, name: "whole", stack: ["whole \(rep)"], track: "GPU (whole)",
                                       bytes: wholeBytes, flops: wholeFlops, variant: spec.variant)
            for _ in 0..<loopCount { for si in spec.steps.indices { encode(enc, step: si) } }
            prof.end(enc)
        } else {
            for it in 0..<loopCount {
                for (si, s) in spec.steps.enumerated() {
                    if pass == .warmup {
                        let enc = cb.makeComputeCommandEncoder()!
                        encode(enc, step: si)
                        enc.endEncoding()
                        continue
                    }
                    var stack = ["rep \(rep)"]
                    if let l = spec.loop { stack.append("\(l.name) \(it)") }
                    if let g = s.group { stack.append(g) }
                    stack.append(s.name)
                    let enc = prof.makeEncoder(cb, name: s.name, kernel: s.kernel, stack: stack,
                                               bytes: stepBytes(s), flops: stepFlops(s), variant: spec.variant,
                                               args: ["repeat": String(s.repeatCount)])
                    encode(enc, step: si)
                    prof.end(enc)
                }
            }
        }
        cb.commit()
        last = cb
    }
    last?.waitUntilCompleted()
    if let e = last?.error { die("GPU error: \(e)") }
}

runReps(warmup, .warmup)
if opts.noise { monitor.start() }
runReps(reps, .steps)
if spec.whole { runReps(reps, .whole) }
let runSamples = opts.noise ? monitor.stop() : []
let endSnap = NoiseMonitor.snapshot(topProcesses: false)
let (events, invalid) = prof.collect()
if invalid > 0 { FileHandle.standardError.write("kbench: warning: \(invalid) regions had no valid GPU timestamps\n".data(using: .utf8)!) }

// MARK: - results

var peaks = loadPeaks(opts.peaks) ?? Peaks()
var results: [KProf.Result] = []
for s in spec.steps {
    let perDispatchUs = events.filter { $0.name == s.name && $0.track == "GPU" }.map { Double($0.durNs) / 1e3 / Double(s.repeatCount) }
    guard !perDispatchUs.isEmpty else { continue }
    let st = Summary(perDispatchUs)
    let gbps = s.bytes.map { $0 / (st.median * 1e3) }
    let gflops = s.flops.map { $0 / (st.median * 1e3) }
    results.append(KProf.Result(
        name: s.name, variant: spec.variant,
        method: "kbench: \(st.n) reps pipelined, \(s.repeatCount) dispatch(es) per timed encoder, us = encoder time / repeat",
        n: st.n, medianUs: st.median, meanUs: st.mean, stdUs: st.std, minUs: st.min, p10Us: st.p10, p90Us: st.p90,
        bytes: s.bytes, flops: s.flops, gbps: gbps, gflops: gflops,
        pctPeakBw: gbps.flatMap { g in peaks.bandwidthGbps.map { 100 * g / $0 } },
        pctPeakFlops: gflops.flatMap { g in peaks.flopsGflops.map { 100 * g / $0 } },
        extra: ["kernel": s.kernel]))
}

// Pipelines: the sum of per-step times vs the same work as one encoder.
var pipelineLine: String?
if spec.whole {
    let w = events.filter { $0.track == "GPU (whole)" }.map { Double($0.durNs) / 1e3 }
    if !w.isEmpty {
        let st = Summary(w)
        let sumSteps = zip(spec.steps, results).map { $1.medianUs * Double($0.repeatCount) * Double(loopCount) }.reduce(0, +)
        results.append(KProf.Result(
            name: "whole", variant: spec.variant,
            method: "kbench: \(st.n) reps, the whole rep (\(loopCount) x \(spec.steps.count) steps) in one encoder, no per-step timestamps",
            n: st.n, medianUs: st.median, meanUs: st.mean, stdUs: st.std, minUs: st.min, p10Us: st.p10, p90Us: st.p90,
            bytes: wholeBytes, flops: wholeFlops,
            gbps: wholeBytes.map { $0 / (st.median * 1e3) }, gflops: wholeFlops.map { $0 / (st.median * 1e3) },
            pctPeakBw: nil, pctPeakFlops: nil,
            extra: ["sum_of_steps_us": String(format: "%.2f", sumSteps)]))
        pipelineLine = String(format: "pipeline: whole rep %.1f us as one encoder; sum of split steps %.1f us (%+.1f%% from splitting into %d timed encoders)",
                              st.median, sumSteps, 100 * (sumSteps - st.median) / st.median, loopCount * spec.steps.count)
    }
}

// Roofline specs record the best throughput ever measured as the peak. A run
// slowed down by noise never lowers the roof (use --reset-peaks to start over).
var peakMessage: String?
if let roof = spec.roof {
    let from = "\(spec.name) [\(spec.variant)] \(isoNow())"
    func update(_ old: Double?, _ new: Double?, _ unit: String, _ set: (Double) -> Void) {
        guard let new else { return }
        if opts.resetPeaks || old == nil || new > old! {
            set(new)
            peakMessage = String(format: "peak %@ set to %.1f (was %@)", unit, new, old.map { String(format: "%.1f", $0) } ?? "unset")
        } else {
            peakMessage = String(format: "peak %@ kept at %.1f (this run %.1f; --reset-peaks to overwrite)", unit, old!, new)
        }
    }
    if roof == "bandwidth" {
        update(peaks.bandwidthGbps, results.compactMap({ $0.gbps }).max(), "GB/s") { peaks.bandwidthGbps = $0; peaks.bandwidthFrom = from }
    } else if roof == "flops" {
        update(peaks.flopsGflops, results.compactMap({ $0.gflops }).max(), "GFLOP/s") { peaks.flopsGflops = $0; peaks.flopsFrom = from }
    }
    peaks.measured = isoNow()
    savePeaks(peaks, opts.peaks)
    // Recompute % of peak with the updated roof.
    for i in results.indices {
        results[i].pctPeakBw = results[i].gbps.flatMap { g in peaks.bandwidthGbps.map { 100 * g / $0 } }
        results[i].pctPeakFlops = results[i].gflops.flatMap { g in peaks.flopsGflops.map { 100 * g / $0 } }
    }
}

for i in results.indices where results[i].pctPeakBw == nil && results[i].pctPeakFlops == nil {
    results[i].pctPeakBw = results[i].gbps.flatMap { g in peaks.bandwidthGbps.map { 100 * g / $0 } }
    results[i].pctPeakFlops = results[i].gflops.flatMap { g in peaks.flopsGflops.map { 100 * g / $0 } }
}

let idleStats = NoiseMonitor.stats(idleSamples)
let noise: KProf.Noise? = opts.noise ? KProf.Noise(
    start: startSnap, end: endSnap, gpuUtilIdle: idleStats, gpuUtilRun: NoiseMonitor.stats(runSamples),
    notes: NoiseMonitor.notes(start: startSnap, end: endSnap, idle: idleStats)) : nil

let meta = KProf.Meta(tool: "kbench", name: spec.name, created: isoNow(), device: deviceMeta(device),
                      os: ProcessInfo.processInfo.operatingSystemVersionString, git: gitDescribe(profilingDir.path),
                      argv: CommandLine.arguments, spec: (opts.spec as NSString).standardizingPath, params: spec.params)
var counters = idleSamples.map { KProf.Counter(track: "GPU util % (device-wide, idle baseline)", tNs: $0.0, value: $0.1) }
counters += runSamples.map { KProf.Counter(track: "GPU util % (device-wide)", tNs: $0.0, value: $0.1) }
let out = KProf(meta: meta, noise: noise, events: events, counters: counters, results: results)

let stamp: String = {
    let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"; return f.string(from: Date())
}()
let slug = spec.name.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" }.reduce("") { $0 + String($1) }
let outPath = opts.out ?? profDir.appendingPathComponent("kbench/\(slug)-\(stamp).json").path
do { try out.write(to: outPath) } catch { die("cannot write \(outPath): \(error)") }

// MARK: - report

func fmt(_ v: Double?, _ f: String = "%.1f") -> String { v.map { String(format: f, $0) } ?? "-" }
func pad(_ s: String, _ n: Int, left: Bool = false) -> String {
    s.count >= n ? s : (left ? s + String(repeating: " ", count: n - s.count) : String(repeating: " ", count: n - s.count) + s)
}
print("\(spec.name) [\(spec.variant)] on \(device.name), \(reps) reps (+\(warmup) warmup)")
if let d = spec.description { print(d) }
let nameW = max(12, (results.map { $0.name.count }.max() ?? 0) + 1)
print(pad("step", nameW, left: true) + pad("median us", 11) + pad("min", 9) + pad("p90", 9) + pad("spread%", 9)
      + pad("GB/s", 9) + pad("%bw", 6) + pad("GFLOP/s", 10) + pad("%flop", 7))
for r in results {
    print(pad(r.name, nameW, left: true) + pad(fmt(r.medianUs, "%.2f"), 11) + pad(fmt(r.minUs, "%.2f"), 9)
          + pad(fmt(r.p90Us, "%.2f"), 9) + pad(fmt(100 * (r.p90Us - r.p10Us) / r.medianUs), 9) + pad(fmt(r.gbps), 9)
          + pad(fmt(r.pctPeakBw, "%.0f"), 6) + pad(fmt(r.gflops), 10) + pad(fmt(r.pctPeakFlops, "%.0f"), 7))
    if (r.p90Us - r.p10Us) / r.medianUs > 0.10 {
        print("  ^ noisy: p10-p90 spread > 10% of the median. The median is robust; see the noise line, or raise reps/repeat")
    }
}
if let l = spec.loop { print("(per-step rows: \(reps) reps x \(l.count) \(l.name)s = \(reps * l.count) samples each)") }
if let p = pipelineLine { print(p) }
if let noise {
    print(NoiseMonitor.summaryLine(noise))
    for n in noise.notes { print("  note: \(n)") }
}
if let m = peakMessage { print("\(m): \(opts.peaks)") }
else if peaks.bandwidthGbps == nil && peaks.flopsGflops == nil { print("(no peaks yet: run examples/roofline/*.json to get % of roof)") }
print("wrote \(outPath)")
