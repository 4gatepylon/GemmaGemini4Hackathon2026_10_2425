// The common profiling output format, `kprof/1`. Every producer (kbench,
// RegionProfiler users, the xctrace and llama.cpp exporters) writes it and
// every consumer (charts, comparisons) reads it. Keys are snake_case JSON.
// The field reference is in ../README.md ("Output format").

import Foundation
import Metal

struct KProf: Codable {
    var schema = "kprof/1"
    var meta: Meta
    var noise: Noise?
    var events: [Event] = []
    var counters: [Counter] = []
    var results: [Result] = []

    struct Meta: Codable {
        var tool: String
        var name: String
        var created: String
        var device: Device
        var os: String
        var git: String?
        var argv: [String]
        var spec: String?
        var params: [String: String]?
    }

    struct Device: Codable {
        var name: String
        var model: String?
        var gpuCores: Int?
        var recommendedWorkingSetMb: Int
        var maxThreadgroupMemory: Int
    }

    /// One timed span of GPU (or CPU) work. Times are ns on the CLOCK_UPTIME_RAW
    /// clock (Apple silicon GPU timestamps are in that domain). `stack` is the
    /// nesting path, outermost first, for the timeline and flame views.
    struct Event: Codable {
        var name: String
        var kernel: String?
        var stack: [String]
        var track: String
        var startNs: UInt64
        var durNs: UInt64
        var bytes: Double?
        var flops: Double?
        var variant: String?
        var args: [String: String]?
    }

    struct Counter: Codable {
        var track: String
        var tNs: UInt64
        var value: Double
    }

    /// One aggregate row: a step (or op) of a variant, over its repetitions.
    struct Result: Codable {
        var name: String
        var variant: String?
        var method: String
        var n: Int
        var medianUs: Double
        var meanUs: Double
        var stdUs: Double
        var minUs: Double
        var p10Us: Double
        var p90Us: Double
        var bytes: Double?
        var flops: Double?
        var gbps: Double?
        var gflops: Double?
        var pctPeakBw: Double?
        var pctPeakFlops: Double?
        var extra: [String: String]?
    }

    struct Noise: Codable {
        var start: Snapshot
        var end: Snapshot
        var gpuUtilIdle: UtilStats?
        var gpuUtilRun: UtilStats?
        var notes: [String]
    }

    struct Snapshot: Codable {
        var tNs: UInt64
        var power: String
        var lowPowerMode: Bool
        var thermal: String
        var loadavg: [Double]
        var topCpu: [String]?
    }

    struct UtilStats: Codable {
        var n: Int
        var mean: Double
        var max: Double
        var samples: [Double]
    }

    func write(to path: String) throws {
        let enc = JSONEncoder()
        enc.keyEncodingStrategy = .convertToSnakeCase
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try enc.encode(self).write(to: url)
    }
}

// MARK: - metadata helpers

func nowNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

func isoNow() -> String { ISO8601DateFormatter().string(from: Date()) }

func sysctlString(_ name: String) -> String? {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return nil }
    return String(cString: buf)
}

func deviceMeta(_ d: MTLDevice) -> KProf.Device {
    KProf.Device(name: d.name, model: sysctlString("hw.model"), gpuCores: GPUStats.coreCount(),
                 recommendedWorkingSetMb: Int(d.recommendedMaxWorkingSetSize >> 20),
                 maxThreadgroupMemory: d.maxThreadgroupMemoryLength)
}

/// Run a command and return its stdout (nil on failure). For metadata only.
func shell(_ args: [String], cwd: String? = nil) -> String? {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
    p.arguments = args
    if let cwd { p.currentDirectoryURL = URL(fileURLWithPath: cwd) }
    let out = Pipe()
    p.standardOutput = out
    p.standardError = FileHandle.nullDevice
    do { try p.run() } catch { return nil }
    let data = out.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    guard p.terminationStatus == 0 else { return nil }
    return String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
}

/// "abc1234" or "abc1234-dirty" for the repo containing `dir`.
func gitDescribe(_ dir: String) -> String? {
    guard let sha = shell(["git", "rev-parse", "--short", "HEAD"], cwd: dir) else { return nil }
    let dirty = shell(["git", "status", "--porcelain", "--untracked-files=no"], cwd: dir).map { !$0.isEmpty } ?? false
    return dirty ? sha + "-dirty" : sha
}
