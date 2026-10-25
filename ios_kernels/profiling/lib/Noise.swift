// Records the noise present during a measurement. We don't require a quiet
// machine (real workloads aren't quiet). Instead every result carries what
// else was going on: device-wide GPU utilization, power source, thermal state,
// load, and the busiest processes.

import Foundation
import IOKit
import IOKit.ps

enum GPUStats {
    private static func accelerators<T>(_ body: (io_object_t) -> T?) -> T? {
        var it: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) == KERN_SUCCESS else {
            return nil
        }
        defer { IOObjectRelease(it) }
        var result: T?
        var s = IOIteratorNext(it)
        while s != 0 {
            if result == nil { result = body(s) }
            IOObjectRelease(s)
            s = IOIteratorNext(it)
        }
        return result
    }

    /// Device-wide GPU busy %, from IOAccelerator "PerformanceStatistics" (what
    /// `ioreg -r -c IOAccelerator` shows; no sudo needed). It includes our own
    /// work. Measured quirks: the first read after a pause is stale (often
    /// 90-100%), and polling faster than ~100 ms reads 100% even when idle, so
    /// NoiseMonitor polls every 250 ms and drops the first sample.
    static func utilization() -> Double? {
        accelerators { s in
            let p = IORegistryEntryCreateCFProperty(s, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any]
            return (p?["Device Utilization %"] as? NSNumber)?.doubleValue
        }
    }

    static func coreCount() -> Int? {
        accelerators { s in
            (IORegistryEntryCreateCFProperty(s, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? NSNumber)?.intValue
        }
    }
}

final class NoiseMonitor {
    let intervalMs: Int
    private let lock = NSLock()
    private var samples: [(UInt64, Double)] = []
    private var running = false
    private var thread: Thread?

    init(intervalMs: Int = 250) { self.intervalMs = intervalMs }

    static func snapshot(topProcesses: Bool) -> KProf.Snapshot {
        var power = "unknown"
        if let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
           let type = IOPSGetProvidingPowerSourceType(info)?.takeUnretainedValue() {
            power = type as String
        }
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        var la = [Double](repeating: 0, count: 3)
        getloadavg(&la, 3)
        var top: [String]?
        if topProcesses, let ps = shell(["ps", "-Ao", "pcpu=,comm=", "-r"]) {
            top = ps.split(separator: "\n").prefix(6).map {
                // "27.6 /System/.../WindowServer" -> "27.6% WindowServer"
                let parts = $0.trimmingCharacters(in: .whitespaces).split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return String($0) }
                return "\(parts[0])% \((String(parts[1]) as NSString).lastPathComponent)"
            }
        }
        return KProf.Snapshot(tNs: nowNs(), power: power, lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                              thermal: thermal, loadavg: la, topCpu: top)
    }

    /// Blocking: sample GPU utilization for `ms` while we're idle. That's the
    /// baseline other processes put on the GPU.
    func sampleIdle(ms: Int) -> [(UInt64, Double)] {
        _ = GPUStats.utilization()  // first read is stale
        var out: [(UInt64, Double)] = []
        let n = max(1, ms / intervalMs)
        for _ in 0..<n {
            usleep(UInt32(intervalMs * 1000))
            if let u = GPUStats.utilization() { out.append((nowNs(), u)) }
        }
        return out
    }

    /// Start sampling in the background (during the measured run).
    func start() {
        lock.lock(); samples = []; running = true; lock.unlock()
        _ = GPUStats.utilization()
        let t = Thread { [self] in
            while true {
                usleep(UInt32(intervalMs * 1000))
                lock.lock()
                let keepGoing = running
                lock.unlock()
                if !keepGoing { break }
                if let u = GPUStats.utilization() {
                    lock.lock(); samples.append((nowNs(), u)); lock.unlock()
                }
            }
        }
        thread = t
        t.start()
    }

    func stop() -> [(UInt64, Double)] {
        lock.lock(); running = false; let s = samples; lock.unlock()
        return s
    }

    static func stats(_ s: [(UInt64, Double)]) -> KProf.UtilStats? {
        guard !s.isEmpty else { return nil }
        let v = s.map { $0.1 }
        return KProf.UtilStats(n: v.count, mean: v.reduce(0, +) / Double(v.count), max: v.max()!, samples: v)
    }

    /// Human-readable warnings derived from the snapshots and samples.
    static func notes(start: KProf.Snapshot, end: KProf.Snapshot, idle: KProf.UtilStats?) -> [String] {
        var n: [String] = []
        if start.power != "AC Power" || end.power != "AC Power" { n.append("not on AC power (\(start.power) -> \(end.power))") }
        if start.lowPowerMode || end.lowPowerMode { n.append("Low Power Mode is on") }
        if start.thermal != "nominal" || end.thermal != "nominal" { n.append("thermal state \(start.thermal) -> \(end.thermal)") }
        if let idle, idle.mean >= 5 { n.append(String(format: "GPU busy %.0f%% (max %.0f%%) before our run: other processes are using the GPU", idle.mean, idle.max)) }
        return n
    }

    static func summaryLine(_ noise: KProf.Noise) -> String {
        var s = "noise: \(noise.start.power), thermal \(noise.start.thermal)->\(noise.end.thermal)"
        if let i = noise.gpuUtilIdle { s += String(format: ", GPU idle-baseline %.0f%% (max %.0f%%, n=%d)", i.mean, i.max, i.n) }
        s += String(format: ", load %.1f", noise.start.loadavg.first ?? 0)
        if let top = noise.start.topCpu?.prefix(3) { s += ", top: " + top.joined(separator: ", ") }
        return s
    }
}
