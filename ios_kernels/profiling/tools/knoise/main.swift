// knoise: record the noise around any command (llama.cpp tools, our engine,
// an xctrace recording...) in kprof format.
//
//   knoise [--out PATH] [--idle-ms MS] [--name NAME] -- COMMAND [ARGS...]
//
// It takes the idle GPU baseline, runs COMMAND (stdin/stdout/stderr pass
// through), samples GPU utilization while it runs, and writes a kprof file
// with meta (device, git, argv), noise, a GPU-util counter track, and one
// CPU-track event spanning the command. Exits with COMMAND's status.

import Foundation
import Metal

func die(_ m: String) -> Never {
    FileHandle.standardError.write("knoise: \(m)\n".data(using: .utf8)!)
    exit(2)
}

let profilingDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
    .deletingLastPathComponent().deletingLastPathComponent()
let repoRoot = profilingDir.deletingLastPathComponent().deletingLastPathComponent()
let dataDir = ProcessInfo.processInfo.environment["DATA_DIR"].map { URL(fileURLWithPath: $0) }
    ?? repoRoot.appendingPathComponent(".context")

var out: String?
var idleMs = 1000
var name: String?
var cmd: [String] = []
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--out": out = args.isEmpty ? nil : args.removeFirst()
    case "--idle-ms": idleMs = args.isEmpty ? 1000 : Int(args.removeFirst()) ?? 1000
    case "--name": name = args.isEmpty ? nil : args.removeFirst()
    case "--": cmd = args; args = []
    default: die("unexpected \(a)\nusage: knoise [--out PATH] [--idle-ms MS] [--name NAME] -- COMMAND [ARGS...]")
    }
}
if cmd.isEmpty { die("usage: knoise [--out PATH] [--idle-ms MS] [--name NAME] -- COMMAND [ARGS...]") }
guard let device = MTLCreateSystemDefaultDevice() else { die("no Metal device") }

let monitor = NoiseMonitor()
let startSnap = NoiseMonitor.snapshot(topProcesses: true)
let idle = monitor.sampleIdle(ms: idleMs)

let p = Process()
p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
p.arguments = cmd
monitor.start()
let t0 = nowNs()
do { try p.run() } catch { die("cannot run \(cmd[0]): \(error)") }
p.waitUntilExit()
let t1 = nowNs()
let run = monitor.stop()
let endSnap = NoiseMonitor.snapshot(topProcesses: false)

let idleStats = NoiseMonitor.stats(idle)
let noise = KProf.Noise(start: startSnap, end: endSnap, gpuUtilIdle: idleStats, gpuUtilRun: NoiseMonitor.stats(run),
                        notes: NoiseMonitor.notes(start: startSnap, end: endSnap, idle: idleStats))
let label = name ?? (cmd[0] as NSString).lastPathComponent
let meta = KProf.Meta(tool: "knoise", name: label, created: isoNow(), device: deviceMeta(device),
                      os: ProcessInfo.processInfo.operatingSystemVersionString, git: gitDescribe(profilingDir.path),
                      argv: CommandLine.arguments, spec: nil, params: ["exit_status": String(p.terminationStatus)])
var counters = idle.map { KProf.Counter(track: "GPU util % (device-wide, idle baseline)", tNs: $0.0, value: $0.1) }
counters += run.map { KProf.Counter(track: "GPU util % (device-wide)", tNs: $0.0, value: $0.1) }
let event = KProf.Event(name: label, kernel: nil, stack: [label], track: "CPU (process)", startNs: t0, durNs: t1 - t0,
                        bytes: nil, flops: nil, variant: nil, args: ["command": cmd.joined(separator: " ")])
let k = KProf(meta: meta, noise: noise, events: [event], counters: counters, results: [])

let stamp: String = { let f = DateFormatter(); f.dateFormat = "yyyyMMdd-HHmmss"; return f.string(from: Date()) }()
let path = out ?? dataDir.appendingPathComponent("prof/knoise/\(label)-\(stamp).json").path
do { try k.write(to: path) } catch { die("cannot write \(path): \(error)") }
FileHandle.standardError.write("knoise: \(NoiseMonitor.summaryLine(noise))\nknoise: wrote \(path)\n".data(using: .utf8)!)
exit(p.terminationStatus)
