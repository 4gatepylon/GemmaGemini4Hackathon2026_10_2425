// kbench spec files: JSON that describes kernels, buffers, argument bindings
// and grids. Sizes are expressions over `vars` and numeric `defines`, so one
// spec covers many shapes (`--set M=8192`). Field reference: ../README.md.

import Foundation
import Metal

struct SpecError: Error, CustomStringConvertible {
    let message: String
    var description: String { message }
}

enum ScalarType: String {
    case half, float, int, uint, short, ushort, char, uchar
    var size: Int {
        switch self {
        case .half, .short, .ushort: return 2
        case .float, .int, .uint: return 4
        case .char, .uchar: return 1
        }
    }
}

enum BufferInit {
    case zero, iota
    case constant(Double)
    case random(Double, Double)     // uniform [lo, hi)
    case randomInt(Int, Int)        // uniform integers [lo, hi)
}

struct BufferSpec {
    let name: String
    let type: ScalarType
    let count: Int
    let initial: BufferInit
    let rotate: Int                 // copies cycled across dispatches (defeats cache reuse)
    var bytes: Int { count * type.size }
}

enum BindValue {
    case buffer(String, offset: Int)
    case bytes([UInt8])             // setBytes: scalars and packed structs
}

enum Dispatch: CustomStringConvertible {
    case threadgroups(MTLSize, threadsPerGroup: MTLSize)
    case threads(MTLSize, threadsPerGroup: MTLSize)

    var description: String {
        func f(_ s: MTLSize) -> String { "[\(s.width),\(s.height),\(s.depth)]" }
        switch self {
        case .threadgroups(let g, let t): return "threadgroups \(f(g)) x threads \(f(t))"
        case .threads(let g, let t): return "grid \(f(g)) threads, \(f(t)) per group"
        }
    }
}

struct StepSpec {
    let name: String
    let kernel: String
    let group: String?              // optional nesting level for charts, e.g. "attention"
    let binds: [BindValue]
    let threadgroupMemory: [Int]    // bytes per threadgroup-memory index
    let dispatch: Dispatch
    let bytes: Double?              // bytes moved per dispatch (for GB/s)
    let flops: Double?              // FLOPs per dispatch (for GFLOP/s)
    let repeatCount: Int            // dispatches per timed encoder
}

struct ResolvedSpec {
    let name: String
    let description: String?
    let sourcePath: String
    let defines: [(String, String)]
    let vars: [String: Double]
    let buffers: [BufferSpec]
    let steps: [StepSpec]
    let loop: (name: String, count: Int)?   // run the step list `count` times per rep (e.g. layers)
    let whole: Bool                         // also time the whole rep as one encoder (split overhead)
    let warmup: Int
    let reps: Int
    let roof: String?
    let fastMath: Bool
    let variant: String             // "default" or the overrides, e.g. "NR=2,M=8192"
    var params: [String: String] {
        var p: [String: String] = [:]
        for (k, v) in defines { p[k] = v }
        for (k, v) in vars where p[k] == nil { p[k] = v == v.rounded() ? String(Int(v)) : String(v) }
        return p
    }
}

final class BenchSpec {
    let path: String
    let json: [String: Any]

    init(path: String) throws {
        self.path = (path as NSString).standardizingPath
        guard let data = FileManager.default.contents(atPath: self.path) else { throw SpecError(message: "cannot read \(path)") }
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw SpecError(message: "\(path): top level must be an object")
        }
        json = j
    }

    var dir: String { (path as NSString).deletingLastPathComponent }

    /// Evaluate the spec with `overrides` (name -> value) applied to defines or vars.
    func resolve(overrides: [(String, String)] = []) throws -> ResolvedSpec {
        func fail(_ m: String) -> SpecError { SpecError(message: "\((path as NSString).lastPathComponent): \(m)") }

        // defines (kept as text for the shader) and vars (numbers or expressions)
        var defines: [String: String] = [:]
        for (k, v) in json["defines"] as? [String: Any] ?? [:] { defines[k] = "\(v)" }
        var rawVars = json["vars"] as? [String: Any] ?? [:]
        for (k, v) in overrides {
            if defines[k] != nil { defines[k] = v }
            else if rawVars[k] != nil { rawVars[k] = v }
            else { throw fail("--set \(k): not a define or var of this spec (defines: \(defines.keys.sorted()), vars: \(rawVars.keys.sorted()))") }
        }
        var env: [String: Double] = [:]
        for (k, v) in defines { if let d = Double(v) { env[k] = d } }
        // vars may reference each other: resolve in passes until nothing changes.
        var pending = rawVars
        while !pending.isEmpty {
            var progressed = false
            for (k, v) in pending {
                if let d = try? Expr.number(v, env) { env[k] = d; pending[k] = nil; progressed = true }
            }
            if !progressed {
                let (k, v) = pending.first!
                do { _ = try Expr.number(v, env) } catch { throw fail("var \(k): \(error)") }
                throw fail("cannot resolve var \(k)")
            }
        }

        func int(_ v: Any?, _ what: String, default d: Int? = nil) throws -> Int {
            guard let v else {
                if let d { return d }
                throw fail("missing \(what)")
            }
            do { return try Expr.int(v, env) } catch { throw fail("\(what): \(error)") }
        }
        func num(_ v: Any?, _ what: String) throws -> Double? {
            guard let v else { return nil }
            do { return try Expr.number(v, env) } catch { throw fail("\(what): \(error)") }
        }
        func size3(_ v: Any?, _ what: String) throws -> MTLSize {
            guard let a = v as? [Any], (1...3).contains(a.count) else { throw fail("\(what) must be an array of 1-3 sizes") }
            var s = try a.enumerated().map { try int($0.element, "\(what)[\($0.offset)]") }
            while s.count < 3 { s.append(1) }
            return MTLSize(width: s[0], height: s[1], depth: s[2])
        }

        // buffers
        var buffers: [BufferSpec] = []
        for (name, raw) in (json["buffers"] as? [String: Any] ?? [:]).sorted(by: { $0.key < $1.key }) {
            guard let b = raw as? [String: Any] else { throw fail("buffer \(name) must be an object") }
            guard let t = ScalarType(rawValue: b["type"] as? String ?? "") else {
                throw fail("buffer \(name): type must be one of half, float, int, uint, short, ushort, char, uchar")
            }
            let initial: BufferInit
            let initMsg = "buffer \(name): init must be zero|ones|random|iota|{random:[lo,hi]}|{random_int:[lo,hi]}|{const:v}"
            if b["init"] == nil {
                initial = .zero
            } else if let s = b["init"] as? String {
                switch s {
                case "zero": initial = .zero
                case "random": initial = .random(-1, 1)
                case "ones": initial = .constant(1)
                case "iota": initial = .iota
                default: throw fail(initMsg)
                }
            } else if let d = b["init"] as? [String: Any] {
                if let r = d["random"] as? [Any], r.count == 2 {
                    initial = .random(try num(r[0], "\(name).init")!, try num(r[1], "\(name).init")!)
                } else if let r = d["random_int"] as? [Any], r.count == 2 {
                    initial = .randomInt(try int(r[0], "\(name).init"), try int(r[1], "\(name).init"))
                } else if let c = d["const"] {
                    initial = .constant(try num(c, "\(name).init")!)
                } else { throw fail(initMsg) }
            } else { throw fail(initMsg) }
            buffers.append(BufferSpec(name: name, type: t, count: try int(b["count"], "buffer \(name).count"),
                                      initial: initial, rotate: try int(b["rotate"], "buffer \(name).rotate", default: 1)))
        }
        let bufferNames = Set(buffers.map { $0.name })

        func packStruct(_ fields: [[String: Any]], _ what: String) throws -> [UInt8] {
            var out: [UInt8] = []
            var align = 1
            for f in fields {
                guard f.count == 1, let (k, v) = f.first else { throw fail("\(what): each field is {type: value}") }
                var bytes: [UInt8]
                switch k {
                case "u32": bytes = withUnsafeBytes(of: UInt32(try int(v, what)).littleEndian) { Array($0) }
                case "i32": bytes = withUnsafeBytes(of: Int32(try int(v, what)).littleEndian) { Array($0) }
                case "f32": bytes = withUnsafeBytes(of: Float(try num(v, what)!)) { Array($0) }
                case "f16": bytes = withUnsafeBytes(of: Float16(Float(try num(v, what)!))) { Array($0) }
                case "u16": bytes = withUnsafeBytes(of: UInt16(try int(v, what)).littleEndian) { Array($0) }
                default: throw fail("\(what): unknown scalar type \(k) (u32, i32, f32, f16, u16, or buf/struct)")
                }
                while out.count % bytes.count != 0 { out.append(0) }  // natural alignment, like MSL
                align = max(align, bytes.count)
                out += bytes
            }
            while out.count % align != 0 { out.append(0) }
            return out
        }

        // steps
        guard let rawSteps = json["steps"] as? [[String: Any]], !rawSteps.isEmpty else { throw fail("needs a non-empty steps array") }
        var steps: [StepSpec] = []
        for (si, s) in rawSteps.enumerated() {
            guard let kernel = s["kernel"] as? String else { throw fail("step \(si) needs a kernel name") }
            let sname = s["name"] as? String ?? kernel
            var binds: [BindValue] = []
            for (bi, b) in (s["bind"] as? [[String: Any]] ?? []).enumerated() {
                let what = "step \(sname) bind[\(bi)]"
                if let name = b["buf"] as? String {
                    guard bufferNames.contains(name) else { throw fail("\(what): unknown buffer \(name)") }
                    binds.append(.buffer(name, offset: try int(b["offset"], "\(what).offset", default: 0)))
                } else if let fields = b["struct"] as? [[String: Any]] {
                    binds.append(.bytes(try packStruct(fields, what)))
                } else {
                    binds.append(.bytes(try packStruct([b], what)))
                }
            }
            let tpg = try size3(s["threads"], "step \(sname).threads")
            let dispatch: Dispatch
            if let tg = s["threadgroups"] { dispatch = .threadgroups(try size3(tg, "step \(sname).threadgroups"), threadsPerGroup: tpg) }
            else if let g = s["grid"] { dispatch = .threads(try size3(g, "step \(sname).grid"), threadsPerGroup: tpg) }
            else { throw fail("step \(sname) needs threadgroups or grid") }
            let tgm = try (s["threadgroup_memory"] as? [Any] ?? []).enumerated().map { try int($0.element, "step \(sname).threadgroup_memory[\($0.offset)]") }
            steps.append(StepSpec(name: sname, kernel: kernel, group: s["group"] as? String, binds: binds, threadgroupMemory: tgm, dispatch: dispatch,
                                  bytes: try num(s["bytes"], "step \(sname).bytes"),
                                  flops: try num(s["flops"], "step \(sname).flops"),
                                  repeatCount: try int(s["repeat"], "step \(sname).repeat", default: 1)))
        }

        guard let source = json["source"] as? String else { throw fail("needs a source (path to a .metal file, relative to the spec)") }
        let sourcePath = source.hasPrefix("/") ? source : (dir as NSString).appendingPathComponent(source)
        return ResolvedSpec(
            name: json["name"] as? String ?? ((path as NSString).lastPathComponent as NSString).deletingPathExtension,
            description: json["description"] as? String,
            sourcePath: (sourcePath as NSString).standardizingPath,
            defines: defines.sorted { $0.key < $1.key }.map { ($0.key, $0.value) },
            vars: env, buffers: buffers, steps: steps,
            loop: try (json["loop"] as? [String: Any]).map {
                (name: $0["name"] as? String ?? "iter", count: try int($0["count"], "loop.count"))
            },
            whole: json["whole"] as? Bool ?? false,
            warmup: try int(json["warmup"], "warmup", default: 3),
            reps: try int(json["reps"], "reps", default: 20),
            roof: json["roof"] as? String,
            fastMath: json["fast_math"] as? Bool ?? true,
            variant: overrides.isEmpty ? "default" : overrides.map { "\($0.0)=\($0.1)" }.joined(separator: ","))
    }
}

// MARK: - buffer filling

/// Deterministic xorshift64* so runs are reproducible.
struct Rng {
    var s: UInt64
    init(seed: UInt64) { s = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed }
    mutating func next() -> UInt64 {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27
        return s &* 2_685_821_657_736_338_717
    }
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}

func fill(_ buf: MTLBuffer, _ spec: BufferSpec, seed: UInt64) {
    let p = buf.contents()
    let n = spec.count
    if case .zero = spec.initial { memset(p, 0, spec.bytes); return }
    var rng = Rng(seed: seed)
    func value(_ i: Int) -> Double {
        switch spec.initial {
        case .zero: return 0
        case .iota: return Double(i)
        case .constant(let c): return c
        case .random(let lo, let hi): return lo + (hi - lo) * rng.unit()
        case .randomInt(let lo, let hi): return Double(lo + Int(rng.next() % UInt64(max(hi - lo, 1))))
        }
    }
    switch spec.type {
    case .half: let q = p.bindMemory(to: Float16.self, capacity: n); for i in 0..<n { q[i] = Float16(value(i)) }
    case .float: let q = p.bindMemory(to: Float.self, capacity: n); for i in 0..<n { q[i] = Float(value(i)) }
    case .int: let q = p.bindMemory(to: Int32.self, capacity: n); for i in 0..<n { q[i] = Int32(value(i)) }
    case .uint: let q = p.bindMemory(to: UInt32.self, capacity: n); for i in 0..<n { q[i] = UInt32(value(i)) }
    case .short: let q = p.bindMemory(to: Int16.self, capacity: n); for i in 0..<n { q[i] = Int16(value(i)) }
    case .ushort: let q = p.bindMemory(to: UInt16.self, capacity: n); for i in 0..<n { q[i] = UInt16(value(i)) }
    case .char: let q = p.bindMemory(to: Int8.self, capacity: n); for i in 0..<n { q[i] = Int8(value(i)) }
    case .uchar: let q = p.bindMemory(to: UInt8.self, capacity: n); for i in 0..<n { q[i] = UInt8(value(i)) }
    }
}
