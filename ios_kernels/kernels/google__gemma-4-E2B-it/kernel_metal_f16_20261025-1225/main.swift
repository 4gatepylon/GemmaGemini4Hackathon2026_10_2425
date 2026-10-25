// Minimal Gemma 4 E2B (text) inference engine on hand-written Metal kernels.
//
// Build:  swiftc -O -o .build/llm main.swift   (see build.py)
// Usage:
//   llm --weights W.bin --prompt-ids 2,105,2364 --gen 32       greedy generation, prints ids as JSON
//   llm --weights W.bin --bench-tg 128 --bench-pp 512 --reps 5 llama-bench style throughput
//   llm --weights W.bin --serve                                stay resident, JSON lines on stdin/stdout
//
// --serve protocol (one JSON object per line; see ios_kernels/surfaces/README.md):
//   out (once): {"ready": true, "max_ctx": N, "vocab": V, "device": "..."}
//   in:         {"ids": [int], "max_tokens": N, "stop_ids": [int]}
//   out:        {"token": id}   for every generated token, as soon as it's ready (stop token not sent)
//   out:        {"done": true, "finish_reason": "stop"|"length", "prompt_tokens": P,
//                "completion_tokens": n, "prefill_s": s, "decode_s": s}
//   out:        {"error": "..."} for a bad request (the server keeps running)
//   EOF on stdin exits.
// Options:
//   --kernels PATH        kernels.metal (default: next to this source file)
//   --define NAME=VALUE   extra #define for the kernels (repeatable), e.g. MV_NR=8
//   --max-ctx N           KV cache length (default 4096)
//   --dump-logits PATH    write fp32 logits (after final softcapping) of the last prompt position
//   --no-gemm             prefill token-by-token through the decode path

import Foundation
import Metal

// MARK: - CLI

struct Options {
    var weights = ""
    var kernels = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .appendingPathComponent("kernels.metal").path
    var defines: [String] = []
    var maxCtx = 4096
    var promptIds: [Int32] = []
    var gen = 0
    var benchTG = 0
    var benchPP = 0
    var reps = 5
    var dumpLogits: String?
    var useGemm = true
    var serve = false
}

func parseOptions() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    func next(_ flag: String) -> String {
        guard let v = it.next() else { fatalError("missing value for \(flag)") }
        return v
    }
    while let a = it.next() {
        switch a {
        case "--weights": o.weights = next(a)
        case "--kernels": o.kernels = next(a)
        case "--define": o.defines.append(next(a))
        case "--max-ctx": o.maxCtx = Int(next(a))!
        case "--prompt-ids": o.promptIds = next(a).split(separator: ",").map { Int32($0)! }
        case "--gen": o.gen = Int(next(a))!
        case "--bench-tg": o.benchTG = Int(next(a))!
        case "--bench-pp": o.benchPP = Int(next(a))!
        case "--reps": o.reps = Int(next(a))!
        case "--dump-logits": o.dumpLogits = next(a)
        case "--no-gemm": o.useGemm = false
        case "--serve": o.serve = true
        default: fatalError("unknown argument \(a)")
        }
    }
    precondition(!o.weights.isEmpty, "--weights is required")
    return o
}

// MARK: - Model file

/// Per-layer facts, resolved by convert_weights.py (header "engine").
struct Layer {
    let headDim: Int, ffnDim: Int, sliding: Bool, kvLayer: Int, layerScalar: Float
    let invFreq: [Float]  // headDim / 2 entries; 0 past the rotated pairs (identity)
}

struct Config {
    let dim: Int, nLayers: Int, nHeads: Int, pleDim: Int, vocab: Int, eps: Float
    let window: Int, softcap: Float, bos: Int32
    let layers: [Layer]

    init(_ c: [String: Any], engine: [String: Any]) {
        dim = c["hidden_size"] as! Int
        nLayers = c["num_hidden_layers"] as! Int
        nHeads = c["num_attention_heads"] as! Int
        pleDim = c["hidden_size_per_layer_input"] as! Int
        vocab = c["vocab_size"] as! Int
        eps = Float(c["rms_norm_eps"] as! Double)
        window = c["sliding_window"] as! Int
        softcap = Float((c["final_logit_softcapping"] as? Double) ?? 0)
        bos = Int32(c["bos_token_id"] as! Int)
        precondition(c["num_key_value_heads"] as! Int == 1, "kernels assume one KV head")
        layers = (engine["layers"] as! [[String: Any]]).map { l in
            let hd = l["head_dim"] as! Int
            let pairs = l["rope_pairs"] as! Int
            let theta = l["rope_theta"] as! Double
            // Mirrors transformers' default / "proportional" RoPE inverse frequencies.
            let inv = (0..<hd / 2).map { i in i < pairs ? Float(1.0 / pow(theta, Double(2 * i) / Double(hd))) : 0 }
            return Layer(headDim: hd, ffnDim: l["ffn_dim"] as! Int, sliding: l["sliding"] as! Bool,
                         kvLayer: l["kv_layer"] as! Int, layerScalar: Float(l["layer_scalar"] as! Double),
                         invFreq: inv)
        }
    }
}

/// The whole weights file, mmapped and wrapped in a single zero-copy MTLBuffer.
final class ModelFile {
    let buffer: MTLBuffer
    let config: Config
    private let offsets: [String: Int]

    init(path: String, device: MTLDevice) {
        let fd = open(path, O_RDONLY)
        precondition(fd >= 0, "cannot open \(path)")
        var st = stat()
        fstat(fd, &st)
        let size = Int(st.st_size)
        guard let ptr = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), ptr != MAP_FAILED else {
            fatalError("mmap failed")
        }
        close(fd)
        let hlen = Int(ptr.load(as: UInt64.self))
        let hdata = Data(bytes: ptr + 8, count: hlen)
        let header = try! JSONSerialization.jsonObject(with: hdata) as! [String: Any]
        config = Config(header["config"] as! [String: Any], engine: header["engine"] as! [String: Any])
        var offs: [String: Int] = [:]
        for (name, meta) in header["tensors"] as! [String: [String: Any]] {
            offs[name] = meta["offset"] as? Int
        }
        offsets = offs
        precondition(size <= device.maxBufferLength, "weights exceed maxBufferLength")
        buffer = device.makeBuffer(bytesNoCopy: ptr, length: size, options: .storageModeShared,
                                   deallocator: nil)!
    }

    func offset(_ name: String) -> Int {
        guard let o = offsets[name] else { fatalError("missing tensor \(name)") }
        return o
    }
}

// MARK: - Engine

final class Engine {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let model: ModelFile
    let cfg: Config
    let maxCtx: Int
    let maxBatch: Int
    var pipelines: [String: MTLComputePipelineState] = [:]
    var tunables: [String: Int] = [:]

    // Activations (sized for maxBatch rows so prefill can reuse them).
    let x, xn, xnHalf, qkv, att, attHalf, hid, hidHalf, tmp, ple, pleProj, pleg, plegHalf: MTLBuffer
    let logits, tokens, invFreq, kvcache: MTLBuffer
    var kvOffset: [Int] = []  // per layer: byte offset of its K cache in kvcache (V follows)

    var LP: Int { cfg.nLayers * cfg.pleDim }  // floats per PLE row
    func qDim(_ l: Int) -> Int { cfg.nHeads * cfg.layers[l].headDim }
    func ownsKV(_ l: Int) -> Bool { cfg.layers[l].kvLayer == l }
    func qkvDim(_ l: Int) -> Int { qDim(l) + (ownsKV(l) ? 2 * cfg.layers[l].headDim : 0) }

    init(model: ModelFile, kernelsPath: String, defines: [String], maxCtx: Int, maxBatch: Int) {
        let device = model.buffer.device
        self.device = device
        queue = device.makeCommandQueue()!
        self.model = model
        cfg = model.config
        self.maxCtx = maxCtx
        self.maxBatch = maxBatch

        tunables = ["MV_NSG": 8, "MV_NR": 4, "ATT_NSG": 8, "GEMM_BM": 64, "GEMM_BN": 64]
        for d in defines {
            let kv = d.split(separator: "=")
            tunables[String(kv[0])] = Int(kv[1])!
        }
        var src = tunables.map { "#define \($0.key) \($0.value)\n" }.joined()
        src += try! String(contentsOfFile: kernelsPath, encoding: .utf8)
        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        let lib: MTLLibrary
        do { lib = try device.makeLibrary(source: src, options: opts) } catch {
            fatalError("kernel compile failed:\n\(error)")
        }
        for name in lib.functionNames {
            pipelines[name] = try! device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
        }

        func buf(_ bytes: Int) -> MTLBuffer { device.makeBuffer(length: bytes, options: .storageModeShared)! }
        let B = (maxBatch + 63) / 64 * 64  // GEMM tiles read/write padding rows
        let cfg = model.config
        let n = cfg.nLayers
        let maxQKV = (0..<n).map { cfg.nHeads * cfg.layers[$0].headDim + 2 * cfg.layers[$0].headDim }.max()!
        let maxQ = cfg.layers.map { cfg.nHeads * $0.headDim }.max()!
        let maxFFN = cfg.layers.map(\.ffnDim).max()!
        let lp = n * cfg.pleDim
        x = buf(B * cfg.dim * 4)
        xn = buf(B * cfg.dim * 4)
        xnHalf = buf(B * cfg.dim * 2)
        tmp = buf(B * cfg.dim * 4)
        qkv = buf(B * maxQKV * 4)
        att = buf(B * maxQ * 4)
        attHalf = buf(B * maxQ * 2)
        hid = buf(maxFFN * 4)  // decode only; prefill goes through hidHalf
        hidHalf = buf(B * maxFFN * 2)
        ple = buf(B * lp * 4)
        pleProj = buf(B * lp * 4)
        pleg = buf(B * cfg.pleDim * 4)
        plegHalf = buf(B * cfg.pleDim * 2)
        logits = buf(cfg.vocab * 4)
        tokens = buf((maxCtx + 1) * 4)
        let inv = cfg.layers.flatMap { $0.invFreq + [Float](repeating: 0, count: 256 - $0.invFreq.count) }
        invFreq = device.makeBuffer(bytes: inv, length: inv.count * 4, options: .storageModeShared)!
        var kvBytes = 0
        for l in 0..<n {
            kvOffset.append(kvBytes)
            if cfg.layers[l].kvLayer == l { kvBytes += 2 * maxCtx * cfg.layers[l].headDim * 2 }
        }
        kvcache = buf(kvBytes)
    }

    var w: MTLBuffer { model.buffer }
    func off(_ n: String) -> Int { model.offset(n) }

    func pipeline(_ name: String) -> MTLComputePipelineState {
        guard let p = pipelines[name] else { fatalError("no kernel \(name)") }
        return p
    }

    // MARK: encoding helpers

    func setU32(_ enc: MTLComputeCommandEncoder, _ v: Int, _ i: Int) {
        var u = UInt32(v)
        enc.setBytes(&u, length: 4, index: i)
    }

    func setF32(_ enc: MTLComputeCommandEncoder, _ v: Float, _ i: Int) {
        var f = v
        enc.setBytes(&f, length: 4, index: i)
    }

    /// dst[t, :] = table[tokens[pos0 + t], :] * scale for t < rows.
    func embed(_ enc: MTLComputeCommandEncoder, _ table: String, width: Int, scale: Float,
               _ dst: MTLBuffer, pos0: Int, rows: Int) {
        enc.setComputePipelineState(pipeline("embed_rows"))
        enc.setBuffer(w, offset: off(table), index: 0)
        enc.setBuffer(tokens, offset: pos0 * 4, index: 1)
        enc.setBuffer(dst, offset: 0, index: 2)
        setU32(enc, width, 3)
        setF32(enc, scale, 4)
        enc.dispatchThreads(MTLSize(width: width, height: rows, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func rmsnorm(_ enc: MTLComputeCommandEncoder, _ src: MTLBuffer, _ weight: String,
                 _ dst: MTLBuffer, rows: Int, half: Bool, srcOffset: Int = 0) {
        enc.setComputePipelineState(pipeline(half ? "rmsnorm_f16" : "rmsnorm_f32"))
        enc.setBuffer(src, offset: srcOffset, index: 0)
        enc.setBuffer(w, offset: off(weight), index: 1)
        enc.setBuffer(dst, offset: 0, index: 2)
        setU32(enc, cfg.dim, 3)
        setF32(enc, cfg.eps, 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// x = (x + rmsnorm(src) * weight) * post, per row.
    func rmsnormAdd(_ enc: MTLComputeCommandEncoder, _ src: MTLBuffer, _ weight: String, rows: Int,
                    post: Float = 1) {
        enc.setComputePipelineState(pipeline("rmsnorm_add"))
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(w, offset: off(weight), index: 1)
        enc.setBuffer(x, offset: 0, index: 2)
        setU32(enc, cfg.dim, 3)
        setF32(enc, cfg.eps, 4)
        setF32(enc, post, 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    /// ple = (rmsnorm(pleProj / sqrt(D)) * w + ple) / sqrt(2), per (layer, row).
    func pleCombine(_ enc: MTLComputeCommandEncoder, rows: Int) {
        enc.setComputePipelineState(pipeline("ple_combine"))
        enc.setBuffer(pleProj, offset: 0, index: 0)
        enc.setBuffer(ple, offset: 0, index: 1)
        enc.setBuffer(w, offset: off("ple_norm"), index: 2)
        setU32(enc, LP, 3)
        setF32(enc, 1 / Float(cfg.dim).squareRoot(), 4)
        setF32(enc, cfg.eps, 5)
        enc.dispatchThreadgroups(MTLSize(width: cfg.nLayers, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: cfg.pleDim, height: 1, depth: 1))
    }

    var mvRowsPerTG: Int { tunables["MV_NSG"]! * tunables["MV_NR"]! }

    enum Epilogue: String { case store = "matvec_f16", add = "matvec_f16_add", geluMul = "matvec_f16_gelu_mul" }

    func matvec(_ enc: MTLComputeCommandEncoder, _ weight: String, _ src: MTLBuffer,
                _ dst: MTLBuffer, M: Int, K: Int, epi: Epilogue = .store,
                aux: MTLBuffer? = nil, auxOffset: Int = 0) {
        precondition(K % 256 == 0 && M % mvRowsPerTG == 0, "matvec shape \(M)x\(K) unsupported")
        enc.setComputePipelineState(pipeline(epi.rawValue))
        enc.setBuffer(w, offset: off(weight), index: 0)
        enc.setBuffer(src, offset: 0, index: 1)
        enc.setBuffer(dst, offset: 0, index: 2)
        setU32(enc, M, 3)
        setU32(enc, K, 4)
        enc.setBuffer(aux ?? dst, offset: auxOffset, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: M / mvRowsPerTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * tunables["MV_NSG"]!, height: 1, depth: 1))
    }

    func gateUp(_ enc: MTLComputeCommandEncoder, layer l: Int) {
        let M = cfg.layers[l].ffnDim, K = cfg.dim
        precondition(K % 256 == 0 && M % mvRowsPerTG == 0)
        enc.setComputePipelineState(pipeline("matvec_gate_up_gelu"))
        enc.setBuffer(w, offset: off("l\(l).wgate"), index: 0)
        enc.setBuffer(w, offset: off("l\(l).wup"), index: 1)
        enc.setBuffer(xn, offset: 0, index: 2)
        enc.setBuffer(hid, offset: 0, index: 3)
        setU32(enc, M, 4)
        setU32(enc, K, 5)
        enc.dispatchThreadgroups(MTLSize(width: M / mvRowsPerTG, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * tunables["MV_NSG"]!, height: 1, depth: 1))
    }

    /// K cache and V cache (byte offsets into kvcache) that layer l reads.
    func kv(_ l: Int) -> (Int, Int) {
        let src = cfg.layers[l].kvLayer
        return (kvOffset[src], kvOffset[src] + maxCtx * cfg.layers[src].headDim * 2)
    }

    func qkvNormRope(_ enc: MTLComputeCommandEncoder, layer l: Int, pos0: Int, rows: Int) {
        struct QKVParams { var nHeads, headDim, ropePairs, rowLen, pos0: UInt32; var eps: Float }
        let lay = cfg.layers[l]
        var p = QKVParams(nHeads: UInt32(cfg.nHeads), headDim: UInt32(lay.headDim),
                          ropePairs: UInt32(lay.invFreq.filter { $0 != 0 }.count),
                          rowLen: UInt32(qkvDim(l)), pos0: UInt32(pos0), eps: cfg.eps)
        let (k, v) = kv(l)
        enc.setComputePipelineState(pipeline("qkv_norm_rope"))
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(kvcache, offset: k, index: 1)
        enc.setBuffer(kvcache, offset: v, index: 2)
        enc.setBuffer(w, offset: off("l\(l).q_norm"), index: 3)
        enc.setBuffer(w, offset: off(ownsKV(l) ? "l\(l).k_norm" : "l\(l).q_norm"), index: 4)
        enc.setBuffer(invFreq, offset: l * 256 * 4, index: 5)
        enc.setBytes(&p, length: MemoryLayout<QKVParams>.size, index: 6)
        let heads = cfg.nHeads + (ownsKV(l) ? 2 : 0)  // KV-shared layers only normalize/rotate q
        enc.dispatchThreadgroups(MTLSize(width: heads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: lay.headDim / 2, height: 1, depth: 1))
    }

    func attention(_ enc: MTLComputeCommandEncoder, layer l: Int, pos0: Int, rows: Int) {
        struct AttnParams { var nHeads, pos0, qStride, oStride, window: UInt32; var scale: Float }
        let lay = cfg.layers[l]
        let window = lay.sliding ? cfg.window : 0
        var p = AttnParams(nHeads: UInt32(cfg.nHeads), pos0: UInt32(pos0), qStride: UInt32(qkvDim(l)),
                           oStride: UInt32(qDim(l)), window: UInt32(window), scale: 1)  // QK-norm, no 1/sqrt(d)
        let (k, v) = kv(l)
        enc.setComputePipelineState(pipeline("attention_\(lay.headDim)"))
        enc.setBuffer(qkv, offset: 0, index: 0)
        enc.setBuffer(kvcache, offset: k, index: 1)
        enc.setBuffer(kvcache, offset: v, index: 2)
        enc.setBuffer(att, offset: 0, index: 3)
        enc.setBytes(&p, length: MemoryLayout<AttnParams>.size, index: 4)
        let maxKeys = window > 0 ? min(pos0 + rows, window) : pos0 + rows
        let floats = max(maxKeys, tunables["ATT_NSG"]! * lay.headDim)
        enc.setThreadgroupMemoryLength(floats * 4 + 15 & ~15, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: cfg.nHeads, height: rows, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32 * tunables["ATT_NSG"]!, height: 1, depth: 1))
    }

    func lmHeadAndArgmax(_ enc: MTLComputeCommandEncoder, outIndex: Int) {
        // Tied embedding. The final softcap is monotonic, so argmax skips it.
        matvec(enc, "embed", xn, logits, M: cfg.vocab, K: cfg.dim)
        enc.setComputePipelineState(pipeline("argmax"))
        enc.setBuffer(logits, offset: 0, index: 0)
        enc.setBuffer(tokens, offset: 0, index: 1)
        setU32(enc, cfg.vocab, 2)
        setU32(enc, outIndex, 3)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 1024, height: 1, depth: 1))
    }

    // MARK: forward passes

    /// One decode step: reads tokens[pos], updates the KV cache at `pos`, and if
    /// `sample` is set writes argmax(logits) to tokens[pos + 1].
    func encodeDecode(_ cb: MTLCommandBuffer, pos: Int, sample: Bool) {
        let enc = cb.makeComputeCommandEncoder()!
        let D = cfg.dim, P = cfg.pleDim
        embed(enc, "embed", width: D, scale: Float(D).squareRoot(), x, pos0: pos, rows: 1)
        embed(enc, "ple_embed", width: LP, scale: Float(P).squareRoot(), ple, pos0: pos, rows: 1)
        matvec(enc, "ple_proj", x, pleProj, M: LP, K: D)
        pleCombine(enc, rows: 1)

        for l in 0..<cfg.nLayers {
            let lay = cfg.layers[l]
            rmsnorm(enc, x, "l\(l).attn_norm", xn, rows: 1, half: false)
            matvec(enc, "l\(l).wqkv", xn, qkv, M: qkvDim(l), K: D)
            qkvNormRope(enc, layer: l, pos0: pos, rows: 1)
            attention(enc, layer: l, pos0: pos, rows: 1)
            matvec(enc, "l\(l).wo", att, tmp, M: D, K: qDim(l))
            rmsnormAdd(enc, tmp, "l\(l).post_attn_norm", rows: 1)
            rmsnorm(enc, x, "l\(l).ffn_norm", xn, rows: 1, half: false)
            gateUp(enc, layer: l)
            matvec(enc, "l\(l).wdown", hid, tmp, M: D, K: lay.ffnDim)
            rmsnormAdd(enc, tmp, "l\(l).post_ffn_norm", rows: 1)
            matvec(enc, "l\(l).ple_gate", x, pleg, M: P, K: D, epi: .geluMul, aux: ple, auxOffset: l * P * 4)
            matvec(enc, "l\(l).ple_out", pleg, tmp, M: D, K: P)
            rmsnormAdd(enc, tmp, "l\(l).post_ple_norm", rows: 1, post: lay.layerScalar)
        }
        if sample {
            rmsnorm(enc, x, "final_norm", xn, rows: 1, half: false)
            lmHeadAndArgmax(enc, outIndex: pos + 1)
        }
        enc.endEncoding()
    }

    func run(_ encode: (MTLCommandBuffer) -> Void) -> MTLCommandBuffer {
        let cb = queue.makeCommandBuffer()!
        encode(cb)
        cb.commit()
        return cb
    }

    /// Prompt processing for tokens[0..<n] (positions 0..<n). Samples the next token.
    func prefill(n: Int, useGemm: Bool) {
        if useGemm, n > 1 {
            var done = 0
            while done < n {
                let rows = min(maxBatch, n - done)
                let last = done + rows == n
                let p0 = done
                _ = run { self.encodePrefillChunk($0, pos0: p0, rows: rows, sample: last) }
                done += rows
            }
        } else {
            for p in 0..<n { _ = run { self.encodeDecode($0, pos: p, sample: p == n - 1) } }
        }
        let cb = run { _ in }
        cb.waitUntilCompleted()
    }

    /// Greedy decode `count` tokens starting at `pos` (tokens[pos] must be set). Pipelined:
    /// the CPU queues all command buffers without waiting, since sampling happens on the GPU.
    func decode(from pos: Int, count: Int) {
        var last: MTLCommandBuffer?
        for i in 0..<count { last = run { self.encodeDecode($0, pos: pos + i, sample: true) } }
        last?.waitUntilCompleted()
        if let e = last?.error { fatalError("GPU error: \(e)") }
    }

    func setTokens(_ ids: [Int32]) {
        let p = tokens.contents().bindMemory(to: Int32.self, capacity: maxCtx + 1)
        for (i, t) in ids.enumerated() { p[i] = t }
    }

    func token(at i: Int) -> Int32 {
        tokens.contents().bindMemory(to: Int32.self, capacity: maxCtx + 1)[i]
    }

    // MARK: prefill (batched GEMM path)

    /// GEMM: Y[T, M] (+)= X[T, K] (fp16) · W[M, K]^T, Y fp32.
    func gemm(_ enc: MTLComputeCommandEncoder, _ weight: String, _ src: MTLBuffer, _ dst: MTLBuffer,
              T: Int, M: Int, K: Int, add: Bool = false) {
        let bm = tunables["GEMM_BM"]!, bn = tunables["GEMM_BN"]!
        precondition(K % 32 == 0 && M % bn == 0, "gemm shape \(M)x\(K) unsupported")
        enc.setComputePipelineState(pipeline(add ? "gemm_f16_add" : "gemm_f16"))
        enc.setBuffer(w, offset: off(weight), index: 0)
        enc.setBuffer(src, offset: 0, index: 1)
        enc.setBuffer(dst, offset: 0, index: 2)
        setU32(enc, T, 3)
        setU32(enc, M, 4)
        setU32(enc, K, 5)
        enc.dispatchThreadgroups(MTLSize(width: M / bn, height: (T + bm - 1) / bm, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }

    func encodePrefillChunk(_ cb: MTLCommandBuffer, pos0: Int, rows T: Int, sample: Bool) {
        let enc = cb.makeComputeCommandEncoder()!
        let D = cfg.dim, P = cfg.pleDim
        embed(enc, "embed", width: D, scale: Float(D).squareRoot(), x, pos0: pos0, rows: T)
        embed(enc, "ple_embed", width: LP, scale: Float(P).squareRoot(), ple, pos0: pos0, rows: T)
        toHalf(enc, x, xnHalf, count: T * D)
        gemm(enc, "ple_proj", xnHalf, pleProj, T: T, M: LP, K: D)
        pleCombine(enc, rows: T)

        for l in 0..<cfg.nLayers {
            let lay = cfg.layers[l]
            rmsnorm(enc, x, "l\(l).attn_norm", xnHalf, rows: T, half: true)
            gemm(enc, "l\(l).wqkv", xnHalf, qkv, T: T, M: qkvDim(l), K: D)
            qkvNormRope(enc, layer: l, pos0: pos0, rows: T)
            attention(enc, layer: l, pos0: pos0, rows: T)
            toHalf(enc, att, attHalf, count: T * qDim(l))
            gemm(enc, "l\(l).wo", attHalf, tmp, T: T, M: D, K: qDim(l))
            rmsnormAdd(enc, tmp, "l\(l).post_attn_norm", rows: T)
            rmsnorm(enc, x, "l\(l).ffn_norm", xnHalf, rows: T, half: true)
            gemmGateUp(enc, layer: l, T: T)
            gemm(enc, "l\(l).wdown", hidHalf, tmp, T: T, M: D, K: lay.ffnDim)
            rmsnormAdd(enc, tmp, "l\(l).post_ffn_norm", rows: T)
            toHalf(enc, x, xnHalf, count: T * D)  // the PLE gate reads the un-normalized residual
            gemm(enc, "l\(l).ple_gate", xnHalf, pleg, T: T, M: P, K: D)
            pleGateAct(enc, layer: l, T: T)
            gemm(enc, "l\(l).ple_out", plegHalf, tmp, T: T, M: D, K: P)
            rmsnormAdd(enc, tmp, "l\(l).post_ple_norm", rows: T, post: lay.layerScalar)
        }
        if sample {
            // Only the last row needs logits: normalize it straight into the decode buffer.
            rmsnorm(enc, x, "final_norm", xn, rows: 1, half: false, srcOffset: (T - 1) * D * 4)
            lmHeadAndArgmax(enc, outIndex: pos0 + T)
        }
        enc.endEncoding()
    }

    func toHalf(_ enc: MTLComputeCommandEncoder, _ src: MTLBuffer, _ dst: MTLBuffer, count: Int) {
        enc.setComputePipelineState(pipeline("f32_to_f16"))
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func pleGateAct(_ enc: MTLComputeCommandEncoder, layer l: Int, T: Int) {
        enc.setComputePipelineState(pipeline("ple_gate_act"))
        enc.setBuffer(pleg, offset: 0, index: 0)
        enc.setBuffer(ple, offset: l * cfg.pleDim * 4, index: 1)
        enc.setBuffer(plegHalf, offset: 0, index: 2)
        setU32(enc, cfg.pleDim, 3)
        setU32(enc, LP, 4)
        enc.dispatchThreads(MTLSize(width: cfg.pleDim, height: T, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func gemmGateUp(_ enc: MTLComputeCommandEncoder, layer l: Int, T: Int) {
        let bm = tunables["GEMM_BM"]!, bn = tunables["GEMM_BN"]!
        let M = cfg.layers[l].ffnDim, K = cfg.dim
        precondition(K % 32 == 0 && M % (bn / 2) == 0)
        enc.setComputePipelineState(pipeline("gemm_gate_up_gelu"))
        enc.setBuffer(w, offset: off("l\(l).wgate"), index: 0)
        enc.setBuffer(w, offset: off("l\(l).wup"), index: 1)
        enc.setBuffer(xnHalf, offset: 0, index: 2)
        enc.setBuffer(hidHalf, offset: 0, index: 3)
        setU32(enc, T, 4)
        setU32(enc, M, 5)
        setU32(enc, K, 6)
        enc.dispatchThreadgroups(MTLSize(width: M / (bn / 2), height: (T + bm - 1) / bm, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
}

extension Engine {
    /// Greedy decode with at most `window` command buffers in flight. `emit` gets each
    /// token as soon as its step completes; returning false stops generation (the
    /// already-queued steps are drained and discarded).
    func decodeStreaming(from pos: Int, count: Int, window: Int = 3, emit: (Int32) -> Bool) {
        var inflight: [(cb: MTLCommandBuffer, pos: Int)] = []
        var queued = 0
        var stopped = false
        while !inflight.isEmpty || (!stopped && queued < count) {
            while !stopped && queued < count && inflight.count < window {
                let p = pos + queued
                inflight.append((run { self.encodeDecode($0, pos: p, sample: true) }, p))
                queued += 1
            }
            let step = inflight.removeFirst()
            step.cb.waitUntilCompleted()
            if !stopped && !emit(token(at: step.pos + 1)) { stopped = true }
        }
    }
}

// MARK: - serve mode

func writeLine(_ obj: [String: Any]) {
    var data = try! JSONSerialization.data(withJSONObject: obj)
    data.append(0x0A)
    FileHandle.standardOutput.write(data)
}

func serve(_ engine: Engine, useGemm: Bool) {
    let cfg = engine.cfg
    // Warm up both paths (first dispatches pay one-time driver/pipeline costs).
    engine.setTokens([cfg.bos, cfg.bos])
    engine.prefill(n: 2, useGemm: useGemm)
    engine.decode(from: 2, count: 2)
    writeLine(["ready": true, "max_ctx": engine.maxCtx, "vocab": cfg.vocab, "device": engine.device.name])
    while let line = readLine() {
        guard let data = line.data(using: .utf8),
              let req = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let ids = (req["ids"] as? [Int])?.map(Int32.init), !ids.isEmpty else {
            writeLine(["error": "expected {\"ids\": [int, ...], \"max_tokens\": int}"])
            continue
        }
        if let bad = ids.first(where: { $0 < 0 || Int($0) >= cfg.vocab }) {
            writeLine(["error": "token id \(bad) out of range"]); continue
        }
        if ids.count >= engine.maxCtx {
            writeLine(["error": "prompt has \(ids.count) tokens, max_ctx is \(engine.maxCtx)"]); continue
        }
        let maxTokens = min(req["max_tokens"] as? Int ?? 256, engine.maxCtx - ids.count)
        let stopIds = Set((req["stop_ids"] as? [Int] ?? []).map(Int32.init))

        var produced = 0
        var finish = "length"
        func emit(_ t: Int32) -> Bool {
            if stopIds.contains(t) { finish = "stop"; return false }
            writeLine(["token": Int(t)])
            produced += 1
            return produced < maxTokens
        }

        engine.setTokens(ids)
        let t0 = Date()
        engine.prefill(n: ids.count, useGemm: useGemm)  // also samples the first token
        let tPrefill = Date().timeIntervalSince(t0)
        let t1 = Date()
        if maxTokens > 0 && emit(engine.token(at: ids.count)) && maxTokens > 1 {
            engine.decodeStreaming(from: ids.count, count: maxTokens - 1, emit: emit)
        }
        writeLine(["done": true, "finish_reason": finish, "prompt_tokens": ids.count,
                   "completion_tokens": produced, "prefill_s": tPrefill,
                   "decode_s": Date().timeIntervalSince(t1)])
    }
}

// MARK: - main

func stats(_ xs: [Double]) -> (Double, Double) {
    let m = xs.reduce(0, +) / Double(xs.count)
    let v = xs.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(max(xs.count - 1, 1))
    return (m, v.squareRoot())
}

let opts = parseOptions()
guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
let model = ModelFile(path: opts.weights, device: device)
let maxBatch = opts.serve ? 512 : max(64, min(opts.benchPP, 512), min(opts.promptIds.count, 512))
let engine = Engine(model: model, kernelsPath: opts.kernels, defines: opts.defines,
                    maxCtx: opts.maxCtx, maxBatch: maxBatch)
let cfg = model.config
FileHandle.standardError.write("device: \(device.name), dim \(cfg.dim), layers \(cfg.nLayers), vocab \(cfg.vocab), gemm \(opts.useGemm)\n".data(using: .utf8)!)

if !opts.promptIds.isEmpty {
    let ids = opts.promptIds
    precondition(ids.count + opts.gen <= opts.maxCtx)
    engine.setTokens(ids)
    let t0 = Date()
    engine.prefill(n: ids.count, useGemm: opts.useGemm)
    let tPrefill = Date().timeIntervalSince(t0)
    if let path = opts.dumpLogits {
        let raw = engine.logits.contents().bindMemory(to: Float.self, capacity: cfg.vocab)
        let c = cfg.softcap
        let capped = (0..<cfg.vocab).map { c > 0 ? c * tanh(raw[$0] / c) : raw[$0] }
        try! capped.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: path))
    }
    let t1 = Date()
    if opts.gen > 1 { engine.decode(from: ids.count, count: opts.gen - 1) }
    let tDecode = Date().timeIntervalSince(t1)
    let out = (0..<opts.gen).map { engine.token(at: ids.count + $0) }
    let result: [String: Any] = [
        "tokens": out,
        "prefill_s": tPrefill, "decode_s": tDecode,
        "decode_tok_s": opts.gen > 1 ? Double(opts.gen - 1) / tDecode : 0,
    ]
    print(String(data: try! JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
}

if opts.benchPP > 0 || opts.benchTG > 0 {
    var report: [String: Any] = ["tunables": engine.tunables]
    if opts.benchPP > 0 {
        // Same as llama-bench pp: random-ish prompt tokens, empty cache each rep.
        engine.setTokens((0..<opts.benchPP).map { Int32(($0 * 7919) % 100000 + 1000) })
        engine.prefill(n: opts.benchPP, useGemm: opts.useGemm)  // warmup
        var rates: [Double] = []
        for _ in 0..<opts.reps {
            let t0 = Date()
            engine.prefill(n: opts.benchPP, useGemm: opts.useGemm)
            rates.append(Double(opts.benchPP) / Date().timeIntervalSince(t0))
        }
        let (m, s) = stats(rates)
        report["pp"] = ["n": opts.benchPP, "tok_s": m, "std": s]
        FileHandle.standardError.write(String(format: "pp%d: %.1f ± %.1f tok/s\n", opts.benchPP, m, s).data(using: .utf8)!)
    }
    if opts.benchTG > 0 {
        engine.setTokens([cfg.bos])
        engine.decode(from: 0, count: min(16, opts.benchTG))  // warmup
        var rates: [Double] = []
        for _ in 0..<opts.reps {
            engine.setTokens([cfg.bos])
            let t0 = Date()
            engine.decode(from: 0, count: opts.benchTG)
            rates.append(Double(opts.benchTG) / Date().timeIntervalSince(t0))
        }
        let (m, s) = stats(rates)
        report["tg"] = ["n": opts.benchTG, "tok_s": m, "std": s]
        FileHandle.standardError.write(String(format: "tg%d: %.1f ± %.1f tok/s\n", opts.benchTG, m, s).data(using: .utf8)!)
    }
    print(String(data: try! JSONSerialization.data(withJSONObject: report), encoding: .utf8)!)
}

if opts.serve { serve(engine, useGemm: opts.useGemm) }
