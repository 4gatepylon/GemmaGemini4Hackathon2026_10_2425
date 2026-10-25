// Summary statistics for repeated timings.

struct Summary {
    let n: Int
    let mean, std, min, max, median, p10, p90: Double

    init(_ xs: [Double]) {
        precondition(!xs.isEmpty, "no samples")
        let s = xs.sorted()
        let count = s.count
        let m = s.reduce(0, +) / Double(count)
        // linear interpolation between closest ranks
        func q(_ p: Double) -> Double {
            let x = p * Double(count - 1)
            let lo = Int(x), hi = Swift.min(lo + 1, count - 1)
            return s[lo] + (s[hi] - s[lo]) * (x - Double(lo))
        }
        n = count
        mean = m
        std = (s.map { ($0 - m) * ($0 - m) }.reduce(0, +) / Double(Swift.max(count - 1, 1))).squareRoot()
        min = s[0]
        max = s[count - 1]
        median = q(0.5)
        p10 = q(0.1)
        p90 = q(0.9)
    }
}
