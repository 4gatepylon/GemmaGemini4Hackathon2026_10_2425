// Tiny arithmetic expression evaluator for bench specs, so sizes and grids can
// be written as "M*K*2" or "ceildiv(M, NR*NSG)" instead of precomputed numbers.
//
// Grammar: numbers, variables, + - * / %, unary minus, parentheses, and the
// functions min, max, ceil, floor, round, ceildiv, pow, log2.

import Foundation

struct ExprError: Error, CustomStringConvertible {
    let expr: String
    let message: String
    var description: String { "in expression '\(expr)': \(message)" }
}

enum Expr {
    static func eval(_ s: String, _ vars: [String: Double]) throws -> Double {
        var p = ExprParser(src: s, c: Array(s.utf8), vars: vars)
        let v = try p.expr()
        if let ch = p.peek() { throw p.error("unexpected '\(Character(UnicodeScalar(ch)))'") }
        return v
    }

    /// A JSON value (number or expression string) as a Double.
    static func number(_ v: Any, _ vars: [String: Double]) throws -> Double {
        if let s = v as? String { return try eval(s, vars) }
        if let n = v as? NSNumber { return n.doubleValue }
        throw ExprError(expr: "\(v)", message: "expected a number or an expression string")
    }

    /// A JSON value as an exact integer (errors if the expression isn't integral).
    static func int(_ v: Any, _ vars: [String: Double]) throws -> Int {
        let d = try number(v, vars)
        guard d.isFinite, d == d.rounded() else {
            throw ExprError(expr: "\(v)", message: "= \(d), expected an integer (use ceildiv/floor/ceil)")
        }
        return Int(d)
    }
}

private struct ExprParser {
    let src: String
    let c: [UInt8]
    let vars: [String: Double]
    var i = 0

    func error(_ m: String) -> ExprError { ExprError(expr: src, message: m) }

    mutating func peek() -> UInt8? {
        while i < c.count, c[i] == UInt8(ascii: " ") { i += 1 }
        return i < c.count ? c[i] : nil
    }

    mutating func expect(_ ch: Character) throws {
        guard peek() == ch.asciiValue else { throw error("expected '\(ch)'") }
        i += 1
    }

    mutating func expr() throws -> Double {
        var v = try term()
        while let ch = peek(), ch == UInt8(ascii: "+") || ch == UInt8(ascii: "-") {
            i += 1
            let r = try term()
            v = ch == UInt8(ascii: "+") ? v + r : v - r
        }
        return v
    }

    mutating func term() throws -> Double {
        var v = try unary()
        while let ch = peek(), ch == UInt8(ascii: "*") || ch == UInt8(ascii: "/") || ch == UInt8(ascii: "%") {
            i += 1
            let r = try unary()
            switch ch {
            case UInt8(ascii: "*"): v *= r
            case UInt8(ascii: "/"): v /= r
            default: v = v.truncatingRemainder(dividingBy: r)
            }
        }
        return v
    }

    mutating func unary() throws -> Double {
        if peek() == UInt8(ascii: "-") { i += 1; return -(try unary()) }
        return try primary()
    }

    static func isDigit(_ ch: UInt8) -> Bool { ch >= 48 && ch <= 57 }
    static func isIdent(_ ch: UInt8) -> Bool {
        isDigit(ch) || ch == UInt8(ascii: "_") || (ch | 0x20 >= 97 && ch | 0x20 <= 122)
    }

    mutating func primary() throws -> Double {
        guard let ch = peek() else { throw error("unexpected end") }
        if ch == UInt8(ascii: "(") {
            i += 1
            let v = try expr()
            try expect(")")
            return v
        }
        if Self.isDigit(ch) || ch == UInt8(ascii: ".") {
            let start = i
            while i < c.count, Self.isDigit(c[i]) || c[i] == UInt8(ascii: ".") { i += 1 }
            if i < c.count, c[i] | 0x20 == UInt8(ascii: "e") {
                i += 1
                if i < c.count, c[i] == UInt8(ascii: "+") || c[i] == UInt8(ascii: "-") { i += 1 }
                while i < c.count, Self.isDigit(c[i]) { i += 1 }
            }
            let text = String(decoding: c[start..<i], as: UTF8.self)
            guard let v = Double(text) else { throw error("bad number '\(text)'") }
            return v
        }
        if Self.isIdent(ch) {
            let start = i
            while i < c.count, Self.isIdent(c[i]) { i += 1 }
            let name = String(decoding: c[start..<i], as: UTF8.self)
            if peek() == UInt8(ascii: "(") {
                i += 1
                var args: [Double] = []
                if peek() != UInt8(ascii: ")") {
                    args.append(try expr())
                    while peek() == UInt8(ascii: ",") { i += 1; args.append(try expr()) }
                }
                try expect(")")
                return try call(name, args)
            }
            guard let v = vars[name] else {
                throw error("unknown variable '\(name)' (known: \(vars.keys.sorted().joined(separator: ", ")))")
            }
            return v
        }
        throw error("unexpected '\(Character(UnicodeScalar(ch)))'")
    }

    func call(_ f: String, _ a: [Double]) throws -> Double {
        func need(_ n: Int) throws { if a.count != n { throw error("\(f) takes \(n) argument(s)") } }
        switch f {
        case "min": guard let m = a.min() else { throw error("min of nothing") }; return m
        case "max": guard let m = a.max() else { throw error("max of nothing") }; return m
        case "ceil": try need(1); return a[0].rounded(.up)
        case "floor": try need(1); return a[0].rounded(.down)
        case "round": try need(1); return a[0].rounded()
        case "ceildiv": try need(2); return (a[0] / a[1]).rounded(.up)
        case "pow": try need(2); return pow(a[0], a[1])
        case "log2": try need(1); return log2(a[0])
        default: throw error("unknown function '\(f)'")
        }
    }
}
