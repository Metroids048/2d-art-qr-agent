import Foundation
import Vision

var results: [String: [String]] = [:]
for path in CommandLine.arguments.dropFirst(){
    let request = VNDetectBarcodesRequest()
    request.symbologies = [.qr]
    let handler = VNImageRequestHandler(url: URL(fileURLWithPath: path))
    do {
        try handler.perform([request])
        results[path] = request.results?.compactMap(\.payloadStringValue) ?? []
    } catch {
        results[path] = ["ERROR: \(error)"]
    }
}
let output = try JSONEncoder().encode(results)
print(String(decoding:output,as:UTF8.self))
