import AppKit
import Darwin
import Foundation

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var child: Process?
    private var localURL: URL?
    private var outputBuffer = Data()

    func applicationDidFinishLaunching(_ notification: Notification) {
        do {
            try startChild()
        } catch {
            showGenericFailure()
        }
    }

    func applicationShouldHandleReopen(
        _ sender: NSApplication,
        hasVisibleWindows flag: Bool
    ) -> Bool {
        if let url = localURL {
            NSWorkspace.shared.open(url)
        }
        return true
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        stopOwnedChild()
        return .terminateNow
    }

    private func startChild() throws {
        guard let resources = Bundle.main.resourceURL else {
            throw NSError(domain: "FacialAnalysis", code: 1)
        }
        let process = Process()
        process.executableURL = resources.appendingPathComponent("python/bin/python3")
        process.arguments = ["-I", "-B", resources.appendingPathComponent("bootstrap.py").path]
        process.currentDirectoryURL = resources
        process.environment = [
            "HOME": NSHomeDirectory(),
            "TMPDIR": NSTemporaryDirectory(),
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        ]

        let stdout = Pipe()
        let stderr = Pipe()
        process.standardOutput = stdout
        process.standardError = stderr
        stderr.fileHandleForReading.readabilityHandler = { handle in
            _ = handle.availableData
        }
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consume(handle.availableData)
        }
        process.terminationHandler = { [weak self] finished in
            stdout.fileHandleForReading.readabilityHandler = nil
            stderr.fileHandleForReading.readabilityHandler = nil
            guard finished.terminationStatus != 0 else {
                DispatchQueue.main.async { NSApplication.shared.terminate(nil) }
                return
            }
            DispatchQueue.main.async { self?.showGenericFailure() }
        }
        try process.run()
        child = process
    }

    private func consume(_ data: Data) {
        guard !data.isEmpty else { return }
        outputBuffer.append(data)
        while let newline = outputBuffer.firstIndex(of: 0x0A) {
            let lineData = outputBuffer.prefix(upTo: newline)
            outputBuffer.removeSubrange(...newline)
            guard let line = String(data: lineData, encoding: .utf8),
                  line.hasPrefix("LOCAL_URL=") else { continue }
            let value = String(line.dropFirst("LOCAL_URL=".count))
            guard let candidate = URL(string: value),
                  candidate.scheme == "http",
                  candidate.host == "127.0.0.1",
                  candidate.port != nil else { continue }
            DispatchQueue.main.async { [weak self] in
                self?.localURL = candidate
                NSWorkspace.shared.open(candidate)
            }
        }
    }

    private func stopOwnedChild() {
        guard let process = child, process.isRunning else { return }
        process.interrupt()
        if waitForExit(process, timeout: 2.0) { return }
        process.terminate()
        if waitForExit(process, timeout: 2.0) { return }
        kill(process.processIdentifier, SIGKILL)
        process.waitUntilExit()
    }

    private func waitForExit(_ process: Process, timeout: TimeInterval) -> Bool {
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            RunLoop.current.run(until: Date().addingTimeInterval(0.05))
        }
        return !process.isRunning
    }

    private func showGenericFailure() {
        let alert = NSAlert()
        alert.messageText = "Facial Analysis could not start safely."
        alert.informativeText = "Quit the app and try again."
        alert.alertStyle = .critical
        alert.runModal()
        NSApplication.shared.terminate(nil)
    }
}

let application = NSApplication.shared
let delegate = AppDelegate()
application.delegate = delegate
application.setActivationPolicy(.regular)
application.activate(ignoringOtherApps: true)
application.run()
