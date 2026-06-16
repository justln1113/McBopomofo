// Copyright (c) 2025 and onwards The McBopomofo Authors.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

import SwiftUI

// A tokenizer-playground-style inspector for the engine's scoring pipeline.
// The user types a Bopomofo key stream in English (ASCII) mode; the view shows
// the parsed readings, the plain Viterbi walk, the neural-rescored walk (tinted
// by per-block probability), the n-best paths, and a per-node candidate detail.
struct RescorerDebugView: View {
    @State private var asciiText: String = ""
    @State private var result: EISResult?
    @State private var selectedNode: EISNode?
    @State private var debounceTask: Task<Void, Never>?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                inputSection
                if let result {
                    readingsSection(result)
                    statusSection(result)
                    Divider()
                    walkSection(result)
                    if let node = selectedNode {
                        candidateSection(node)
                    }
                    Divider()
                    nBestSection(result)
                } else {
                    Text(NSLocalizedString(
                        "Type a Bopomofo key sequence above to inspect the engine.",
                        comment: ""))
                        .foregroundStyle(.secondary)
                        .padding(.top, 8)
                }
                Spacer(minLength: 0)
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // MARK: - Input

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(NSLocalizedString("Bopomofo Key Sequence", comment: ""))
                .font(.headline)
            BopomofoInputField(
                asciiText: $asciiText,
                displayText: EngineInspector.readingDisplay(forKeys: asciiText),
                placeholder: NSLocalizedString(
                    "Type Bopomofo keys in English (ABC) mode, e.g. su3 cl3 (= 你好)",
                    comment: ""))
                .frame(maxWidth: .infinity)
                .frame(height: 38)
                .onChange(of: asciiText) { _, newValue in
                    scheduleInspect(newValue)
                }
            if !asciiText.isEmpty {
                Text(String(format: NSLocalizedString("keys: %@", comment: ""), asciiText))
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            Text(NSLocalizedString(
                "Type the same keys as when typing normally; the field shows the converted readings live. Use Space for a first-tone syllable.",
                comment: ""))
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(NSLocalizedString(
                "Shortcuts: ⌫ deletes a key · ⌥⌫ a syllable · esc clears · ⌘A then ⌫ clears · ⌘C/⌘X/⌘V copy/cut/paste keys",
                comment: ""))
                .font(.caption2)
                .foregroundStyle(.tertiary)
            if let result, result.hasNonAsciiInput {
                Label(
                    NSLocalizedString(
                        "Non-ASCII characters detected — switch to English (ABC) input.",
                        comment: ""),
                    systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: - Parsed readings

    @ViewBuilder
    private func readingsSection(_ result: EISResult) -> some View {
        let invalid = Set(result.invalidReadingIndexes.map { $0.intValue })
        VStack(alignment: .leading, spacing: 6) {
            Text(NSLocalizedString("Parsed Readings", comment: ""))
                .font(.headline)
            if result.readings.isEmpty && result.incompleteBuffer == nil {
                Text(NSLocalizedString("(none)", comment: ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                FlowLayout(hSpacing: 6, vSpacing: 6) {
                    ForEach(Array(result.readings.enumerated()), id: \.offset) { index, reading in
                        Text(reading)
                            .font(.system(.body, design: .monospaced))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 5)
                                    .fill(invalid.contains(index)
                                          ? Color.red.opacity(0.18)
                                          : Color.secondary.opacity(0.12)))
                            .overlay(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(invalid.contains(index) ? Color.red : .clear,
                                            lineWidth: 1))
                    }
                    if let incomplete = result.incompleteBuffer, !incomplete.isEmpty {
                        Text(incomplete)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(
                                RoundedRectangle(cornerRadius: 5)
                                    .stroke(style: StrokeStyle(lineWidth: 1, dash: [3]))
                                    .foregroundStyle(.secondary))
                    }
                }
            }
        }
    }

    // MARK: - Status

    private func statusSection(_ result: EISResult) -> some View {
        let rescorerState: String
        if !result.rescorerEnabled {
            rescorerState = NSLocalizedString("Rescorer: off", comment: "")
        } else if !result.rescorerLoaded {
            rescorerState = NSLocalizedString("Rescorer: not loaded", comment: "")
        } else {
            rescorerState = String(
                format: NSLocalizedString("Rescorer: on (λ=%.2f)", comment: ""),
                result.lambda)
        }
        return HStack(spacing: 14) {
            badge(rescorerState,
                  color: result.rescorerEnabled && result.rescorerLoaded ? .green : .secondary)
            badge(String(format: NSLocalizedString("walk %llu µs", comment: ""),
                         result.walkMicroseconds), color: .secondary)
            if result.rescorerLoaded {
                badge(String(format: NSLocalizedString("rescore %llu µs", comment: ""),
                             result.rescoreMicroseconds), color: .secondary)
            }
        }
        .font(.caption)
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(RoundedRectangle(cornerRadius: 5).fill(color.opacity(0.15)))
            .foregroundStyle(color == .secondary ? Color.secondary : color)
    }

    // MARK: - Walk segmentation

    @ViewBuilder
    private func walkSection(_ result: EISResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                Text(NSLocalizedString("Original Walk", comment: ""))
                    .font(.headline)
                tokenRow(result.plainWalk, changedAgainst: nil, tinted: false)
            }

            if let rescored = result.rescoredWalk {
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(NSLocalizedString("After Rescoring", comment: ""))
                            .font(.headline)
                        if result.rescorerPickIndex == 0 {
                            Text(NSLocalizedString("(unchanged)", comment: ""))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    tokenRow(rescored, changedAgainst: result.plainWalk, tinted: true)
                    probabilityLegend
                }
            } else if result.rescorerEnabled && !result.rescorerLoaded {
                Text(NSLocalizedString(
                    "Rescorer model is not bundled; only the original walk is shown.",
                    comment: ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !result.rescorerEnabled {
                Text(NSLocalizedString(
                    "Neural rescorer is disabled (toggle it in the input menu).",
                    comment: ""))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func tokenRow(_ nodes: [EISNode], changedAgainst plain: [EISNode]?,
                          tinted: Bool) -> some View {
        let changedStarts = changedStartSet(rescored: nodes, plain: plain)
        var offset = 0
        var entries: [(Int, EISNode, Bool)] = []
        for node in nodes {
            let isChanged = plain != nil && changedStarts.contains(StartKey(start: offset, value: node.value))
            entries.append((offset, node, isChanged))
            offset += node.value.count
        }
        return FlowLayout(hSpacing: 8, vSpacing: 8) {
            ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                let (_, node, isChanged) = entry
                Button {
                    selectedNode = (node === selectedNode) ? nil : node
                } label: {
                    TokenBox(
                        node: node,
                        tinted: tinted,
                        changed: isChanged,
                        selected: node === selectedNode)
                }
                .buttonStyle(.plain)
                .accessibilityLabel(Text("\(node.value) \(node.reading)"))
            }
        }
    }

    private var probabilityLegend: some View {
        HStack(spacing: 6) {
            Text(NSLocalizedString("less likely", comment: ""))
                .font(.caption2).foregroundStyle(.secondary)
            LinearGradient(
                colors: [colorForLogProb(-12), colorForLogProb(-7), colorForLogProb(-2)],
                startPoint: .leading, endPoint: .trailing)
                .frame(width: 120, height: 8)
                .clipShape(Capsule())
            Text(NSLocalizedString("more likely", comment: ""))
                .font(.caption2).foregroundStyle(.secondary)
        }
    }

    // MARK: - Candidate detail

    @ViewBuilder
    private func candidateSection(_ node: EISNode) -> some View {
        let probs = softmax(node.candidates.map { $0.score })
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(String(format: NSLocalizedString("Candidates for “%@” (%@)", comment: ""),
                            node.reading, node.value))
                    .font(.headline)
                Spacer()
                Button {
                    selectedNode = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
            if node.uomOverride, let uomValue = node.uomValue {
                Text(String(
                    format: node.uomForced
                        ? NSLocalizedString("UserOverrideModel forces “%@” here (high-score).", comment: "")
                        : NSLocalizedString("UserOverrideModel suggests “%@” here (decaying).", comment: ""),
                    uomValue))
                    .font(.caption)
                    .foregroundStyle(.purple)
            }
            VStack(spacing: 0) {
                candidateHeaderRow
                ForEach(Array(node.candidates.enumerated()), id: \.offset) { index, candidate in
                    candidateRow(candidate, prob: index < probs.count ? probs[index] : 0)
                        .background(candidate.chosen ? Color.accentColor.opacity(0.12) : .clear)
                }
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.06)))
        }
    }

    private var candidateHeaderRow: some View {
        HStack(spacing: 8) {
            Text(NSLocalizedString("Value", comment: "")).frame(width: 80, alignment: .leading)
            Text(NSLocalizedString("logP", comment: "")).frame(width: 80, alignment: .trailing)
            Text(NSLocalizedString("Prob.", comment: "")).frame(width: 70, alignment: .trailing)
            Text(NSLocalizedString("Source", comment: "")).frame(maxWidth: .infinity, alignment: .leading)
        }
        .font(.caption.bold())
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
    }

    private func candidateRow(_ candidate: EISCandidate, prob: Double) -> some View {
        HStack(spacing: 8) {
            HStack(spacing: 4) {
                if candidate.chosen {
                    Image(systemName: "checkmark").font(.caption2).foregroundStyle(Color.accentColor)
                }
                Text(candidate.value).font(.system(.body, design: .monospaced))
            }
            .frame(width: 80, alignment: .leading)
            Text(String(format: "%.3f", candidate.score))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 80, alignment: .trailing)
            Text(String(format: "%.1f%%", prob * 100))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 70, alignment: .trailing)
            sourceBadge(candidate)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private func sourceBadge(_ candidate: EISCandidate) -> some View {
        HStack(spacing: 4) {
            if candidate.fromUserPhrase {
                tag(NSLocalizedString("User Phrase", comment: ""), .blue)
            }
            if candidate.uomSuggested {
                tag(candidate.uomForced
                    ? NSLocalizedString("UOM (forced)", comment: "")
                    : NSLocalizedString("UOM (decaying)", comment: ""), .purple)
            }
            if !candidate.fromUserPhrase && !candidate.uomSuggested {
                Text(NSLocalizedString("Dictionary", comment: ""))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private func tag(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(RoundedRectangle(cornerRadius: 4).fill(color.opacity(0.18)))
            .foregroundStyle(color)
    }

    // MARK: - N-best

    @ViewBuilder
    private func nBestSection(_ result: EISResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(NSLocalizedString("N-best Paths", comment: ""))
                .font(.headline)
            VStack(spacing: 0) {
                nBestHeaderRow(showModel: result.rescorerLoaded)
                ForEach(Array(result.nBest.enumerated()), id: \.offset) { index, path in
                    nBestRow(index: index, path: path, showModel: result.rescorerLoaded)
                        .background(path.rescorerPick
                                    ? Color.green.opacity(0.12)
                                    : (index % 2 == 0 ? Color.secondary.opacity(0.05) : .clear))
                }
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.04)))
        }
    }

    private func nBestHeaderRow(showModel: Bool) -> some View {
        HStack(spacing: 8) {
            Text("#").frame(width: 62, alignment: .leading)
            Text(NSLocalizedString("Sentence", comment: ""))
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(NSLocalizedString("unigram", comment: "")).frame(width: 70, alignment: .trailing)
            if showModel {
                Text(NSLocalizedString("model", comment: "")).frame(width: 70, alignment: .trailing)
                Text(NSLocalizedString("combined", comment: "")).frame(width: 70, alignment: .trailing)
            }
        }
        .font(.caption.bold())
        .foregroundStyle(.secondary)
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
    }

    private func nBestRow(index: Int, path: EISPath, showModel: Bool) -> some View {
        HStack(spacing: 8) {
            HStack(spacing: 3) {
                Text("\(index + 1)")
                if path.plainWalkTop {
                    Image(systemName: "arrow.right.circle").font(.caption2).foregroundStyle(.secondary)
                }
                if path.rescorerPick {
                    Image(systemName: "star.fill").font(.caption2).foregroundStyle(.green)
                }
            }
            .frame(width: 62, alignment: .leading)
            Text(path.values.joined())
                .font(.system(.body, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(String(format: "%.2f", path.unigramScore))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 70, alignment: .trailing)
            if showModel {
                Text(path.hasModelScore ? String(format: "%.2f", path.modelScore) : "—")
                    .font(.system(.caption, design: .monospaced))
                    .frame(width: 70, alignment: .trailing)
                Text(path.hasModelScore ? String(format: "%.2f", path.combinedScore) : "—")
                    .font(.system(.caption, design: .monospaced).bold())
                    .frame(width: 70, alignment: .trailing)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
    }

    // MARK: - Logic

    private func scheduleInspect(_ text: String) {
        debounceTask?.cancel()
        selectedNode = nil
        let trimmed = text
        debounceTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 250_000_000)
            if Task.isCancelled { return }
            if trimmed.isEmpty {
                result = nil
            } else {
                result = EngineInspector.inspect(trimmed, nBest: 5)
            }
        }
    }

    private struct StartKey: Hashable {
        let start: Int
        let value: String
    }

    private func changedStartSet(rescored: [EISNode], plain: [EISNode]?) -> Set<StartKey> {
        guard let plain else { return [] }
        // A rescored block is "changed" if no plain block starts at the same
        // character offset with the same value.
        var plainKeys = Set<StartKey>()
        var offset = 0
        for node in plain {
            plainKeys.insert(StartKey(start: offset, value: node.value))
            offset += node.value.count
        }
        var changed = Set<StartKey>()
        offset = 0
        for node in rescored {
            let key = StartKey(start: offset, value: node.value)
            if !plainKeys.contains(key) {
                changed.insert(key)
            }
            offset += node.value.count
        }
        return changed
    }
}

// MARK: - Token box

private struct TokenBox: View {
    let node: EISNode
    let tinted: Bool
    let changed: Bool
    let selected: Bool

    var body: some View {
        VStack(spacing: 2) {
            Text(node.value)
                .font(.system(.title3, design: .monospaced))
            Text(node.reading)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(scoreText)
                .font(.system(size: 9, design: .monospaced))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(backgroundColor))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(borderColor, lineWidth: borderWidth))
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 2) {
                if node.fromUserPhrase {
                    cornerBadge("詞", .blue)
                }
                if node.uomOverride {
                    cornerBadge(node.uomForced ? "UOM↑" : "UOM", .purple)
                }
            }
            .offset(x: 5, y: -6)
        }
        .contentShape(RoundedRectangle(cornerRadius: 8))
    }

    private func cornerBadge(_ text: String, _ color: Color) -> some View {
        Text(text)
            .font(.system(size: 7, weight: .bold))
            .padding(.horizontal, 3).padding(.vertical, 1)
            .background(Capsule().fill(color))
            .foregroundStyle(.white)
    }

    private var scoreText: String {
        if tinted && node.hasNeuralLogProb {
            return String(format: "%.2f", node.neuralLogProb)
        }
        return String(format: "%.2f", node.score)
    }

    private var backgroundColor: Color {
        if tinted && node.hasNeuralLogProb {
            return colorForLogProb(node.neuralLogProb).opacity(0.30)
        }
        return Color.secondary.opacity(0.12)
    }

    private var borderColor: Color {
        if selected { return .accentColor }
        if changed { return .orange }
        return .clear
    }

    private var borderWidth: CGFloat {
        if selected { return 2 }
        if changed { return 1.5 }
        return 0
    }
}

// MARK: - Helpers

// Maps a (negative) log probability to a red→yellow→green color. Clamped to a
// readable range; near 0 is "likely" (green), very negative is "unlikely" (red).
private func colorForLogProb(_ logProb: Double) -> Color {
    let lo = -12.0
    let hi = -2.0
    let t = max(0.0, min(1.0, (logProb - lo) / (hi - lo)))
    // Hue 0 (red) -> 0.33 (green).
    return Color(hue: 0.33 * t, saturation: 0.75, brightness: 0.85)
}

private func softmax(_ scores: [Double]) -> [Double] {
    guard let maxScore = scores.max() else { return [] }
    let exps = scores.map { exp($0 - maxScore) }
    let sum = exps.reduce(0, +)
    guard sum > 0 else { return scores.map { _ in 0 } }
    return exps.map { $0 / sum }
}

// MARK: - Flow layout

struct FlowLayout: Layout {
    var hSpacing: CGFloat = 8
    var vSpacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews,
                      cache: inout Void) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0
        var widest: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                widest = max(widest, x - hSpacing)
                x = 0
                y += rowHeight + vSpacing
                rowHeight = 0
            }
            x += size.width + hSpacing
            rowHeight = max(rowHeight, size.height)
        }
        widest = max(widest, x - hSpacing)
        let totalHeight = y + rowHeight
        return CGSize(
            width: maxWidth.isFinite ? maxWidth : max(0, widest),
            height: totalHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                       subviews: Subviews, cache: inout Void) {
        let maxWidth = bounds.width
        var x: CGFloat = bounds.minX
        var y: CGFloat = bounds.minY
        var rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x - bounds.minX + size.width > maxWidth && x > bounds.minX {
                x = bounds.minX
                y += rowHeight + vSpacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), anchor: .topLeading,
                          proposal: ProposedViewSize(size))
            x += size.width + hSpacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

// MARK: - Bopomofo input field

// An input control the user types into with the English (ASCII) keyboard, but
// which displays the converted Bopomofo readings live. The raw key sequence is
// the source of truth (bound to `asciiText`); `displayText` is the converted
// Bopomofo computed by the caller. Supports append, Backspace (one key),
// Option-Backspace (one syllable), Esc / Cmd-A+delete (clear), and Cmd-C/X/V.
// Editing in the middle is intentionally not supported — this is a debug entry
// field, not a full text editor.
struct BopomofoInputField: NSViewRepresentable {
    @Binding var asciiText: String
    var displayText: String
    var placeholder: String

    func makeCoordinator() -> Coordinator { Coordinator(asciiText: $asciiText) }

    func makeNSView(context: Context) -> KeyCaptureView {
        let view = KeyCaptureView()
        let coordinator = context.coordinator
        view.onInsert = { text in coordinator.asciiText.wrappedValue += text }
        view.onBackspace = {
            if !coordinator.asciiText.wrappedValue.isEmpty {
                coordinator.asciiText.wrappedValue.removeLast()
            }
        }
        view.onDeleteSyllable = {
            coordinator.asciiText.wrappedValue =
                EngineInspector.keys(byDeletingLastSyllable: coordinator.asciiText.wrappedValue)
        }
        view.onClear = { coordinator.asciiText.wrappedValue = "" }
        return view
    }

    func updateNSView(_ view: KeyCaptureView, context: Context) {
        context.coordinator.asciiText = $asciiText
        view.placeholderString = placeholder
        view.asciiKeys = asciiText
        view.displayString = displayText
    }

    final class Coordinator {
        var asciiText: Binding<String>
        init(asciiText: Binding<String>) { self.asciiText = asciiText }
    }
}

// AppKit view that captures raw keystrokes and renders a supplied display string.
final class KeyCaptureView: NSView {
    var onInsert: ((String) -> Void)?
    var onBackspace: (() -> Void)?
    var onDeleteSyllable: (() -> Void)?
    var onClear: (() -> Void)?

    // The raw ASCII keys (for Cmd-C / Cmd-X). Set by the representable.
    var asciiKeys: String = ""

    private let label = NSTextField(labelWithString: "")
    private var focused = false
    private var selectedAll = false

    var displayString: String = "" { didSet { refresh() } }
    var placeholderString: String = "" { didSet { refresh() } }

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.cornerRadius = 6
        layer?.borderWidth = 1
        layer?.borderColor = NSColor.separatorColor.cgColor
        layer?.backgroundColor = NSColor.textBackgroundColor.cgColor
        label.font = .monospacedSystemFont(ofSize: 18, weight: .regular)
        label.lineBreakMode = .byTruncatingHead
        label.isEditable = false
        label.isSelectable = false
        label.isBordered = false
        label.translatesAutoresizingMaskIntoConstraints = false
        addSubview(label)
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 10),
            label.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -10),
            label.centerYAnchor.constraint(equalTo: centerYAnchor),
        ])
        refresh()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }

    private func refresh() {
        if displayString.isEmpty {
            label.drawsBackground = false
            label.stringValue = focused ? "▏" : placeholderString
            label.textColor = focused ? .labelColor : .placeholderTextColor
        } else if selectedAll {
            label.drawsBackground = true
            label.backgroundColor = .selectedTextBackgroundColor
            label.stringValue = displayString
            label.textColor = .selectedTextColor
        } else {
            label.drawsBackground = false
            label.stringValue = displayString + (focused ? "\u{2009}▏" : "")
            label.textColor = .labelColor
        }
    }

    override var acceptsFirstResponder: Bool { true }

    override func becomeFirstResponder() -> Bool {
        focused = true
        layer?.borderColor = NSColor.controlAccentColor.cgColor
        layer?.borderWidth = 2
        refresh()
        return true
    }

    override func resignFirstResponder() -> Bool {
        focused = false
        selectedAll = false
        layer?.borderColor = NSColor.separatorColor.cgColor
        layer?.borderWidth = 1
        refresh()
        return true
    }

    override func mouseDown(with event: NSEvent) {
        selectedAll = false
        refresh()
        window?.makeFirstResponder(self)
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if window != nil {
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.window?.makeFirstResponder(self)
            }
        }
    }

    override func keyDown(with event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)

        if flags.contains(.command) {
            switch event.charactersIgnoringModifiers {
            case "a":
                if !displayString.isEmpty {
                    selectedAll = true
                    refresh()
                }
            case "c":
                if !asciiKeys.isEmpty {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(asciiKeys, forType: .string)
                }
            case "x":
                if !asciiKeys.isEmpty {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(asciiKeys, forType: .string)
                }
                clearSelectionAndCallClear()
            case "v":
                if let pasted = NSPasteboard.general.string(forType: .string) {
                    if selectedAll { onClear?() }
                    selectedAll = false
                    onInsert?(pasted)
                }
            default:
                super.keyDown(with: event)
            }
            return
        }

        // Option-Backspace: delete the last whole syllable.
        if flags.contains(.option) && event.keyCode == 51 {
            selectedAll = false
            onDeleteSyllable?()
            return
        }

        switch event.keyCode {
        case 51, 117:  // backspace / forward delete
            if selectedAll {
                clearSelectionAndCallClear()
            } else {
                onBackspace?()
            }
            return
        case 53:  // esc
            clearSelectionAndCallClear()
            return
        default:
            break
        }

        guard let chars = event.charactersIgnoringModifiers else {
            super.keyDown(with: event)
            return
        }
        var accepted = ""
        for scalar in chars.unicodeScalars where scalar.value >= 0x20 && scalar.value < 0x7f {
            accepted.unicodeScalars.append(scalar)
        }
        if accepted.isEmpty {
            super.keyDown(with: event)
            return
        }
        if selectedAll { onClear?() }
        selectedAll = false
        onInsert?(accepted)
    }

    private func clearSelectionAndCallClear() {
        selectedAll = false
        onClear?()
    }
}
