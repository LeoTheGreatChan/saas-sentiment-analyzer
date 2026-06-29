// Lightweight lexicon-based sentiment scorer used as a resilient fallback
// when the AI Gateway is unavailable. Inspired by VADER-style scoring.

const POSITIVE: Record<string, number> = {
  good: 1, great: 2, excellent: 2.5, amazing: 2.5, awesome: 2.5, love: 2.5, loved: 2.5,
  nice: 1.2, best: 2, perfect: 2.5, helpful: 1.5, friendly: 1.5, fast: 1.2, easy: 1.2,
  convenient: 1.5, smooth: 1.3, reliable: 1.5, clean: 1, polite: 1.5, wonderful: 2.3,
  fantastic: 2.5, superb: 2.3, recommend: 1.5, satisfied: 1.5, happy: 1.8, comfortable: 1.3,
  affordable: 1.3, cheap: 0.8, quick: 1.2, professional: 1.5, safe: 1.3, efficient: 1.4,
}

const NEGATIVE: Record<string, number> = {
  bad: -1.5, worst: -2.5, terrible: -2.5, horrible: -2.5, awful: -2.3, hate: -2.5, hated: -2.5,
  poor: -1.5, slow: -1.2, expensive: -1.2, overpriced: -1.8, scam: -2.5, fraud: -2.5,
  rude: -2, unprofessional: -2, glitch: -1.5, glitching: -1.5, bug: -1.3, buggy: -1.5,
  crash: -1.8, crashes: -1.8, crashing: -1.8, broken: -1.8, useless: -2.2, disappointed: -1.8,
  disappointing: -1.8, annoying: -1.5, cancel: -1.2, canceled: -1.3, cancelled: -1.3,
  charged: -0.8, overcharged: -2, refund: -1, waiting: -1, delay: -1.3, delayed: -1.3,
  unreliable: -1.8, dirty: -1.3, dangerous: -2, error: -1.3, problem: -1.2, issue: -1,
  worse: -1.8, pathetic: -2.2, frustrating: -1.8, frustrated: -1.8, never: -0.8, worstapp: -2.5,
}

const NEGATORS = new Set(["not", "no", "never", "dont", "don't", "cant", "can't", "didnt", "didn't", "wont", "won't", "isnt", "isn't", "without"])
const INTENSIFIERS: Record<string, number> = { very: 1.4, really: 1.4, so: 1.3, extremely: 1.7, super: 1.5, totally: 1.4, absolutely: 1.6, too: 1.2 }

export type LocalSentiment = {
  sentiment: "Positive" | "Negative" | "Neutral"
  score: number
  confidence: number
  summary: string
  themes: string[]
}

const THEME_KEYWORDS: Record<string, string[]> = {
  Pricing: ["price", "expensive", "overpriced", "charged", "overcharged", "fare", "cost", "refund", "coupon", "cheap", "affordable"],
  "Driver Behavior": ["driver", "rude", "polite", "friendly", "professional", "cancel", "canceled", "cancelled"],
  "App Performance": ["app", "glitch", "glitching", "bug", "crash", "crashes", "error", "slow", "loading", "update"],
  "Support": ["support", "customer", "service", "help", "response", "contact", "respond"],
  "Location & GPS": ["gps", "location", "map", "pickup", "navigation", "address"],
  "Safety": ["safe", "dangerous", "unsafe", "accident", "security"],
}

export function analyzeLocally(text: string): LocalSentiment {
  const cleaned = text.toLowerCase().replace(/[^a-z0-9'\s]/g, " ")
  const tokens = cleaned.split(/\s+/).filter(Boolean)

  let raw = 0
  let hits = 0

  for (let i = 0; i < tokens.length; i++) {
    const word = tokens[i]
    let val = POSITIVE[word] ?? NEGATIVE[word] ?? 0
    if (val === 0) continue

    const prev = tokens[i - 1]
    const prev2 = tokens[i - 2]
    if (prev && INTENSIFIERS[prev]) val *= INTENSIFIERS[prev]
    if ((prev && NEGATORS.has(prev)) || (prev2 && NEGATORS.has(prev2))) val *= -0.85

    raw += val
    hits++
  }

  // Normalize using a VADER-like squashing function
  const score = hits === 0 ? 0 : Math.max(-1, Math.min(1, raw / Math.sqrt(raw * raw + 15)))

  const sentiment: LocalSentiment["sentiment"] =
    score > 0.05 ? "Positive" : score < -0.05 ? "Negative" : "Neutral"

  const confidence = hits === 0 ? 0.3 : Math.min(0.99, 0.45 + Math.min(hits, 6) * 0.09)

  const themes: string[] = []
  for (const [theme, kws] of Object.entries(THEME_KEYWORDS)) {
    if (kws.some((kw) => tokens.includes(kw))) themes.push(theme)
    if (themes.length >= 3) break
  }

  const summary =
    sentiment === "Positive"
      ? "Customer expresses satisfaction with the experience."
      : sentiment === "Negative"
        ? "Customer reports a frustrating or unsatisfactory experience."
        : "Customer feedback is mixed or neutral in tone."

  return { sentiment, score: Number(score.toFixed(4)), confidence: Number(confidence.toFixed(2)), summary, themes }
}
