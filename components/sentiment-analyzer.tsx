"use client"

import { useState } from "react"
import { Sparkles, Loader2, CornerDownLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { SentimentBadge } from "@/components/sentiment-badge"
import { cn } from "@/lib/utils"

type AnalysisResult = {
  sentiment: "Positive" | "Negative" | "Neutral"
  score: number
  confidence: number
  summary: string
  themes: string[]
}

const SAMPLES = [
  "The driver was super friendly and the ride was smooth and on time.",
  "Charged way more than the quoted fare and support never replied.",
  "App keeps crashing when I try to book. Please fix the GPS bug.",
]

export function SentimentAnalyzer() {
  const [text, setText] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<AnalysisResult | null>(null)

  async function analyze() {
    if (!text.trim() || loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || "Something went wrong")
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to analyze")
    } finally {
      setLoading(false)
    }
  }

  const scorePct = result ? Math.round(((result.score + 1) / 2) * 100) : 0

  return (
    <div className="rounded-xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-5 py-4">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-primary/30 bg-primary/10 text-primary">
          <Sparkles className="h-4 w-4" />
        </span>
        <div>
          <h2 className="text-sm font-semibold leading-none">Live Sentiment Analyzer</h2>
          <p className="mt-1 text-xs text-muted-foreground">Paste any review to score it instantly with AI</p>
        </div>
      </div>

      <div className="grid gap-5 p-5 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <div className="relative">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze()
              }}
              placeholder="e.g. The app charged me twice and the driver canceled last minute..."
              rows={5}
              className="w-full resize-none rounded-lg border border-input bg-secondary px-3 py-2.5 text-sm leading-relaxed text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={analyze} disabled={loading || !text.trim()}>
              {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
              {loading ? "Analyzing..." : "Analyze Sentiment"}
            </Button>
            <span className="hidden items-center gap-1 text-xs text-muted-foreground sm:flex">
              <CornerDownLeft className="h-3 w-3" /> ⌘ + Enter
            </span>
          </div>
          <div className="flex flex-wrap gap-2">
            {SAMPLES.map((s, i) => (
              <button
                key={i}
                onClick={() => setText(s)}
                className="rounded-full border border-border bg-secondary px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:text-foreground"
              >
                {`Sample ${i + 1}`}
              </button>
            ))}
          </div>
        </div>

        <div className="flex min-h-[180px] flex-col rounded-lg border border-border bg-secondary/40 p-4">
          {!result && !error && !loading && (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <Sparkles className="h-6 w-6 text-muted-foreground/50" />
              <p className="mt-2 text-sm text-muted-foreground">Your analysis will appear here</p>
            </div>
          )}
          {loading && (
            <div className="flex flex-1 flex-col items-center justify-center text-center">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
              <p className="mt-2 text-sm text-muted-foreground">Scoring your text...</p>
            </div>
          )}
          {error && (
            <div className="flex flex-1 items-center justify-center text-center text-sm text-negative">{error}</div>
          )}
          {result && (
            <div className="flex flex-1 flex-col gap-4">
              <div className="flex items-center justify-between">
                <SentimentBadge sentiment={result.sentiment} />
                <span className="font-mono text-sm tabular-nums text-muted-foreground">
                  score {result.score.toFixed(2)}
                </span>
              </div>

              <div>
                <div className="mb-1.5 flex items-center justify-between text-xs text-muted-foreground">
                  <span>Negative</span>
                  <span>Positive</span>
                </div>
                <div className="relative h-2 w-full overflow-hidden rounded-full bg-secondary">
                  <div
                    className={cn(
                      "absolute inset-y-0 left-0 rounded-full transition-all",
                      result.score >= 0 ? "bg-positive" : "bg-negative",
                    )}
                    style={{ width: `${scorePct}%` }}
                  />
                </div>
                <p className="mt-1.5 text-right text-xs text-muted-foreground">
                  {Math.round(result.confidence * 100)}% confidence
                </p>
              </div>

              <p className="text-sm leading-relaxed text-foreground">{result.summary}</p>

              {result.themes?.length > 0 && (
                <div className="mt-auto flex flex-wrap gap-1.5">
                  {result.themes.map((t, i) => (
                    <span key={i} className="rounded-md border border-border bg-card px-2 py-0.5 text-xs text-muted-foreground">
                      {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
