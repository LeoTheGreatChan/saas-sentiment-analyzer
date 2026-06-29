import type { Review } from "@/lib/reviews"
import { SentimentBadge } from "@/components/sentiment-badge"
import { cn } from "@/lib/utils"

function ScoreCell({ score }: { score: number }) {
  return (
    <span className={cn("font-mono text-xs tabular-nums", score >= 0 ? "text-positive" : "text-negative")}>
      {score >= 0 ? "+" : ""}
      {score.toFixed(3)}
    </span>
  )
}

function formatDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function ReviewsTable({ reviews }: { reviews: Review[] }) {
  if (reviews.length === 0) {
    return (
      <div className="flex h-32 items-center justify-center text-sm text-muted-foreground">
        No reviews match this filter.
      </div>
    )
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
            <th className="px-4 py-3 font-medium">Date</th>
            <th className="px-4 py-3 font-medium">Version</th>
            <th className="px-4 py-3 font-medium">Review</th>
            <th className="px-4 py-3 font-medium">Sentiment</th>
            <th className="px-4 py-3 text-right font-medium">Score</th>
            <th className="px-4 py-3 text-right font-medium">Likes</th>
          </tr>
        </thead>
        <tbody>
          {reviews.map((r) => (
            <tr key={r.id} className="border-b border-border/60 transition-colors hover:bg-secondary/40">
              <td className="whitespace-nowrap px-4 py-3 text-xs text-muted-foreground">{formatDate(r.date)}</td>
              <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-muted-foreground">{r.version}</td>
              <td className="max-w-md px-4 py-3">
                <span className="line-clamp-2 text-foreground">{r.review || "—"}</span>
              </td>
              <td className="px-4 py-3">
                <SentimentBadge sentiment={r.sentiment} />
              </td>
              <td className="px-4 py-3 text-right">
                <ScoreCell score={r.score} />
              </td>
              <td className="px-4 py-3 text-right font-mono text-xs tabular-nums text-muted-foreground">{r.likes}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
