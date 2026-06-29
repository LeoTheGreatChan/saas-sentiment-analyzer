import type { Review } from "@/lib/reviews"
import { AlertTriangle, ThumbsUp } from "lucide-react"
import { cn } from "@/lib/utils"

function formatDate(iso: string) {
  const d = new Date(iso)
  return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
}

export function CriticalAlerts({ reviews }: { reviews: Review[] }) {
  const alerts = reviews
    .filter((r) => r.score < -0.6 || (r.sentiment === "Negative" && r.likes > 0))
    .sort((a, b) => b.likes - a.likes || a.score - b.score)

  return (
    <div className="space-y-4">
      <div className="flex items-start gap-2 rounded-lg border border-chart-4/30 bg-chart-4/10 px-4 py-3 text-sm text-foreground">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-chart-4" />
        <p>
          Showing reviews with very negative sentiment{" "}
          <span className="text-muted-foreground">(score &lt; -0.6)</span> or high community agreement{" "}
          <span className="text-muted-foreground">(likes)</span>.
        </p>
      </div>

      {alerts.length === 0 ? (
        <div className="flex h-32 items-center justify-center rounded-lg border border-border text-sm text-muted-foreground">
          No critical alerts for this version. Nice and calm.
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {alerts.map((r) => (
            <div key={r.id} className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-negative/40">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="font-mono">{r.version}</span>
                <span>{formatDate(r.date)}</span>
              </div>
              <p className="mt-2 line-clamp-3 text-sm leading-relaxed text-foreground">{r.review || "—"}</p>
              <div className="mt-3 flex items-center gap-3 text-xs">
                <span className="font-mono tabular-nums text-negative">score {r.score.toFixed(3)}</span>
                <span
                  className={cn(
                    "inline-flex items-center gap-1 text-muted-foreground",
                    r.likes > 0 && "text-foreground",
                  )}
                >
                  <ThumbsUp className="h-3 w-3" /> {r.likes}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
