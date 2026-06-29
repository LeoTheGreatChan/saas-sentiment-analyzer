import { Activity, AlertTriangle, MessageSquare, GitBranch } from "lucide-react"
import { cn } from "@/lib/utils"

type Metric = {
  label: string
  value: string
  hint: string
  icon: React.ReactNode
  tone?: "default" | "positive" | "negative" | "warning"
}

function MetricCard({ label, value, hint, icon, tone = "default" }: Metric) {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-border bg-card p-5 transition-colors hover:border-primary/40">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">{label}</span>
        <span
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-secondary text-muted-foreground",
            tone === "positive" && "border-positive/30 bg-positive/10 text-positive",
            tone === "negative" && "border-negative/30 bg-negative/10 text-negative",
            tone === "warning" && "border-chart-4/30 bg-chart-4/10 text-chart-4",
            tone === "default" && "border-primary/30 bg-primary/10 text-primary",
          )}
        >
          {icon}
        </span>
      </div>
      <div className="mt-4 font-mono text-3xl font-semibold tracking-tight tabular-nums">{value}</div>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </div>
  )
}

export function MetricsGrid({
  totalSample,
  avgScore,
  criticalCount,
  activeVersion,
}: {
  totalSample: number
  avgScore: number
  criticalCount: number
  activeVersion: string
}) {
  const avgTone = avgScore >= 0.15 ? "positive" : avgScore <= -0.15 ? "negative" : "default"
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <MetricCard
        label="Total Sample"
        value={totalSample.toLocaleString()}
        hint="Most recent reviews analyzed"
        icon={<MessageSquare className="h-4 w-4" />}
      />
      <MetricCard
        label="Avg Sentiment"
        value={avgScore.toFixed(2)}
        hint="Mean score from -1 to +1"
        icon={<Activity className="h-4 w-4" />}
        tone={avgTone}
      />
      <MetricCard
        label="Critical Alerts"
        value={criticalCount.toLocaleString()}
        hint="High-priority negative reviews"
        icon={<AlertTriangle className="h-4 w-4" />}
        tone={criticalCount > 0 ? "warning" : "default"}
      />
      <MetricCard
        label="Active Version"
        value={activeVersion}
        hint="Current filter selection"
        icon={<GitBranch className="h-4 w-4" />}
      />
    </div>
  )
}
