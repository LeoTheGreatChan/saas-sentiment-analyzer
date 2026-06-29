"use client"

import { useMemo, useState } from "react"
import { Download, LineChart, TriangleAlert, ClipboardList } from "lucide-react"
import type { DashboardData } from "@/lib/reviews"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { MetricsGrid } from "@/components/metrics-grid"
import { SentimentAnalyzer } from "@/components/sentiment-analyzer"
import { VersionChart } from "@/components/version-chart"
import { TrendChart } from "@/components/trend-chart"
import { ReviewsTable } from "@/components/reviews-table"
import { CriticalAlerts } from "@/components/critical-alerts"
import { cn } from "@/lib/utils"

const ALL_VERSIONS = "All Versions"

function toCsv(rows: DashboardData["reviews"]) {
  const headers = ["Date", "Version", "Review", "Sentiment", "Score", "Likes"]
  const escape = (v: string) => `"${String(v).replace(/"/g, '""')}"`
  const lines = rows.map((r) =>
    [r.date, r.version, escape(r.review), r.sentiment, r.score, r.likes].join(","),
  )
  return [headers.join(","), ...lines].join("\n")
}

export function Dashboard({ data }: { data: DashboardData }) {
  const [version, setVersion] = useState(ALL_VERSIONS)
  const [view, setView] = useState<"trends" | "alerts">("trends")

  const filtered = useMemo(
    () => (version === ALL_VERSIONS ? data.reviews : data.reviews.filter((r) => r.version === version)),
    [data.reviews, version],
  )

  const metrics = useMemo(() => {
    const total = filtered.length
    const avg = total ? filtered.reduce((a, r) => a + r.score, 0) / total : 0
    const critical = filtered.filter((r) => r.score < -0.6 || (r.sentiment === "Negative" && r.likes > 0)).length
    return {
      total,
      avg: Math.round(avg * 100) / 100,
      critical,
      active: version === ALL_VERSIONS ? "Multiple" : version,
    }
  }, [filtered, version])

  function download() {
    const blob = new Blob([toCsv(filtered)], { type: "text/csv" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = "uber_sentiment_export.csv"
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-8 sm:px-6">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-pretty text-2xl font-semibold tracking-tight sm:text-3xl">
            Product Insights Dashboard
          </h1>
          <p className="mt-1 max-w-xl text-pretty text-sm text-muted-foreground">
            Sentiment intelligence across the latest {data.totalSample} Uber reviews — track release health, surface
            critical issues, and analyze feedback in real time.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={version} onValueChange={setVersion}>
            <SelectTrigger className="w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL_VERSIONS}>All Versions</SelectItem>
              {data.versions.map((v) => (
                <SelectItem key={v} value={v}>
                  {v}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button variant="outline" size="icon" onClick={download} aria-label="Download dataset">
            <Download className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Metrics */}
      <MetricsGrid
        totalSample={metrics.total}
        avgScore={metrics.avg}
        criticalCount={metrics.critical}
        activeVersion={metrics.active}
      />

      {/* Live analyzer */}
      <SentimentAnalyzer />

      {/* View toggle */}
      <div>
        <div className="mb-4 flex items-center justify-between gap-4">
          <h2 className="text-lg font-semibold tracking-tight">Explore Review Insights</h2>
          <div className="inline-flex rounded-lg border border-border bg-card p-1">
            <button
              onClick={() => setView("trends")}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                view === "trends" ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <LineChart className="h-4 w-4" /> Performance Trends
            </button>
            <button
              onClick={() => setView("alerts")}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                view === "alerts" ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <TriangleAlert className="h-4 w-4" /> Critical Alerts
            </button>
          </div>
        </div>

        {view === "trends" ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Sentiment by Version</CardTitle>
                <CardDescription>App health by release (mean score)</CardDescription>
              </CardHeader>
              <CardContent>
                <VersionChart data={data.byVersion} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Time-Based Trend</CardTitle>
                <CardDescription>Average sentiment by hour of day</CardDescription>
              </CardHeader>
              <CardContent>
                <TrendChart data={data.byHour} />
              </CardContent>
            </Card>
          </div>
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">High-Priority Customer Issues</CardTitle>
              <CardDescription>
                {metrics.active === "Multiple" ? "Across all versions" : `Filtered to ${metrics.active}`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CriticalAlerts reviews={filtered} />
            </CardContent>
          </Card>
        )}
      </div>

      {/* Review explorer */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <div className="flex items-center gap-2">
            <ClipboardList className="h-4 w-4 text-muted-foreground" />
            <div>
              <CardTitle className="text-base">Review Explorer</CardTitle>
              <CardDescription>
                {metrics.active === "Multiple" ? "All versions" : metrics.active} · {filtered.length} reviews
              </CardDescription>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={download}>
            <Download className="h-4 w-4" /> Export CSV
          </Button>
        </CardHeader>
        <CardContent className="px-0">
          <ReviewsTable reviews={filtered} />
        </CardContent>
      </Card>
    </div>
  )
}
