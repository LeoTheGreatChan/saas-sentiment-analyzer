import "server-only"
import fs from "node:fs"
import path from "node:path"
import Papa from "papaparse"
import Sentiment from "sentiment"

const DATA_PATH = path.join(process.cwd(), "uber_reviews.csv")
const SAMPLE_SIZE = 200

export type Review = {
  id: number
  user: string
  date: string // ISO
  version: string
  review: string
  likes: number
  score: number // -1..1
  sentiment: "Positive" | "Negative"
}

export type DashboardData = {
  reviews: Review[]
  versions: string[]
  totalSample: number
  avgScore: number
  criticalCount: number
  byVersion: { version: string; score: number; count: number }[]
  byHour: { hour: string; score: number; count: number }[]
}

const analyzer = new Sentiment()

/** Map an AFINN comparative score into a smooth -1..1 sentiment score. */
function normalizeScore(comparative: number): number {
  const s = Math.tanh(comparative * 1.6)
  return Math.round(s * 10000) / 10000
}

function versionSortKey(version: string): number[] {
  const parts = String(version).split(".")
  const nums = parts.map((p) => Number.parseInt(p, 10))
  return nums.every((n) => Number.isFinite(n)) ? nums : [0]
}

function compareVersionsDesc(a: string, b: string): number {
  const ka = versionSortKey(a)
  const kb = versionSortKey(b)
  const len = Math.max(ka.length, kb.length)
  for (let i = 0; i < len; i++) {
    const diff = (kb[i] ?? 0) - (ka[i] ?? 0)
    if (diff !== 0) return diff
  }
  return 0
}

type RawRow = {
  userName?: string
  content?: string
  thumbsUpCount?: string
  reviewCreatedVersion?: string
  at?: string
}

let cache: DashboardData | null = null

export function getDashboardData(): DashboardData {
  if (cache) return cache

  const file = fs.readFileSync(DATA_PATH, "utf-8")
  const parsed = Papa.parse<RawRow>(file, {
    header: true,
    skipEmptyLines: true,
  })

  const rows = (parsed.data || [])
    .map((r, i) => {
      const dateStr = (r.at || "").trim()
      const ts = Date.parse(dateStr.replace(" ", "T"))
      return {
        i,
        date: ts,
        dateStr,
        version: (r.reviewCreatedVersion || "").trim() || "Unknown",
        review: (r.content || "").trim(),
        likes: Number.parseInt(r.thumbsUpCount || "0", 10) || 0,
        user: (r.userName || "").trim(),
      }
    })
    .filter((r) => Number.isFinite(r.date))
    .sort((a, b) => b.date - a.date)
    .slice(0, SAMPLE_SIZE)

  const reviews: Review[] = rows.map((r, idx) => {
    const result = analyzer.analyze(r.review.slice(0, 512))
    const score = normalizeScore(result.comparative)
    const sentiment: Review["sentiment"] = score >= 0 ? "Positive" : "Negative"
    return {
      id: idx,
      user: r.user,
      date: new Date(r.date).toISOString(),
      version: r.version,
      review: r.review,
      likes: r.likes,
      score,
      sentiment,
    }
  })

  // Aggregations
  const versionMap = new Map<string, { sum: number; count: number }>()
  for (const rev of reviews) {
    const cur = versionMap.get(rev.version) || { sum: 0, count: 0 }
    cur.sum += rev.score
    cur.count += 1
    versionMap.set(rev.version, cur)
  }
  const byVersion = [...versionMap.entries()]
    .map(([version, { sum, count }]) => ({
      version,
      score: Math.round((sum / count) * 1000) / 1000,
      count,
    }))
    .sort((a, b) => compareVersionsDesc(b.version, a.version))

  const hourMap = new Map<number, { sum: number; count: number }>()
  for (const rev of reviews) {
    const h = new Date(rev.date).getHours()
    const cur = hourMap.get(h) || { sum: 0, count: 0 }
    cur.sum += rev.score
    cur.count += 1
    hourMap.set(h, cur)
  }
  const byHour = [...hourMap.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([hour, { sum, count }]) => ({
      hour: `${String(hour).padStart(2, "0")}:00`,
      score: Math.round((sum / count) * 1000) / 1000,
      count,
    }))

  const avgScore = reviews.length
    ? Math.round((reviews.reduce((a, r) => a + r.score, 0) / reviews.length) * 100) / 100
    : 0

  const criticalCount = reviews.filter(
    (r) => r.score < -0.6 || (r.sentiment === "Negative" && r.likes > 0),
  ).length

  const versions = [...new Set(reviews.map((r) => r.version))].sort(compareVersionsDesc)

  cache = {
    reviews,
    versions,
    totalSample: reviews.length,
    avgScore,
    criticalCount,
    byVersion,
    byHour,
  }
  return cache
}
