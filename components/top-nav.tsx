import { Gauge } from "lucide-react"

export function TopNav() {
  return (
    <header className="sticky top-0 z-30 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Gauge className="h-4 w-4" />
          </span>
          <span className="text-sm font-semibold tracking-tight">Pulse</span>
          <span className="hidden text-sm text-muted-foreground sm:inline">/ Uber Product Insights</span>
        </div>
        <nav className="flex items-center gap-1 text-sm">
          <span className="rounded-md bg-secondary px-2.5 py-1 text-foreground">Overview</span>
          <span className="hidden rounded-md px-2.5 py-1 text-muted-foreground sm:inline">Reviews</span>
          <span className="hidden rounded-md px-2.5 py-1 text-muted-foreground sm:inline">Releases</span>
        </nav>
      </div>
    </header>
  )
}
