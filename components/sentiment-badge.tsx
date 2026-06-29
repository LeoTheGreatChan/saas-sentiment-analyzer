import { cn } from "@/lib/utils"

export function SentimentBadge({
  sentiment,
  className,
}: {
  sentiment: string
  className?: string
}) {
  const isPositive = sentiment === "Positive"
  const isNeutral = sentiment === "Neutral"
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        isPositive && "border-positive/30 bg-positive/10 text-positive",
        isNeutral && "border-border bg-secondary text-muted-foreground",
        !isPositive && !isNeutral && "border-negative/30 bg-negative/10 text-negative",
        className,
      )}
    >
      <span
        className={cn(
          "h-1.5 w-1.5 rounded-full",
          isPositive ? "bg-positive" : isNeutral ? "bg-muted-foreground" : "bg-negative",
        )}
      />
      {sentiment}
    </span>
  )
}
