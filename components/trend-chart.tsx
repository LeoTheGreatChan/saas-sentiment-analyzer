"use client"

import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function TrendChart({
  data,
}: {
  data: { hour: string; score: number; count: number }[]
}) {
  return (
    <ChartContainer
      config={{
        score: { label: "Avg Score", color: "var(--chart-1)" },
      }}
      className="aspect-auto h-[300px] w-full"
    >
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <defs>
          <linearGradient id="trendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--color-score)" stopOpacity={0.4} />
            <stop offset="95%" stopColor="var(--color-score)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis dataKey="hour" tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
        <YAxis domain={[-1, 1]} tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
        <ChartTooltip content={<ChartTooltipContent labelKey="hour" />} />
        <Area
          type="monotone"
          dataKey="score"
          stroke="var(--color-score)"
          strokeWidth={2}
          fill="url(#trendFill)"
          dot={{ r: 3, fill: "var(--color-score)" }}
          activeDot={{ r: 5 }}
        />
      </AreaChart>
    </ChartContainer>
  )
}
