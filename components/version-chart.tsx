"use client"

import { Bar, BarChart, CartesianGrid, Cell, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"

export function VersionChart({
  data,
}: {
  data: { version: string; score: number; count: number }[]
}) {
  return (
    <ChartContainer
      config={{
        score: { label: "Avg Score", color: "var(--chart-1)" },
      }}
      className="aspect-auto h-[300px] w-full"
    >
      <BarChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
        <CartesianGrid vertical={false} strokeDasharray="3 3" />
        <XAxis
          dataKey="version"
          tickLine={false}
          axisLine={false}
          tickMargin={8}
          angle={-35}
          textAnchor="end"
          height={70}
          interval={0}
          tick={{ fontSize: 10 }}
        />
        <YAxis domain={[-1, 1]} tickLine={false} axisLine={false} tickMargin={8} tick={{ fontSize: 11 }} />
        <ChartTooltip content={<ChartTooltipContent labelKey="version" />} />
        <Bar dataKey="score" radius={[4, 4, 4, 4]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.score >= 0 ? "var(--chart-2)" : "var(--chart-3)"} />
          ))}
        </Bar>
      </BarChart>
    </ChartContainer>
  )
}
