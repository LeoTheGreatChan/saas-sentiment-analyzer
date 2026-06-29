import { generateText, Output } from "ai"
import { z } from "zod"
import { analyzeLocally } from "@/lib/local-sentiment"

export const maxDuration = 30

const schema = z.object({
  sentiment: z.enum(["Positive", "Negative", "Neutral"]),
  score: z.number().min(-1).max(1),
  confidence: z.number().min(0).max(1),
  summary: z.string(),
  themes: z.array(z.string()),
})

export async function POST(req: Request) {
  try {
    const { text } = await req.json()

    if (!text || typeof text !== "string" || text.trim().length === 0) {
      return Response.json({ error: "Please provide text to analyze." }, { status: 400 })
    }

    try {
      const { experimental_output } = await generateText({
        model: "openai/gpt-5.4-mini",
        experimental_output: Output.object({ schema }),
        system:
          "You are a product-feedback sentiment analyst for a ride-hailing app. " +
          "Analyze the customer review and return a sentiment classification, a numeric score " +
          "from -1 (extremely negative) to 1 (extremely positive), a confidence value from 0 to 1, " +
          "a one-sentence summary, and up to 3 short product themes (e.g. 'pricing', 'driver behavior', 'app bug').",
        prompt: `Analyze this customer review:\n\n"""${text.slice(0, 2000)}"""`,
      })

      return Response.json({ ...experimental_output, engine: "ai" })
    } catch (aiErr) {
      // AI Gateway unavailable (e.g. no credits) — fall back to the local lexicon scorer
      console.log("[v0] AI gateway unavailable, using local scorer:", aiErr instanceof Error ? aiErr.message : aiErr)
      return Response.json({ ...analyzeLocally(text), engine: "local" })
    }
  } catch (err) {
    console.log("[v0] analyze error:", err instanceof Error ? err.message : err)
    return Response.json({ error: "Failed to analyze text. Please try again." }, { status: 500 })
  }
}
