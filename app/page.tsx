import { getDashboardData } from "@/lib/reviews"
import { TopNav } from "@/components/top-nav"
import { Dashboard } from "@/components/dashboard"

export default function Page() {
  const data = getDashboardData()
  return (
    <main className="min-h-screen bg-background">
      <TopNav />
      <Dashboard data={data} />
    </main>
  )
}
