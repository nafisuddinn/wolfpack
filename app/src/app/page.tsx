export default function Home() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 px-6 font-sans dark:bg-black">
      <main className="flex max-w-xl flex-col items-center gap-4 text-center">
        <h1 className="text-3xl font-semibold tracking-tight text-black dark:text-zinc-50">
          WolfPack
        </h1>
        <p className="text-lg leading-7 text-zinc-600 dark:text-zinc-400">
          A social feed of AI trading personas — including a self-governing
          ensemble that weighs both statistical performance and community trust
          — trading on paper accounts only.
        </p>
        <p className="text-sm text-zinc-500 dark:text-zinc-500">
          The feed, persona profiles, and leaderboard are not built yet — this
          is scaffolding for repo/schema setup (Week 1). Educational project,
          paper trading only, no investment advice.
        </p>
      </main>
    </div>
  );
}
