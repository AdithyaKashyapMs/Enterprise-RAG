"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { api, ApiError } from "@/lib/api";
import { setCode } from "@/lib/session";
import { Button, Card, ErrorNote, Screen } from "@/components/ui";

export default function Login() {
  const router = useRouter();
  const [mode, setMode] = useState<"new" | "code">("new");
  const [name, setName] = useState("");
  const [time, setTime] = useState("19:00");
  const [code, setCodeInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function createAccount(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const res = await api.register(name.trim(), time);
      setCode(res.code);
      router.replace("/settings?welcome=1");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setBusy(false);
    }
  }

  async function signIn(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.profile(code.trim().toUpperCase());
      setCode(code.trim());
      router.replace("/today");
    } catch (err) {
      setError(err instanceof ApiError && err.status === 404 ? "No account with that code." : (err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Screen title="Daily 10" subtitle="Ten minutes a day. No panic before exams.">
      <Card>
        <ol className="space-y-2 text-sm text-muted">
          <li>📖 One short topic from your syllabus</li>
          <li>✅ 5 quick questions from your own notes</li>
          <li>🔁 Old topics come back just before you'd forget them</li>
        </ol>
      </Card>

      <div className="grid grid-cols-2 rounded-xl border border-line bg-card p-1 text-sm font-semibold" role="tablist">
        {(["new", "code"] as const).map((m) => (
          <button
            key={m}
            role="tab"
            aria-selected={mode === m}
            onClick={() => setMode(m)}
            className={`rounded-lg py-2 ${mode === m ? "bg-accent-soft text-accent" : "text-muted"}`}
          >
            {m === "new" ? "I'm new" : "I have a code"}
          </button>
        ))}
      </div>

      {mode === "new" ? (
        <form onSubmit={createAccount} className="flex flex-col gap-3">
          <label className="text-sm font-medium" htmlFor="name">Your name</label>
          <input
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            maxLength={120}
            autoComplete="name"
            className="min-h-12 rounded-xl border border-line bg-card px-4 text-base outline-none focus:border-accent"
          />
          <label className="text-sm font-medium" htmlFor="time">Remind me every day at</label>
          <input
            id="time"
            type="time"
            value={time}
            onChange={(e) => setTime(e.target.value)}
            required
            className="min-h-12 rounded-xl border border-line bg-card px-4 text-base outline-none focus:border-accent"
          />
          <Button type="submit" disabled={busy || !name.trim()}>{busy ? "Creating…" : "Create my account"}</Button>
        </form>
      ) : (
        <form onSubmit={signIn} className="flex flex-col gap-3">
          <label className="text-sm font-medium" htmlFor="code">Your 6-character code</label>
          <input
            id="code"
            value={code}
            onChange={(e) => setCodeInput(e.target.value.toUpperCase())}
            required
            maxLength={6}
            autoCapitalize="characters"
            className="min-h-12 rounded-xl border border-line bg-card px-4 font-mono text-lg tracking-[0.3em] outline-none focus:border-accent"
          />
          <Button type="submit" disabled={busy || code.trim().length < 6}>{busy ? "Checking…" : "Continue"}</Button>
        </form>
      )}

      {error && <ErrorNote message={error} />}
    </Screen>
  );
}
