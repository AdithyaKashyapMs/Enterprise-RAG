"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type AnswerResult, type Profile, type Session } from "@/lib/api";
import { useStudentCode } from "@/lib/session";
import { BottomNav } from "@/components/BottomNav";
import { Button, Card, ErrorNote, Screen, Spinner } from "@/components/ui";

type Step = { kind: "card" } | { kind: "question"; index: number } | { kind: "past" } | { kind: "done" };

const LETTERS = ["A", "B", "C", "D"];

function firstStep(s: Session): Step {
  const open = s.questions.findIndex((q) => !q.answered);
  if (s.completed || (s.questions.length > 0 && open === -1)) return s.past_question ? { kind: "past" } : { kind: "done" };
  if (open > 0) return { kind: "question", index: open };
  if (s.card) return { kind: "card" };
  if (s.questions.length) return { kind: "question", index: 0 };
  return { kind: "done" };
}

export default function Today() {
  const code = useStudentCode();
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState<Step>({ kind: "card" });
  const [results, setResults] = useState<Record<number, AnswerResult & { chosen: number }>>({});
  const [busy, setBusy] = useState(false);
  const [reported, setReported] = useState<Set<number>>(new Set());

  const load = useCallback(async () => {
    if (!code) return;
    setError(null);
    try {
      const [s, p] = await Promise.all([api.today(code), api.profile(code)]);
      setSession(s);
      setProfile(p);
      setStep(firstStep(s));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [code]);

  useEffect(() => {
    load();
  }, [load]);

  const total = session?.questions.length ?? 0;
  const answered = useMemo(
    () => (session ? session.questions.filter((q) => q.answered || results[q.id]).length : 0),
    [session, results],
  );
  const correctToday = Object.values(results).filter((r) => r.correct).length;

  async function choose(qid: number, index: number) {
    if (!code || results[qid] || busy) return;
    setBusy(true);
    try {
      const r = await api.answer(code, qid, index);
      setResults((prev) => ({ ...prev, [qid]: { ...r, chosen: index } }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  function next() {
    if (!session) return;
    if (step.kind === "card") return setStep(total ? { kind: "question", index: 0 } : session.past_question ? { kind: "past" } : { kind: "done" });
    if (step.kind === "question") {
      const after = session.questions.findIndex((q, i) => i > step.index && !q.answered && !results[q.id]);
      if (after !== -1) return setStep({ kind: "question", index: after });
      return setStep(session.past_question ? { kind: "past" } : { kind: "done" });
    }
    if (step.kind === "past") return setStep({ kind: "done" });
  }

  async function report(qid: number) {
    try {
      await api.report(qid);
      setReported((prev) => new Set(prev).add(qid));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  const dateLabel = session
    ? new Date(session.day + "T00:00:00").toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "short" })
    : "";

  let body: React.ReactNode;
  if (!code || (!session && !error)) {
    body = <Spinner label="Getting today's 10 minutes ready…" />;
  } else if (!session) {
    body = <ErrorNote message={error ?? "Couldn't load today's session."} onRetry={load} />;
  } else if (profile && profile.subjects.length === 0) {
    body = (
      <Card className="flex flex-col gap-3">
        <p className="font-semibold">Add your subjects to start</p>
        <p className="text-sm text-muted">Pick your subjects and exam dates; Daily 10 spreads the syllabus over the days you have.</p>
        <Link href="/settings" className="flex min-h-12 items-center justify-center rounded-xl bg-accent font-semibold text-white">
          Add subjects
        </Link>
      </Card>
    );
  } else if (step.kind === "card" && session.card) {
    const c = session.card;
    body = (
      <>
        <Card className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2 text-xs font-medium">
            <span className="rounded-full bg-accent-soft px-2.5 py-1 text-accent">{c.subject}</span>
            {c.exam_weight > 0 && (
              <span className="rounded-full bg-card px-2.5 py-1 text-streak ring-1 ring-line">📝 asked {c.exam_weight}× in exams</span>
            )}
          </div>
          <h2 className="text-2xl font-bold leading-snug" style={{ fontFamily: "var(--font-fraunces), var(--font-display)" }}>
            {c.title}
          </h2>
          {c.summary ? <p className="leading-relaxed">{c.summary}</p> : <p className="text-muted">Summary not generated yet.</p>}
          {c.key_points?.length > 0 && (
            <ul className="space-y-2 border-t border-line pt-3">
              {c.key_points.map((p, i) => (
                <li key={i} className="flex gap-2 text-[15px] leading-snug">
                  <span className="mt-1 h-2 w-2 shrink-0 rounded-full bg-accent" aria-hidden />
                  {p}
                </li>
              ))}
            </ul>
          )}
          <p className="text-xs text-muted">From your notes: {c.source}</p>
        </Card>
        <Button onClick={next}>{total ? `Start ${total} questions` : "Continue"}</Button>
      </>
    );
  } else if (step.kind === "question") {
    const q = session.questions[step.index];
    const r = results[q.id];
    body = (
      <>
        <Card className="flex flex-col gap-4">
          <div className="flex items-center justify-between text-xs font-medium text-muted">
            <span>{q.kind === "new" ? "🆕 New" : "🔁 Review"} · {q.topic}</span>
            <span>{step.index + 1}/{total}</span>
          </div>
          <p className="text-lg font-semibold leading-snug">{q.prompt}</p>
          <div className="flex flex-col gap-2" role="radiogroup" aria-label="Answer options">
            {q.options.map((opt, i) => {
              const isRight = r && i === r.answer_index;
              const isWrongPick = r && i === r.chosen && !r.correct;
              const tone = isRight
                ? "border-good bg-good-soft"
                : isWrongPick
                  ? "border-bad bg-bad-soft"
                  : "border-line bg-card active:bg-accent-soft";
              return (
                <button
                  key={i}
                  role="radio"
                  aria-checked={r?.chosen === i}
                  disabled={!!r || busy}
                  onClick={() => choose(q.id, i)}
                  className={`flex min-h-14 items-center gap-3 rounded-xl border-2 px-3 py-2 text-left text-[15px] transition ${tone}`}
                >
                  <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-paper text-sm font-bold text-muted">
                    {isRight ? "✓" : isWrongPick ? "✕" : LETTERS[i]}
                  </span>
                  {opt}
                </button>
              );
            })}
          </div>
          {r && (
            <div className={`rounded-xl p-3 text-[15px] ${r.correct ? "bg-good-soft" : "bg-bad-soft"}`} aria-live="polite">
              <p className="font-semibold">{r.correct ? "Correct!" : "Not quite."}</p>
              <p className="mt-1 leading-relaxed">{r.explanation}</p>
              <p className="mt-2 text-sm text-muted">From your notes: “{r.evidence}”</p>
              <p className="mt-1 text-xs text-muted">
                {r.correct ? "You'll see this again on " : "This comes back tomorrow, "}
                {new Date(r.next_review + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" })}
              </p>
            </div>
          )}
        </Card>
        {r && (
          <>
            <Button onClick={next}>Next</Button>
            <button
              onClick={() => report(q.id)}
              disabled={reported.has(q.id)}
              className="text-sm text-muted underline-offset-4 hover:underline disabled:no-underline"
            >
              {reported.has(q.id) ? "Thanks, sent to your faculty" : "This question looks wrong"}
            </button>
          </>
        )}
      </>
    );
  } else if (step.kind === "past" && session.past_question) {
    const p = session.past_question;
    body = (
      <>
        <Card className="flex flex-col gap-3">
          <p className="text-xs font-medium text-streak">
            📝 Past exam question{p.year ? ` · ${p.year}` : ""}{p.marks ? ` · ${p.marks} marks` : ""}
          </p>
          <p className="text-lg font-semibold leading-snug">{p.text}</p>
          <p className="text-sm text-muted">Take 2 minutes: jot down the points you'd write. Only you see this.</p>
          <textarea
            rows={5}
            aria-label="Your answer outline"
            className="rounded-xl border border-line bg-paper p-3 text-base outline-none focus:border-accent"
            placeholder="• First point…"
          />
        </Card>
        <Button onClick={next}>Finish today</Button>
      </>
    );
  } else {
    const nothing = total === 0 && !session.card;
    body = (
      <Card className="flex flex-col items-center gap-2 py-8 text-center">
        <span className="text-5xl" aria-hidden>{nothing ? "🌿" : "🎉"}</span>
        <p className="text-xl font-bold">{nothing ? "Nothing due today" : "Done for today!"}</p>
        {!nothing && Object.keys(results).length > 0 && (
          <p className="text-muted">{correctToday} of {Object.keys(results).length} right today</p>
        )}
        {session.notes.map((n, i) => (
          <p key={i} className="text-sm text-muted">{n}</p>
        ))}
        {profile && <p className="text-sm text-muted">See you tomorrow at {profile.notify_time}.</p>}
        <Link href="/progress" className="mt-2 text-sm font-semibold text-accent">See your progress →</Link>
      </Card>
    );
  }

  return (
    <>
      <Screen title="Today" subtitle={dateLabel}>
        {session && total > 0 && step.kind !== "done" && (
          <div className="flex gap-1.5" aria-label={`${answered} of ${total} questions answered`}>
            {session.questions.map((q) => (
              <span
                key={q.id}
                className={`h-1.5 flex-1 rounded-full ${
                  results[q.id] ? (results[q.id].correct ? "bg-good" : "bg-bad") : q.answered ? "bg-accent" : "bg-line"
                }`}
              />
            ))}
          </div>
        )}
        {session?.notes.length && step.kind === "card" ? (
          session.notes.map((n, i) => (
            <p key={i} className="rounded-xl bg-accent-soft px-3 py-2 text-sm text-ink">{n}</p>
          ))
        ) : null}
        {error && session && <ErrorNote message={error} />}
        {body}
      </Screen>
      <BottomNav />
    </>
  );
}
