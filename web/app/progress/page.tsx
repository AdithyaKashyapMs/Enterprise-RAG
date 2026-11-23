"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Progress } from "@/lib/api";
import { useStudentCode } from "@/lib/session";
import { BottomNav } from "@/components/BottomNav";
import { Card, ErrorNote, Screen, Spinner } from "@/components/ui";

export default function ProgressPage() {
  const code = useStudentCode();
  const [data, setData] = useState<Progress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!code) return;
    setError(null);
    try {
      setData(await api.progress(code));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [code]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <>
      <Screen title="Progress" subtitle={data?.student}>
        {error && <ErrorNote message={error} onRetry={load} />}
        {!data && !error && <Spinner label="Loading your progress…" />}
        {data && (
          <>
            <div className="grid grid-cols-2 gap-3">
              <Card className="flex flex-col gap-1">
                <span className="text-xs font-medium uppercase tracking-wider text-muted">Streak</span>
                <span className="text-4xl font-bold tabular-nums text-streak">🔥 {data.streak_days}</span>
                <span className="text-xs text-muted">day{data.streak_days === 1 ? "" : "s"} in a row</span>
              </Card>
              <Card className="flex flex-col gap-1">
                <span className="text-xs font-medium uppercase tracking-wider text-muted">Accuracy</span>
                <span className="text-4xl font-bold tabular-nums">
                  {data.accuracy === null ? "–" : `${Math.round(data.accuracy * 100)}%`}
                </span>
                <span className="text-xs text-muted">{data.answers} answer{data.answers === 1 ? "" : "s"} so far</span>
              </Card>
            </div>

            <Card className="flex flex-col gap-4">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">Syllabus covered</h2>
              {data.subjects.length === 0 && <p className="text-sm text-muted">No subjects yet.</p>}
              {data.subjects.map((s) => (
                <div key={s.code} className="flex flex-col gap-1.5">
                  <div className="flex items-baseline justify-between gap-2">
                    <span className="font-semibold">{s.subject}</span>
                    <span className="text-sm tabular-nums text-muted">{s.covered}/{s.topics}</span>
                  </div>
                  <div className="h-2.5 overflow-hidden rounded-full bg-line" role="progressbar" aria-valuenow={Math.round(s.coverage * 100)} aria-valuemin={0} aria-valuemax={100}>
                    <div className="h-full rounded-full bg-accent transition-all" style={{ width: `${Math.max(s.coverage * 100, 2)}%` }} />
                  </div>
                  <span className="text-xs text-muted">
                    {s.days_to_exam >= 0 ? `Exam in ${s.days_to_exam} day${s.days_to_exam === 1 ? "" : "s"}` : "Exam done"} ·{" "}
                    {new Date(s.exam_date + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" })}
                  </span>
                </div>
              ))}
            </Card>

            <Card className="flex flex-col gap-2">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">Needs practice</h2>
              {data.weak_topics.length === 0 ? (
                <p className="text-sm text-muted">Nothing yet. Topics you miss show up here, and come back in your reviews.</p>
              ) : (
                data.weak_topics.map((w) => (
                  <div key={w.topic} className="flex items-center justify-between gap-3 border-t border-line pt-2 first:border-0 first:pt-0">
                    <span>{w.topic}</span>
                    <span className="shrink-0 rounded-full bg-bad-soft px-2 py-0.5 text-xs font-medium text-bad">{w.misses} missed</span>
                  </div>
                ))
              )}
            </Card>
          </>
        )}
      </Screen>
      <BottomNav />
    </>
  );
}
