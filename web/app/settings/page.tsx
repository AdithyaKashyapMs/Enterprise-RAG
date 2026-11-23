"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { api, type Profile, type Subject } from "@/lib/api";
import { clearCode, useStudentCode } from "@/lib/session";
import { BottomNav } from "@/components/BottomNav";
import { Button, Card, ErrorNote, Screen, Spinner } from "@/components/ui";

const BOT = process.env.NEXT_PUBLIC_TELEGRAM_BOT ?? "";

function inDays(n: number) {
  const d = new Date();
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}

function SettingsInner() {
  const code = useStudentCode();
  const router = useRouter();
  const welcome = useSearchParams()?.get("welcome") === "1";
  const [profile, setProfile] = useState<Profile | null>(null);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [pick, setPick] = useState("");
  const [exam, setExam] = useState(inDays(90));
  const [time, setTime] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const load = useCallback(async () => {
    if (!code) return;
    try {
      const [p, s] = await Promise.all([api.profile(code), api.subjects()]);
      setProfile(p);
      setTime(p.notify_time);
      setSubjects(s);
      const available = s.filter((x) => !p.subjects.some((e) => e.code === x.code));
      setPick(available[0]?.code ?? "");
    } catch (e) {
      setError((e as Error).message);
    }
  }, [code]);

  useEffect(() => {
    load();
  }, [load]);

  async function addSubject() {
    if (!code || !pick) return;
    try {
      await api.enroll(code, pick, exam);
      setSaved("Subject added. Your plan starts today.");
      await load();
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function saveTime() {
    if (!code) return;
    try {
      await api.updateReminder(code, time);
      setSaved(`Reminder set for ${time} every day.`);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function copyCode() {
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {}
  }

  if (!profile) return error ? <ErrorNote message={error} onRetry={load} /> : <Spinner label="Loading…" />;
  const available = subjects.filter((x) => !profile.subjects.some((e) => e.code === x.code));

  return (
    <>
      {welcome && (
        <Card className="border-accent/40 bg-accent-soft">
          <p className="font-semibold">Welcome, {profile.name.split(" ")[0]}!</p>
          <p className="mt-1 text-sm">Your code is <b className="font-mono tracking-widest">{profile.code}</b>. Keep it to sign in on another phone.</p>
          <p className="mt-1 text-sm">Next: add your subjects and exam dates below.</p>
        </Card>
      )}
      {saved && <p className="rounded-xl bg-good-soft px-3 py-2 text-sm" role="status">{saved}</p>}
      {error && <ErrorNote message={error} />}

      <Card className="flex flex-col gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">My subjects</h2>
        {profile.subjects.length === 0 && <p className="text-sm text-muted">None yet.</p>}
        {profile.subjects.map((s) => (
          <div key={s.code} className="flex items-baseline justify-between gap-3 border-b border-line pb-2">
            <span>{s.name}</span>
            <span className="shrink-0 text-xs text-muted">exam {new Date(s.exam_date + "T00:00:00").toLocaleDateString(undefined, { day: "numeric", month: "short" })}</span>
          </div>
        ))}
        {available.length > 0 ? (
          <div className="flex flex-col gap-2 pt-1">
            <label htmlFor="subject" className="text-sm font-medium">Add a subject</label>
            <select id="subject" value={pick} onChange={(e) => setPick(e.target.value)} className="min-h-12 rounded-xl border border-line bg-card px-3 text-base">
              {available.map((s) => (
                <option key={s.code} value={s.code}>{s.code} · {s.name}</option>
              ))}
            </select>
            <label htmlFor="exam" className="text-sm font-medium">Exam date</label>
            <input id="exam" type="date" min={inDays(1)} value={exam} onChange={(e) => setExam(e.target.value)} className="min-h-12 rounded-xl border border-line bg-card px-3 text-base" />
            <Button onClick={addSubject} disabled={!pick}>Add subject</Button>
          </div>
        ) : (
          <p className="text-sm text-muted">{subjects.length ? "You've added every available subject." : "No subjects yet: your faculty adds them."}</p>
        )}
      </Card>

      <Card className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted">Daily reminder</h2>
        <div className="flex gap-2">
          <input type="time" aria-label="Reminder time" value={time} onChange={(e) => setTime(e.target.value)} className="min-h-12 flex-1 rounded-xl border border-line bg-card px-3 text-base" />
          <div className="w-28"><Button variant="ghost" onClick={saveTime} disabled={time === profile.notify_time}>Save</Button></div>
        </div>
        <p className="text-sm text-muted">
          {profile.telegram_linked
            ? "✅ Telegram reminders are on."
            : BOT
              ? <>Get reminders on Telegram: open <a className="font-semibold text-accent" href={`https://t.me/${BOT}?start=${profile.code}`}>@{BOT}</a> and tap Start.</>
              : "Telegram reminders will appear here once your college sets up the bot."}
        </p>
      </Card>

      <Card className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm text-muted">Your code</p>
          <p className="font-mono text-xl tracking-[0.3em]">{profile.code}</p>
        </div>
        <button onClick={copyCode} className="rounded-lg bg-accent-soft px-3 py-2 text-sm font-semibold text-accent">{copied ? "Copied" : "Copy"}</button>
      </Card>

      <Button variant="ghost" onClick={() => { clearCode(); router.replace("/login"); }}>Sign out</Button>
    </>
  );
}

export default function Settings() {
  return (
    <>
      <Screen title="Me">
        <Suspense fallback={<Spinner label="Loading…" />}>
          <SettingsInner />
        </Suspense>
      </Screen>
      <BottomNav />
    </>
  );
}
