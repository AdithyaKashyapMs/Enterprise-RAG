// Typed client for the Daily 10 API. Requests go to /api/study/*, which Next.js proxies
// to the FastAPI backend (see next.config.ts).

export type Card = {
  topic_id: number;
  subject: string;
  title: string;
  summary: string;
  key_points: string[];
  source: string;
  exam_weight: number;
};

export type SessionQuestion = {
  id: number;
  topic: string;
  prompt: string;
  options: string[];
  kind: "new" | "review";
  answered: boolean;
};

export type PastQuestion = { text: string; year: number | null; marks: number | null };

export type Session = {
  day: string;
  student: string;
  card: Card | null;
  questions: SessionQuestion[];
  past_question: PastQuestion | null;
  notes: string[];
  completed: boolean;
};

export type AnswerResult = {
  correct: boolean;
  answer_index: number;
  explanation: string;
  evidence: string;
  next_review: string;
  session_completed: boolean;
};

export type SubjectProgress = {
  subject: string;
  code: string;
  exam_date: string;
  topics: number;
  covered: number;
  coverage: number;
  days_to_exam: number;
};

export type Progress = {
  student: string;
  subjects: SubjectProgress[];
  streak_days: number;
  accuracy: number | null;
  answers: number;
  weak_topics: { topic: string; misses: number }[];
};

export type Profile = {
  code: string;
  name: string;
  notify_time: string;
  telegram_linked: boolean;
  subjects: { code: string; name: string; exam_date: string }[];
};

export type Subject = { code: string; name: string; department: string; semester: number; topics: number };

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/study${path}`, {
      method,
      headers: body ? { "content-type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      cache: "no-store",
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {}
    throw new ApiError(res.status, detail || `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  register: (name: string, notify_time: string) =>
    call<{ code: string; name: string; notify_time: string }>("POST", "/students", { name, notify_time }),
  profile: (code: string) => call<Profile>("GET", `/students/${code}`),
  updateReminder: (code: string, notify_time: string) => call("PATCH", `/students/${code}`, { notify_time }),
  subjects: () => call<Subject[]>("GET", "/subjects"),
  enroll: (code: string, subject_code: string, exam_date: string) =>
    call("POST", `/students/${code}/enroll`, { subject_code, exam_date }),
  today: (code: string) => call<Session>("GET", `/students/${code}/today`),
  answer: (code: string, question_id: number, chosen_index: number) =>
    call<AnswerResult>("POST", `/students/${code}/answers`, { question_id, chosen_index }),
  report: (question_id: number) => call("POST", `/questions/${question_id}/report`),
  progress: (code: string) => call<Progress>("GET", `/students/${code}/progress`),
};
