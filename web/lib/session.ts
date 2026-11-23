"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

// The student's code is their sign-in for now (college email sign-in is a later step).
const KEY = "daily10.code";

export function getCode(): string | null {
  try {
    return localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function setCode(code: string) {
  try {
    localStorage.setItem(KEY, code.toUpperCase());
  } catch {}
}

export function clearCode() {
  try {
    localStorage.removeItem(KEY);
  } catch {}
}

/** The signed-in student's code; sends the user to /login if there isn't one. */
export function useStudentCode(): string | null {
  const router = useRouter();
  const [code, setState] = useState<string | null>(null);
  useEffect(() => {
    // A link like /today?code=A1B2C3 (e.g. from a reminder) signs the student in on this device.
    const fromLink = new URLSearchParams(window.location.search).get("code");
    if (fromLink && /^[A-Za-z0-9]{6}$/.test(fromLink)) setCode(fromLink);
    const c = getCode();
    if (!c) router.replace("/login");
    else setState(c);
  }, [router]);
  return code;
}
