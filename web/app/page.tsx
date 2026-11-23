"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { getCode } from "@/lib/session";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(getCode() ? "/today" : "/login");
  }, [router]);
  return null;
}
