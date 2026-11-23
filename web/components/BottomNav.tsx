"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/today", label: "Today", icon: "M4 6h16M4 12h10M4 18h6" },
  { href: "/progress", label: "Progress", icon: "M5 19V11M12 19V5M19 19v-6" },
  { href: "/settings", label: "Me", icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm-7 8a7 7 0 0 1 14 0" },
];

export function BottomNav() {
  const path = usePathname();
  return (
    <nav className="safe-bottom sticky bottom-0 z-10 border-t border-line bg-paper/95 backdrop-blur">
      <ul className="grid grid-cols-3">
        {TABS.map((t) => {
          const active = path?.startsWith(t.href);
          return (
            <li key={t.href}>
              <Link
                href={t.href}
                className={`flex flex-col items-center gap-1 py-2 text-xs font-medium focus-visible:outline-2 focus-visible:outline-accent ${
                  active ? "text-accent" : "text-muted"
                }`}
                aria-current={active ? "page" : undefined}
              >
                <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
                  <path d={t.icon} />
                </svg>
                {t.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
