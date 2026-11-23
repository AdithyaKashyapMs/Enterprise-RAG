"use client";

export function Screen({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <main className="safe-top flex flex-1 flex-col gap-4 px-4 pb-6">
      <header>
        {subtitle && <p className="text-sm text-muted">{subtitle}</p>}
        <h1 className="text-[28px] font-bold leading-tight" style={{ fontFamily: "var(--font-fraunces), var(--font-display)" }}>
          {title}
        </h1>
      </header>
      {children}
    </main>
  );
}

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <section className={`rounded-2xl border border-line bg-card p-4 ${className}`}>{children}</section>;
}

export function Button({
  children,
  onClick,
  disabled,
  variant = "primary",
  type = "button",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  type?: "button" | "submit";
}) {
  const style =
    variant === "primary"
      ? "bg-accent text-white disabled:opacity-40"
      : "border border-line bg-card text-ink disabled:opacity-40";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`min-h-12 w-full rounded-xl px-4 text-base font-semibold transition active:scale-[0.98] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${style}`}
    >
      {children}
    </button>
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <Card className="border-bad/40 bg-bad-soft">
      <p className="text-sm text-ink">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="mt-2 text-sm font-semibold text-accent">
          Try again
        </button>
      )}
    </Card>
  );
}

export function Spinner({ label }: { label: string }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 py-16 text-muted" role="status">
      <span className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
      <span className="text-sm">{label}</span>
    </div>
  );
}
