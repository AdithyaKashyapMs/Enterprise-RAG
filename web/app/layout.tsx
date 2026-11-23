import type { Metadata, Viewport } from "next";
import { Fraunces, Inter } from "next/font/google";
import { RegisterServiceWorker } from "@/components/RegisterServiceWorker";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const fraunces = Fraunces({ subsets: ["latin"], variable: "--font-fraunces", weight: ["600", "700"] });

export const metadata: Metadata = {
  title: "Daily 10",
  description: "Ten minutes a day, all semester. No panic before exams.",
  manifest: "/manifest.webmanifest",
  appleWebApp: { capable: true, title: "Daily 10", statusBarStyle: "default" },
  icons: { icon: "/icons/icon-192.png", apple: "/icons/apple-touch-icon.png" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f7f4" },
    { media: "(prefers-color-scheme: dark)", color: "#0f1513" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${fraunces.variable}`}>
      <body className="font-sans antialiased" style={{ fontFamily: "var(--font-inter), var(--font-sans)" }}>
        <div className="mx-auto flex min-h-dvh max-w-md flex-col">{children}</div>
        <RegisterServiceWorker />
      </body>
    </html>
  );
}
