# Daily 10: mobile web app (Next.js)

The student app for Daily 10: a mobile-first, installable web app (PWA). Students open it in
the phone browser and tap **Add to Home Screen**; it then opens full-screen with its own icon.

Screens: **Today** (concept card → one question at a time with instant feedback → past exam
question), **Progress** (streak, accuracy, syllabus covered per subject, topics to practise),
**Me** (subjects and exam dates, reminder time, Telegram link, sign-in code).

```bash
# 1. Backend (repo root)
uvicorn app.main:app --port 8000

# 2. This app
cd web
npm install
npm run dev            # http://localhost:3000, also on your Wi-Fi: http://<laptop-ip>:3000
```

The browser only talks to this server: `/api/*` is proxied to the FastAPI backend
(`API_URL`, default `http://localhost:8000`), so phones on the same Wi-Fi need one address and
no CORS setup. Optional: `NEXT_PUBLIC_TELEGRAM_BOT=<bot username>` shows a one-tap Telegram link.

Installing as an app (service worker, full-screen) needs HTTPS, so on a phone it works once
deployed (e.g. Vercel for this app plus any host for the API) or through a tunnel such as
`cloudflared tunnel --url http://localhost:3000`. Over plain Wi-Fi it runs as a normal web page.

Faculty tools are still the Streamlit page (`streamlit run ui/app.py` → Faculty).
