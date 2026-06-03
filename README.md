# Nevsky Avenue Traffic Intelligence

A Streamlit BI dashboard for pedestrian and vehicle analytics on Nevsky Avenue, Saint Petersburg.

- **Live** — real-time YOLO detection + line-crossing counter over a video/RTSP source.
- **Monthly BI** — daily pedestrian/vehicle summaries with charts.
- **AI Chat** — a Groq-powered (`llama-3.3-70b-versatile`) assistant that answers questions
  about the traffic data using tool calls against the SQLite database.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your real key in .env
```

Set your Groq API key (get one at https://console.groq.com):

```
GROQ_API_KEY=your_key_here
```

## Run

```bash
streamlit run app/dashboard.py
```

## Notes

- The trained detector model is at
  `runs/detect/people_vehicle_detector-5/weights/best.pt`.
- The **Live** tab needs a local CPU/GPU and a video source (`video.mp4` or an RTSP URL)
  and the `streamlit-webrtc` stack; it is meant to run locally, not on a lightweight cloud host.
- The **AI Chat** and **Monthly BI** tabs work anywhere (they only need the SQLite DB and the Groq API).
- `traffic.db` ships with sample data so the chat and BI tabs work out of the box.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. On https://share.streamlit.io create an app pointing at `app/dashboard.py`.
3. In the app's **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. The Chat and Monthly BI tabs will work. The Live (webrtc + YOLO) tab is best run locally.
