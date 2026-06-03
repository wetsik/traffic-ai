from __future__ import annotations

import sys
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import cv2
import plotly.graph_objects as go
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO

from agent.agent import TrafficAnalyticsAgent
from config import load_dotenv_file
from db.database import get_daily_summary, get_last_3h, insert_detection
from tracking import LineCrossCounter, draw_tracking_overlay, extract_tracked_observations


def _svg(body: str, size: int = 24, stroke_width: float = 2.0) -> str:
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' "
        f"viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='{stroke_width}' "
        f"stroke-linecap='round' stroke-linejoin='round'>{body}</svg>"
    )


# Lucide icons (MIT licensed) — https://lucide.dev
ICON_ACTIVITY = "<path d='M22 12h-4l-3 9L9 3l-3 9H2'/>"
ICON_MESSAGE = "<path d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'/>"
ICON_BARCHART = "<path d='M3 3v18h18'/><path d='M18 17V9'/><path d='M13 17V5'/><path d='M8 17v-3'/>"
ICON_USER = "<path d='M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2'/><circle cx='12' cy='7' r='4'/>"


def _render_message(role: str, content: str) -> None:
    import html as _html

    safe = _html.escape(content).replace('\n', '<br>')
    icon = ICON_USER if role == 'user' else ICON_BARCHART
    st.markdown(
        f"<div class='chat-row {role}'>"
        f"<div class='chat-avatar'>{_svg(icon, size=18)}</div>"
        f"<div class='chat-bubble'>{safe}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SOURCE_PATH = PROJECT_ROOT / 'video.mp4'
YOLO_MODEL_PATH = PROJECT_ROOT / 'runs' / 'detect' / 'people_vehicle_detector-5' / 'weights' / 'best.pt'
TRACKER_CONFIG_PATH = PROJECT_ROOT / 'bytetrack_custom.yaml'
# Confidence floor kept at 0.4 for a clean picture (no spurious detections).
# Crowd robustness comes from the tracker's large track_buffer (see
# bytetrack_custom.yaml), which re-attaches the same id after occlusion without
# the false positives a lower floor would introduce. Smaller imgsz keeps
# inference fast so fewer frames are dropped.
LIVE_CONFIDENCE = 0.4
LIVE_IMGSZ = 768
DB_WRITE_INTERVAL_SECONDS = 10.0


st.set_page_config(
    page_title='Nevsky Avenue BI',
    page_icon='🚦',
    layout='wide',
    initial_sidebar_state='expanded',
)
load_dotenv_file()


DARK_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-0: #070b16;
        --bg-1: #0d1424;
        --surface: rgba(20, 28, 46, 0.72);
        --surface-2: rgba(30, 41, 64, 0.6);
        --border: rgba(148, 163, 184, 0.14);
        --text: #e6eaf2;
        --muted: #93a0b8;
        --accent: #6366f1;
        --accent-2: #3b82f6;
    }

    html, body, [class*="css"], .stApp, button, input, textarea {
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 80% -10%, rgba(99, 102, 241, 0.12), transparent 60%),
            radial-gradient(900px 500px at 0% 0%, rgba(59, 130, 246, 0.10), transparent 55%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
        color: var(--text);
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div { color: var(--text); }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    /* hide Streamlit chrome for a cleaner app shell */
    [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
    [data-testid="stHeader"] { background: transparent; }

    /* ===== App header ===== */
    .app-header { display: flex; align-items: center; gap: 0.85rem; margin: 0.2rem 0 0.3rem; }
    .app-logo {
        display: grid; place-items: center; width: 46px; height: 46px; border-radius: 13px;
        color: #fff; background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35);
    }
    .app-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15; }
    .app-sub { color: var(--muted) !important; font-size: 0.9rem; margin-top: 1px; }

    /* ===== Tabs ===== */
    [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid var(--border); }
    [data-baseweb="tab"] {
        background: transparent; border-radius: 10px 10px 0 0; padding: 0.35rem 0.9rem;
        color: var(--muted) !important; font-weight: 500;
    }
    [data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
    [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 3px; border-radius: 3px; }

    /* ===== Cards / metrics ===== */
    .panel {
        background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
        padding: 1.1rem 1.25rem; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    [data-testid='stMetric'] {
        background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
        padding: 0.9rem 1.1rem; box-shadow: 0 12px 36px rgba(0, 0, 0, 0.25);
    }
    [data-testid='stMetricValue'] { color: #f8fafc !important; font-weight: 700; }
    [data-testid='stMetricLabel'] { color: var(--muted) !important; }
    .subtitle { color: var(--muted) !important; font-size: 0.95rem; }

    /* ===== Buttons ===== */
    .stButton > button {
        border-radius: 12px; border: 1px solid var(--border); background: var(--surface-2);
        color: var(--text); font-weight: 500; transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent); background: rgba(99, 102, 241, 0.14);
        transform: translateY(-1px);
    }

    /* ===== Inputs ===== */
    [data-baseweb="input"], [data-baseweb="select"] > div {
        border-radius: 12px !important; background: var(--surface-2) !important;
        border-color: var(--border) !important;
    }
    [data-testid="stTextInput"] input, [data-testid="stDateInput"] input { color: var(--text) !important; }

    /* ===== Chat header / empty state ===== */
    .chat-header { display: flex; align-items: center; gap: 0.55rem; font-size: 1.2rem; font-weight: 600; }
    .chat-header svg { color: var(--accent-2); }
    .chat-empty { text-align: center; padding: 2.4rem 0 1.2rem; }
    .chat-empty-icon {
        display: inline-grid; place-items: center; width: 58px; height: 58px; border-radius: 16px;
        margin-bottom: 0.6rem; color: #fff;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        box-shadow: 0 12px 34px rgba(99, 102, 241, 0.35);
    }
    .chat-empty h3 { margin: 0.4rem 0 0.25rem; font-weight: 600; font-size: 1.2rem; }
    .chat-empty p { color: var(--muted) !important; font-size: 0.95rem; }

    /* ===== Chat bubbles (custom HTML, ChatGPT / Claude style) ===== */
    .chat-row { display: flex; align-items: flex-start; gap: 0.6rem; margin-bottom: 0.7rem; }
    .chat-row.user { flex-direction: row-reverse; }
    .chat-avatar {
        flex: 0 0 34px; width: 34px; height: 34px; border-radius: 10px;
        display: grid; place-items: center; color: #fff;
    }
    .chat-row.assistant .chat-avatar { background: linear-gradient(135deg, #475569, #1e293b); }
    .chat-row.user .chat-avatar { background: linear-gradient(135deg, var(--accent), var(--accent-2)); }
    .chat-bubble {
        max-width: 70%; padding: 0.6rem 0.95rem; border-radius: 16px; line-height: 1.55;
        word-wrap: break-word;
    }
    .chat-row.assistant .chat-bubble { background: var(--surface); border: 1px solid var(--border); }
    .chat-row.user .chat-bubble {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff;
    }
    .chat-row.user .chat-bubble, .chat-row.user .chat-bubble * { color: #fff !important; }

    /* chat input */
    [data-testid="stChatInput"] {
        border-radius: 16px; border: 1px solid var(--border); background: var(--surface-2);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--accent); box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.28);
    }
</style>
"""


st.markdown(DARK_STYLE, unsafe_allow_html=True)


st.sidebar.title('Nevsky Avenue BI')
st.sidebar.caption('Pedestrian and vehicle traffic analytics for Nevsky Avenue.')


def _to_int(value: Any) -> int:
    return int(value or 0)


def _format_number(value: int) -> str:
    return f'{value:,}'


def _ensure_agent() -> TrafficAnalyticsAgent | None:
    if 'traffic_agent' not in st.session_state:
        st.session_state.traffic_agent = TrafficAnalyticsAgent()
    return st.session_state.traffic_agent


def _resolve_source() -> str:
    source = str(st.session_state.get('video_source', str(VIDEO_SOURCE_PATH))).strip()
    return source or str(VIDEO_SOURCE_PATH)


@st.cache_resource(show_spinner=False)
def _load_live_model() -> YOLO:
    return YOLO(str(YOLO_MODEL_PATH))


LIVE_LOCK = threading.Lock()
LIVE_STATE: dict[str, Any] = {
    'counter': LineCrossCounter(),
    'latest_totals': {'pedestrians': 0, 'cars': 0, 'vehicles': 0},
    'latest_passed': {'pedestrians': 0, 'cars': 0, 'vehicles': 0},
    'last_db_write': 0.0,
}


class YOLOProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.model = _load_live_model()
        self.frame_index = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = cv2.resize(frame.to_ndarray(format='bgr24'), (1920, 1080))
            results = self.model.track(
                image,
                conf=LIVE_CONFIDENCE,
                persist=True,
                tracker=str(TRACKER_CONFIG_PATH),
                verbose=False,
                imgsz=LIVE_IMGSZ,
                workers=0,
            )
            observations = extract_tracked_observations(self.model, results[0])
            observations = list(observations)

            with LIVE_LOCK:
                counter: LineCrossCounter = LIVE_STATE['counter']
                passed_now = counter.update(observations, image.shape[1], image.shape[0])
                totals = counter.totals()
                LIVE_STATE['latest_passed'] = dict(passed_now)
                LIVE_STATE['latest_totals'] = dict(totals)
                now = time.monotonic()
                if now - LIVE_STATE['last_db_write'] >= DB_WRITE_INTERVAL_SECONDS:
                    pending = counter.drain_pending()
                    if any(pending.values()):
                        try:
                            insert_detection(
                                pedestrians=int(pending.get('person', 0)),
                                cars=int(pending.get('vehicle', 0)),
                                vehicles=0,
                            )
                        except:
                            pass
                    LIVE_STATE['last_db_write'] = now

            annotated = image.copy()
            
            annotated = draw_tracking_overlay(
                annotated,
                observations,
                totals,
                passed_now,
                counter,
            )
            self.frame_index += 1
            return av.VideoFrame.from_ndarray(annotated, format='bgr24')
        except Exception as e:
            return av.VideoFrame.from_ndarray(image, format='bgr24')


def _render_live_tab() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader('Live')
    st.caption('Live detector video with YOLO annotations.')

    st.text_input(
        'Video source',
        value=_resolve_source(),
        key='video_source',
        help='Use a local file path like video.mp4 or an RTSP URL.',
    )

    source = _resolve_source()
    player_factory = (lambda: MediaPlayer(source, loop=True)) if Path(source).exists() else (lambda: MediaPlayer(source))

    webrtc_ctx = webrtc_streamer(
        key=f'yolo-{source}',
        mode=WebRtcMode.RECVONLY,
        player_factory=player_factory,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={'video': True, 'audio': False},
    )

    status = 'Detector running' if webrtc_ctx.state.playing else 'Detector stopped'
    st.info(status)

    rows = get_last_3h()
    st.markdown('#### Last 3 hours')
    if rows:
        times = [datetime.fromisoformat(str(row['timestamp'])) for row in rows]
        pedestrians = [_to_int(row['pedestrians']) for row in rows]
        cars = [_to_int(row['cars']) for row in rows]
        vehicles = [_to_int(row['vehicles']) for row in rows]
        combined_vehicles = [cars[idx] + vehicles[idx] for idx in range(len(rows))]

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=times, y=pedestrians, mode='lines', name='Pedestrians', line=dict(color='#38bdf8', width=3))
        )
        figure.add_trace(
            go.Scatter(x=times, y=combined_vehicles, mode='lines', name='Vehicles', line=dict(color='#f59e0b', width=3))
        )
        figure.update_layout(
            template='plotly_dark',
            height=320,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Time',
            yaxis_title='Count',
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info('No raw detections found for the last 3 hours yet.')

    st.markdown('</div>', unsafe_allow_html=True)


def _render_monthly_tab() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader('Monthly BI')

    rows = get_daily_summary(30)
    if rows:
        first_day = date.fromisoformat(str(rows[0]['date']))
        last_day = date.fromisoformat(str(rows[-1]['date']))
    else:
        first_day = date.today() - timedelta(days=29)
        last_day = date.today()

    selected = st.date_input(
        'Date range',
        value=(first_day, last_day),
        min_value=first_day,
        max_value=last_day,
    )
    if isinstance(selected, tuple) and len(selected) == 2:
        start_date, end_date = selected
    else:
        start_date, end_date = first_day, last_day

    filtered_rows = [
        row
        for row in rows
        if start_date <= date.fromisoformat(str(row['date'])) <= end_date
    ]

    if filtered_rows:
        x_values = [row['date'] for row in filtered_rows]
        figure = go.Figure()
        figure.add_trace(
            go.Bar(x=x_values, y=[_to_int(row['pedestrians']) for row in filtered_rows], name='Pedestrians')
        )
        figure.add_trace(go.Bar(x=x_values, y=[_to_int(row['cars']) for row in filtered_rows], name='Cars'))
        figure.add_trace(
            go.Bar(x=x_values, y=[_to_int(row['vehicles']) for row in filtered_rows], name='Vehicle class')
        )
        figure.update_layout(
            template='plotly_dark',
            barmode='group',
            height=460,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title='Date',
            yaxis_title='Count',
        )
        st.plotly_chart(figure, use_container_width=True)

        totals = st.columns(3)
        totals[0].metric('Pedestrians total', _format_number(sum(_to_int(row['pedestrians']) for row in filtered_rows)))
        totals[1].metric('Cars total', _format_number(sum(_to_int(row['cars']) for row in filtered_rows)))
        totals[2].metric('Vehicle class total', _format_number(sum(_to_int(row['vehicles']) for row in filtered_rows)))
    else:
        st.info('No summary data found for the selected date range.')

    st.markdown('</div>', unsafe_allow_html=True)


CHAT_SUGGESTIONS = [
    'How many people passed today?',
    'How many vehicles passed today?',
    'What were the busiest hours recently?',
    'Compare today with yesterday',
]


def _render_chat_tab() -> None:
    # display_history holds only what the user should see (their questions and the
    # assistant's final answers). agent_history holds the full model protocol,
    # including tool calls and tool results, which must stay hidden from the UI.
    if 'display_history' not in st.session_state:
        st.session_state.display_history = []
    if 'agent_history' not in st.session_state:
        st.session_state.agent_history = []

    agent = _ensure_agent()

    header = st.columns([1, 0.16])
    with header[0]:
        st.markdown(
            f"<div class='chat-header'>{_svg(ICON_MESSAGE, size=22)}<span>Traffic Assistant</span></div>",
            unsafe_allow_html=True,
        )
    with header[1]:
        if st.button('Clear', use_container_width=True, disabled=not st.session_state.display_history):
            st.session_state.display_history = []
            st.session_state.agent_history = []
            if agent is not None:
                agent.history = []
            st.rerun()

    pending: str | None = None

    # The messages container is created BEFORE st.chat_input so its content always
    # renders above the input box (which sits inline at the bottom of the tab).
    messages = st.container()
    with messages:
        if st.session_state.display_history:
            for message in st.session_state.display_history:
                _render_message(message['role'], message['content'])
        else:
            # Welcome screen with quick-start suggestions (like ChatGPT / Claude).
            st.markdown(
                "<div class='chat-empty'>"
                f"<div class='chat-empty-icon'>{_svg(ICON_BARCHART, size=30)}</div>"
                "<h3>How can I help with Nevsky Avenue traffic?</h3>"
                "<p>Ask about today's pedestrians and vehicles, busiest hours, or trends.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            suggestion_cols = st.columns(2)
            for index, suggestion in enumerate(CHAT_SUGGESTIONS):
                if suggestion_cols[index % 2].button(suggestion, key=f'suggestion_{index}', use_container_width=True):
                    pending = suggestion

    typed = st.chat_input('Message Traffic Assistant…')
    user_input = (typed or pending or '').strip()
    if not user_input:
        return

    if agent is None or agent.client is None:
        st.error('Set GROQ_API_KEY to enable the chat agent.')
        return

    # Render the new exchange into the messages container (above the input), then
    # persist it and rerun so the conversation re-renders cleanly from history.
    st.session_state.display_history.append({'role': 'user', 'content': user_input})
    agent.history = list(st.session_state.agent_history)
    with messages:
        _render_message('user', user_input)
        with st.spinner('Thinking…'):
            try:
                response = agent.respond(user_input)
            except Exception as e:
                response = f'Error: {str(e)[:200]}'
    st.session_state.agent_history = list(agent.history)
    st.session_state.display_history.append({'role': 'assistant', 'content': response})
    st.rerun()


st.markdown(
    "<div class='app-header'>"
    f"<div class='app-logo'>{_svg(ICON_ACTIVITY, size=26)}</div>"
    "<div>"
    "<div class='app-title'>Nevsky Avenue Traffic Intelligence</div>"
    "<div class='app-sub'>Pedestrian &amp; vehicle activity · 3-hour summaries · Groq-powered assistant</div>"
    "</div></div>",
    unsafe_allow_html=True,
)


live_tab, monthly_tab, chat_tab = st.tabs(['Live', 'Monthly BI', 'AI Chat'])
with live_tab:
    _render_live_tab()
with monthly_tab:
    _render_monthly_tab()
with chat_tab:
    _render_chat_tab()


def main() -> None:
    pass


if __name__ == '__main__':
    main()
