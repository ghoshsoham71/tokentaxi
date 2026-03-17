# tokentaxi /_dashboard.py
"""
Local Streamlit dashboard for tokentaxi .

Launch via:
  tokentaxi  dashboard --config router.yaml

Or directly:
  streamlit run tokentaxi /_dashboard.py -- --config router.yaml

Shows live provider headroom bars, requests routed, fallbacks triggered,
and average latency. Auto-refreshes every 3 seconds.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

try:
    import streamlit as st
    import plotly.graph_objects as go  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    print(
        "Dashboard dependencies missing. Install with: pip install 'tokentaxi [dashboard]'",
        file=sys.stderr,
    )
    sys.exit(1)

from tokentaxi import LLMRouter


@st.cache_resource
def _get_router(config_path: str | None = None) -> LLMRouter:
    """Persist the router instance across reruns."""
    if config_path:
        return LLMRouter.from_yaml(config_path)
    return LLMRouter.from_env()

async def _get_status(router: LLMRouter) -> dict:
    return await router.status()

def render_dashboard() -> None:
    st.set_page_config(page_title="tokentaxi Dashboard", page_icon="🚖", layout="wide")
    st.title("🚖 tokentaxi — Live Infrastructure Status")

    # Handle arguments passed via CLI
    config_path = None
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            config_path = sys.argv[idx + 1]

    try:
        router = _get_router(config_path)
    except Exception as e:
        st.error(f"Failed to initialize router: {e}")
        return

    # Create a persistent placeholder for the UI
    placeholder = st.empty()

    # Get status via asyncio
    try:
        status = asyncio.run(_get_status(router))
    except Exception as e:
        st.error(f"Failed to fetch status: {e}")
        return

    if not status:
        st.warning("No providers registered. Use `router.register()` or a config file.")
        if st.button("Retry"):
            st.rerun()
        return

    with placeholder.container():
        cols = st.columns(len(status))
        for col, (name, info) in zip(cols, status.items()):
            with col:
                headroom = info["headroom_pct"]
                circuit = info["circuit_open"]
                latency = info["avg_latency_ms"]

                color = "red" if circuit or headroom < 10 else "green" if headroom > 50 else "orange"
                st.subheader(f"{name.upper()}")
                
                st.markdown(f"**Circuit:** {'🔴 OPEN' if circuit else '🟢 CLOSED'}")

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=headroom,
                        title={"text": "Headroom %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 20], "color": "#ffcccc"},
                                {"range": [20, 60], "color": "#fff3cc"},
                                {"range": [60, 100], "color": "#ccffcc"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=180, margin=dict(t=30, b=0, l=10, r=10))
                st.plotly_chart(fig, use_container_width=True)

                c1, c2 = st.columns(2)
                c1.metric("RPM", f"{info['rpm_used']}/{info['rpm_limit']}")
                c2.metric("Latency", f"{latency}ms")
                
                # TPM with formatting
                st.metric("TPM Used", f"{info['tpm_used']:,} / {info['tpm_limit']:,}")

        st.divider()
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')} (Auto-refreshes every 5s)")

    # Auto-refresh using Streamlit's native rerun (every 5 seconds)
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    render_dashboard()
