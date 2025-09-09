import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv


BASE_DIR = Path(__file__).parent.resolve()
load_dotenv(BASE_DIR / ".env")

# Resolve paths from env (supports absolute or relative-to-project)
def _resolve_dir(env_key: str, default_rel: str) -> Path:
    val = os.getenv(env_key, default_rel)
    p = Path(val)
    if not p.is_absolute():
        p = BASE_DIR / p
    return p.resolve()

RUNS_DIR = _resolve_dir("RUNS_DIR", "runs")
LOCUSTFILES_DIR = _resolve_dir("LOCUSTFILES_DIR", "locustfiles")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
LOCUSTFILES_DIR.mkdir(parents=True, exist_ok=True)


from typing import Optional


def which_locust() -> Optional[str]:
    return shutil.which("locust")


def list_locustfiles() -> list[Path]:
    return sorted(LOCUSTFILES_DIR.glob("**/*.py"))


def ensure_sample_locustfile():
    sample = LOCUSTFILES_DIR / "sample_locustfile.py"
    if not sample.exists():
        sample.write_text(
            """
from locust import HttpUser, task, between


class WebsiteUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def index(self):
        self.client.get("/")

    @task
    def about(self):
        self.client.get("/about")
""".strip()
        )


def run_locust(
    locustfile: Path,
    host: Optional[str],
    users: int,
    spawn_rate: float,
    run_time: str,
    run_dir: Path,
    csv_prefix: str = "stats",
    html_report: bool = True,
    csv_full_history: bool = True,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile = run_dir / "locust.log"
    html_path = run_dir / "report.html"
    csv_prefix_path = run_dir / csv_prefix

    cmd = [
        "locust",
        "-f",
        str(locustfile),
        "--headless",
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "--run-time",
        str(run_time),
        "--csv",
        str(csv_prefix_path),
        "--logfile",
        str(logfile),
        "--only-summary",
    ]
    if host:
        cmd += ["--host", str(host)]
    if html_report:
        cmd += ["--html", str(html_path)]
    if csv_full_history:
        cmd += ["--csv-full-history"]

    start = datetime.utcnow().isoformat()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
    )
    return proc, logfile, html_path, start


def load_stats(run_dir: Path, prefix: str = "stats"):
    files = {
        "stats": run_dir / f"{prefix}_stats.csv",
        "history": run_dir / f"{prefix}_stats_history.csv",
        "failures": run_dir / f"{prefix}_failures.csv",
        "requests": run_dir / f"{prefix}_requests.csv",
        "exceptions": run_dir / f"{prefix}_exceptions.csv",
        "distribution": run_dir / f"{prefix}_distribution.csv",
    }
    data = {}
    for k, p in files.items():
        if p.exists():
            try:
                data[k] = pd.read_csv(p)
            except Exception:
                pass
    return data


def render_summary_from_stats(stats_df: pd.DataFrame):
    # Locust stats CSV has an "Aggregated" row with overall metrics
    agg = None
    if "Name" in stats_df.columns:
        agg_rows = stats_df[stats_df["Name"].astype(str).str.lower() == "aggregated"]
        if not agg_rows.empty:
            agg = agg_rows.iloc[0]

    cols = st.columns(4)
    if agg is not None:
        cols[0].metric("Requests", f"{int(agg.get('Request Count', 0))}")
        cols[1].metric("Failures", f"{int(agg.get('Failure Count', 0))}")
        cols[2].metric("Median (ms)", f"{int(agg.get('50%ile', agg.get('Median Response Time', 0)))}")
        p95 = agg.get("95%ile", agg.get("95%", None))
        if pd.notna(p95):
            cols[3].metric("p95 (ms)", f"{int(p95)}")
    else:
        cols[0].metric("Requests", "-")
        cols[1].metric("Failures", "-")
        cols[2].metric("Median (ms)", "-")
        cols[3].metric("p95 (ms)", "-")


def render_time_series(history_df: pd.DataFrame):
    # history_df has columns like Timestamp, Requests/s, Fails/s, 50%, 95%, Users
    if history_df is None or history_df.empty:
        st.info("Zaman serisi verisi bulunamadı.")
        return
    df = history_df.copy()
    # Normalize timestamp
    ts_col = None
    for c in ["Timestamp", "Time", "timestamp", "time"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        st.info("Zaman sütunu bulunamadı.")
        return
    df[ts_col] = pd.to_datetime(df[ts_col])

    y_cols = [c for c in df.columns if c in ["Requests/s", "Fails/s", "Failures/s", "Users", "User Count"]]
    chart1 = px.line(df, x=ts_col, y=y_cols, title="RPS / Hata / Kullanıcı")
    st.plotly_chart(chart1, use_container_width=True)

    p_cols = [c for c in ["50%", "95%", "99%", "99.9%", "Median Response Time", "95%ile"] if c in df.columns]
    if p_cols:
        chart2 = px.line(df, x=ts_col, y=p_cols, title="Gecikme Yüzdelikleri (ms)")
        st.plotly_chart(chart2, use_container_width=True)


def list_runs() -> list[Path]:
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)


def locustfile_declares_host(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    # naive check for `host = "..."` pattern
    return any(line.strip().startswith("host =") for line in text.splitlines())


def main():
    st.set_page_config(page_title="Locust Web - Streamlit", layout="wide")
    st.title("Locust Yük Testleri - Streamlit Arayüz")

    ensure_sample_locustfile()

    tabs = st.tabs(["Test Çalıştır", "Raporları Gör", "Geçmiş Çalışmalar", "Kurulum"])

    with tabs[0]:
        st.subheader("Locust Testini Çalıştır")

        if which_locust() is None:
            st.error("'locust' komutu bulunamadı. Lütfen 'pip install -r requirements.txt' ile bağımlılıkları kurun.")

        locustfiles = list_locustfiles()
        choices = [str(p.relative_to(BASE_DIR)) for p in locustfiles]
        selected_file = st.selectbox("Locust dosyası", options=choices, index=0 if choices else None)

        # Defaults from .env
        default_host = os.getenv("LOCUST_HOST", "http://localhost:8000")
        default_users = int(os.getenv("LOCUST_USERS", 10))
        default_spawn = float(os.getenv("LOCUST_SPAWN_RATE", 2))
        default_run_time = os.getenv("LOCUST_RUN_TIME", "1m")
        default_csv_prefix = os.getenv("LOCUST_CSV_PREFIX", "stats")
        def _parse_bool(key: str, default: bool) -> bool:
            raw = os.getenv(key)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        default_html_report = _parse_bool("LOCUST_HTML_REPORT", True)
        default_csv_full_history = _parse_bool("LOCUST_CSV_FULL_HISTORY", True)

        selected_path = BASE_DIR / selected_file if selected_file else None
        has_file_host = bool(selected_path and locustfile_declares_host(selected_path))
        use_file_host = st.checkbox("Host'u locustfile içinden kullan", value=has_file_host)

        host = st.text_input(
            "Host (örn: https://example.com)",
            value=default_host,
            disabled=use_file_host,
            help="Boş bırakılırsa ve üstteki seçenek açıksa locustfile'daki host kullanılır."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            users = st.number_input("Kullanıcı sayısı (-u)", min_value=1, value=default_users)
        with col2:
            spawn = st.number_input("Başlatma hızı/s (-r)", min_value=1, value=int(default_spawn))
        with col3:
            run_time = st.text_input("Çalışma süresi (--run-time)", value=default_run_time)

        adv = st.expander("Gelişmiş Ayarlar", expanded=False)
        with adv:
            html_report = st.checkbox("HTML raporu üret", value=default_html_report)
            csv_full_history = st.checkbox("CSV full history", value=default_csv_full_history)
            csv_prefix = st.text_input("CSV prefix", value=default_csv_prefix)

        run_btn = st.button(
            "Testi Başlat",
            type="primary",
            use_container_width=True,
            disabled=(which_locust() is None or not selected_file or (not use_file_host and not host)),
        )

        log_area = st.empty()
        status_area = st.empty()

        if run_btn:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            st.session_state["current_run_dir"] = str(run_dir)
            st.session_state["running"] = True

            locustfile_path = BASE_DIR / selected_file

            proc, logfile, html_path, start = run_locust(
                locustfile=locustfile_path,
                host=None if use_file_host else host,
                users=int(users),
                spawn_rate=float(spawn),
                run_time=run_time,
                run_dir=run_dir,
                csv_prefix=csv_prefix,
                html_report=html_report,
                csv_full_history=csv_full_history,
            )

            log_lines = []
            with st.spinner("Locust çalışıyor... Lütfen bekleyin"):
                if proc.stdout is not None:
                    for line in proc.stdout:
                        log_lines.append(line.rstrip())
                        # Show last ~200 lines to keep UI snappy
                        to_show = "\n".join(log_lines[-200:])
                        log_area.code(to_show)
                proc.wait()

            st.session_state["running"] = False
            status_area.success(f"Tamamlandı. Çalışma klasörü: {run_dir}")

    with tabs[1]:
        st.subheader("Raporları Gör")
        runs = list_runs()
        if not runs:
            st.info("Henüz bir çalışma yok. 'Test Çalıştır' sekmesinden başlatın.")
        else:
            run_opts = [str(p.relative_to(BASE_DIR)) for p in runs]
            sel = st.selectbox("Çalışma seçin", run_opts)
            selected_run = BASE_DIR / sel
            data = load_stats(selected_run)

            if "stats" in data and not data["stats"].empty:
                render_summary_from_stats(data["stats"])
                st.divider()
                st.caption("Ayrıntılı istek istatistikleri")
                st.dataframe(data["stats"], use_container_width=True, height=300)

            if "history" in data and not data["history"].empty:
                st.divider()
                render_time_series(data["history"])

            html_path = selected_run / "report.html"
            if html_path.exists():
                st.divider()
                st.caption("HTML Rapor (gömülü)")
                try:
                    html = html_path.read_text(encoding="utf-8")
                    components.html(html, height=700, scrolling=True)
                except Exception:
                    st.write(f"HTML raporu: {html_path}")
            else:
                st.info("HTML raporu bulunamadı. 'HTML raporu üret' seçeneğini etkinleştirin.")

    with tabs[2]:
        st.subheader("Geçmiş Çalışmalar")
        runs = list_runs()
        if not runs:
            st.info("Kayıtlı çalışma yok.")
        else:
            for r in runs:
                any_stats = any(r.glob("*_stats.csv"))
                html_exists = (r / "report.html").exists()
                st.write(f"- {r.name}  | CSV: {'✅' if any_stats else '❌'} | HTML: {'✅' if html_exists else '❌'}  | Klasör: {r}")

    with tabs[3]:
        st.subheader("Kurulum ve Kullanım")
        st.markdown(
            """
            - Gerekli paketleri kurun: `pip install -r requirements.txt`
            - Uygulamayı başlatın: `streamlit run app.py`
            - Sol tarafta Locust dosyasını ve parametreleri seçin, testi başlatın.
            - Çalışma çıktıları `runs/` klasöründe saklanır.
            - Örnek Locust dosyası: `locustfiles/sample_locustfile.py`
            """
        )


if __name__ == "__main__":
    main()
