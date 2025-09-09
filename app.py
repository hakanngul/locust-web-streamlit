import os
import sys
import shutil
import subprocess
from datetime import datetime
import time
from pathlib import Path
import json
import re

import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from dotenv import load_dotenv
import warnings
from typing import Optional



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
PROFILES_DIR = BASE_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
SCHEDULES_DIR = BASE_DIR / "schedules"
SCHEDULES_DIR.mkdir(parents=True, exist_ok=True)


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str) -> Optional[int]:
    raw = os.getenv(key)
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _env_float(key: str) -> Optional[float]:
    raw = os.getenv(key)
    if not raw:
        return None
    try:
        return float(raw)
    except Exception:
        return None


# Simple parsers for matrix inputs
def _parse_list_int(text: str) -> list[int]:
    vals = []
    for part in (text or "").replace("\n", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(int(p))
        except Exception:
            pass
    return vals


def _parse_list_float(text: str) -> list[float]:
    vals = []
    for part in (text or "").replace("\n", ",").split(","):
        p = part.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals


def _parse_list_str(text: str) -> list[str]:
    vals = []
    for part in (text or "").replace("\n", ",").split(","):
        p = part.strip()
        if p:
            vals.append(p)
    return vals


# Optionally suppress urllib3 OpenSSL warnings (macOS LibreSSL noise)
if _env_bool("SUPPRESS_SSL_WARNINGS", False):
    try:
        from urllib3.exceptions import NotOpenSSLWarning

        warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    except Exception:
        pass


def which_locust() -> Optional[str]:
    return shutil.which("locust")


def list_locustfiles() -> list[Path]:
    return sorted(LOCUSTFILES_DIR.glob("**/*.py"))


# Profile helpers
def _safe_profile_name(name: str) -> str:
    # keep alnum, dash, underscore, dot
    return re.sub(r"[^A-Za-z0-9._-]", "_", name.strip())


def list_profiles() -> list[str]:
    return sorted([p.stem for p in PROFILES_DIR.glob("*.json")])


def load_profile(name: str) -> Optional[dict]:
    path = PROFILES_DIR / f"{_safe_profile_name(name)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_profile(name: str, data: dict) -> Path:
    path = PROFILES_DIR / f"{_safe_profile_name(name)}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def delete_profile(name: str) -> bool:
    path = PROFILES_DIR / f"{_safe_profile_name(name)}.json"
    try:
        if path.exists():
            path.unlink()
            return True
    except Exception:
        return False
    return False


# Schedules helpers
def _schedule_path(sid: str) -> Path:
    return SCHEDULES_DIR / f"{sid}.json"


def list_schedules() -> list[Path]:
    return sorted(SCHEDULES_DIR.glob("*.json"))


def load_schedule(sid: str) -> Optional[dict]:
    p = _schedule_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_schedule(data: dict) -> Path:
    sid = data.get("id") or datetime.utcnow().strftime("sch_%Y%m%d_%H%M%S")
    data["id"] = sid
    p = _schedule_path(sid)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def delete_schedule(sid: str) -> bool:
    p = _schedule_path(sid)
    try:
        if p.exists():
            p.unlink()
            return True
    except Exception:
        return False
    return False


def trigger_due_schedules():
    now = datetime.utcnow()
    for sp in list_schedules():
        try:
            sch = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not sch.get("enabled", True):
            continue
        stype = sch.get("type", "one_time")
        last_run = sch.get("last_run_at")
        # Check one-time
        if stype == "one_time":
            at = sch.get("at")
            if not at or sch.get("done"):
                continue
            try:
                at_dt = datetime.fromisoformat(at)
            except Exception:
                continue
            if now >= at_dt:
                _start_scheduled_run(sch)
        elif stype == "interval":
            minutes = int(sch.get("every_minutes", 0) or 0)
            if minutes <= 0:
                continue
            last_dt = None
            if last_run:
                try:
                    last_dt = datetime.fromisoformat(last_run)
                except Exception:
                    last_dt = None
            due = (last_dt is None) or ((now - last_dt).total_seconds() >= minutes * 60)
            if due:
                _start_scheduled_run(sch)


def _start_scheduled_run(sch: dict):
    # fire-and-forget scheduled run (no UI log streaming)
    try:
        locustfile_path = BASE_DIR / sch["locustfile"]
        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + (f"_{sch['id']}" if sch.get("id") else "")
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        proc, logfile, html_path, start, cmd = run_locust(
            locustfile=locustfile_path,
            host=(None if sch.get("use_file_host") else sch.get("host")),
            users=int(sch.get("users", 1)),
            spawn_rate=float(sch.get("spawn_rate", 1)),
            run_time=str(sch.get("run_time", "1m")),
            run_dir=run_dir,
            csv_prefix=str(sch.get("csv_prefix", "stats")),
            html_report=bool(sch.get("html_report", True)),
            csv_full_history=bool(sch.get("csv_full_history", True)),
            loglevel=str(sch.get("loglevel", "WARNING")),
            csv_flush_interval=None,
            stream_logs=False,
        )
        # detach: do not wait
        sch["last_run_at"] = start
        sch["last_pid"] = proc.pid if proc and proc.pid else None
        _schedule_path(sch["id"]).write_text(json.dumps(sch, indent=2), encoding="utf-8")
    except Exception:
        pass


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
    loglevel: str = "WARNING",
    csv_flush_interval: Optional[int] = None,
    stream_logs: bool = True,
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
    if loglevel:
        cmd += ["--loglevel", loglevel]
    if html_report:
        cmd += ["--html", str(html_path)]
    if csv_full_history:
        cmd += ["--csv-full-history"]
    if csv_flush_interval:
        cmd += ["--csv-flush-interval", str(int(csv_flush_interval))]

    start = datetime.utcnow().isoformat()

    proc = subprocess.Popen(
        cmd,
        stdout=(subprocess.PIPE if stream_logs else subprocess.DEVNULL),
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
    )
    return proc, logfile, html_path, start, cmd


def run_locust_distributed(
    locustfile: Path,
    host: Optional[str],
    users: int,
    spawn_rate: float,
    run_time: str,
    run_dir: Path,
    csv_prefix: str = "stats",
    html_report: bool = True,
    csv_full_history: bool = True,
    loglevel: str = "WARNING",
    csv_flush_interval: Optional[int] = None,
    stream_logs: bool = True,
    workers: int = 2,
    master_bind_host: str = "127.0.0.1",
    master_bind_port: int = 5557,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile_master = run_dir / "locust_master.log"
    html_path = run_dir / "report.html"
    csv_prefix_path = run_dir / csv_prefix

    # Master command
    master_cmd = [
        "locust",
        "-f", str(locustfile),
        "--headless",
        "--master",
        "--expect-workers", str(workers),
        "--master-bind-host", master_bind_host,
        "--master-bind-port", str(master_bind_port),
        "-u", str(users),
        "-r", str(spawn_rate),
        "--run-time", str(run_time),
        "--csv", str(csv_prefix_path),
        "--logfile", str(logfile_master),
        "--only-summary",
        "--loglevel", loglevel,
    ]
    if host:
        master_cmd += ["--host", str(host)]
    if html_report:
        master_cmd += ["--html", str(html_path)]
    if csv_full_history:
        master_cmd += ["--csv-full-history"]
    if csv_flush_interval:
        master_cmd += ["--csv-flush-interval", str(int(csv_flush_interval))]

    start = datetime.utcnow().isoformat()

    master_proc = subprocess.Popen(
        master_cmd,
        stdout=(subprocess.PIPE if stream_logs else subprocess.DEVNULL),
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(BASE_DIR),
    )

    # Workers
    worker_procs = []
    for i in range(workers):
        wlog = run_dir / f"locust_worker_{i+1}.log"
        worker_cmd = [
            "locust",
            "-f", str(locustfile),
            "--worker",
            "--master-host", master_bind_host,
            "--master-port", str(master_bind_port),
            "--logfile", str(wlog),
            "--loglevel", loglevel,
        ]
        p = subprocess.Popen(
            worker_cmd,
            stdout=(subprocess.PIPE if stream_logs else subprocess.DEVNULL),
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BASE_DIR),
        )
        worker_procs.append((p, wlog, worker_cmd))

    return master_proc, worker_procs, logfile_master, html_path, start, master_cmd


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


@st.cache_data(show_spinner=False)
def load_stats_cached(run_dir_str: str, prefix: str, sig: tuple):
    # Cached wrapper using a signature of file mtimes/sizes to invalidate
    return load_stats(Path(run_dir_str), prefix)


def run_signature(run_dir: Path) -> tuple:
    parts = []
    try:
        for p in sorted(run_dir.glob("*")):
            try:
                stt = p.stat()
                parts.append((p.name, stt.st_mtime_ns, stt.st_size))
            except Exception:
                continue
    except Exception:
        pass
    return tuple(parts)


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


def aggregated_metrics(stats_df: pd.DataFrame) -> dict:
    if stats_df is None or stats_df.empty:
        return {}
    name_col = "Name" if "Name" in stats_df.columns else ("name" if "name" in stats_df.columns else None)
    if not name_col:
        return {}
    rows = stats_df[stats_df[name_col].astype(str).str.lower() == "aggregated"]
    if rows.empty:
        return {}
    agg = rows.iloc[0]
    req = int(agg.get("Request Count", agg.get("Requests", 0)) or 0)
    fail = int(agg.get("Failure Count", agg.get("Failures", 0)) or 0)
    median = float(agg.get("50%ile", agg.get("Median Response Time", agg.get("50%", float('nan')))))
    p95 = float(agg.get("95%ile", agg.get("95%", float('nan'))))
    success_rate = (1 - (fail / req)) * 100 if req > 0 else float('nan')
    return {"requests": req, "failures": fail, "median_ms": median, "p95_ms": p95, "success_rate": success_rate}


def normalize_endpoint_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with standardized columns per endpoint, excluding the Aggregated row."""
    if stats_df is None or stats_df.empty:
        return pd.DataFrame(columns=["name", "requests", "failures", "median_ms", "p95_ms"])
    df = stats_df.copy()
    # Find name column
    name_col = "Name" if "Name" in df.columns else ("name" if "name" in df.columns else None)
    if not name_col:
        return pd.DataFrame(columns=["name", "requests", "failures", "median_ms", "p95_ms"])
    # Exclude Aggregated
    try:
        df = df[df[name_col].astype(str).str.lower() != "aggregated"]
    except Exception:
        pass
    # Map columns
    req_col = "Request Count" if "Request Count" in df.columns else ("Requests" if "Requests" in df.columns else None)
    fail_col = "Failure Count" if "Failure Count" in df.columns else ("Failures" if "Failures" in df.columns else None)
    median_col = None
    for c in ["50%ile", "Median Response Time", "50%", "Median"]:
        if c in df.columns:
            median_col = c
            break
    p95_col = None
    for c in ["95%ile", "95%", "Percentile 95%", "p95"]:
        if c in df.columns:
            p95_col = c
            break
    out = pd.DataFrame()
    out["name"] = df[name_col]
    if req_col:
        out["requests"] = pd.to_numeric(df[req_col], errors="coerce")
    else:
        out["requests"] = pd.NA
    if fail_col:
        out["failures"] = pd.to_numeric(df[fail_col], errors="coerce")
    else:
        out["failures"] = pd.NA
    if median_col:
        out["median_ms"] = pd.to_numeric(df[median_col], errors="coerce")
    else:
        out["median_ms"] = pd.NA
    if p95_col:
        out["p95_ms"] = pd.to_numeric(df[p95_col], errors="coerce")
    else:
        out["p95_ms"] = pd.NA
    # Success rate per endpoint
    try:
        out["success_rate"] = (1 - (out["failures"].astype(float) / out["requests"].astype(float))) * 100
    except Exception:
        out["success_rate"] = pd.NA
    return out


def list_runs() -> list[Path]:
    return sorted([p for p in RUNS_DIR.iterdir() if p.is_dir()], reverse=True)


def locustfile_declares_host(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    # naive check for `host = "..."` pattern
    return any(line.strip().startswith("host =") for line in text.splitlines())


def extract_locustfile_host(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r"^\s*host\s*=\s*['\"]([^'\"]+)['\"]", text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def main():
    st.set_page_config(page_title="Locust Web - Streamlit", layout="wide")
    st.title("Locust Yük Testleri - Streamlit Arayüz")

    ensure_sample_locustfile()

    tabs = st.tabs(["Test Çalıştır", "Raporları Gör", "Genel Dashboard", "Geçmiş Çalışmalar", "Kurulum"])

    with tabs[0]:
        st.subheader("Locust Testini Çalıştır")
        # Trigger scheduler on each view
        try:
            trigger_due_schedules()
        except Exception:
            pass

        if which_locust() is None:
            st.error("'locust' komutu bulunamadı. Lütfen 'pip install -r requirements.txt' ile bağımlılıkları kurun.")

        locustfiles = list_locustfiles()
        choices = [str(p.relative_to(BASE_DIR)) for p in locustfiles]
        selected_file = st.selectbox("Locust dosyası", options=choices, index=0 if choices else None, key="ui_selected_file")

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
        use_file_host = st.checkbox("Host'u locustfile içinden kullan", value=has_file_host, key="ui_use_file_host")

        file_host_val = extract_locustfile_host(selected_path) if selected_path else None

        host = st.text_input(
            "Host (örn: https://example.com)",
            value=default_host,
            disabled=use_file_host,
            help="Boş bırakılırsa ve üstteki seçenek açıksa locustfile'daki host kullanılır."
        , key="ui_host")

        effective_host = (file_host_val if use_file_host else host) or ""
        st.caption(f"Etkin host: {effective_host if effective_host else '—'}" + (f" (dosyadan)" if use_file_host and file_host_val else ""))
        col1, col2, col3 = st.columns(3)
        with col1:
            users = st.number_input("Kullanıcı sayısı (-u)", min_value=1, value=default_users, key="ui_users")
        with col2:
            spawn = st.number_input("Başlatma hızı/s (-r)", min_value=1, value=int(default_spawn), key="ui_spawn")
        with col3:
            run_time = st.text_input("Çalışma süresi (--run-time)", value=default_run_time, key="ui_run_time")

        adv = st.expander("Gelişmiş Ayarlar", expanded=False)
        with adv:
            html_report = st.checkbox("HTML raporu üret", value=default_html_report, key="ui_html_report")
            csv_full_history = st.checkbox("CSV full history", value=default_csv_full_history, key="ui_csv_full_history")
            csv_prefix = st.text_input("CSV prefix", value=default_csv_prefix, key="ui_csv_prefix")
            loglevel = st.selectbox("Log seviyesi", options=["ERROR", "WARNING", "INFO", "DEBUG"], index=1, help="Daha düşük seviye daha az çıktı ve daha hızlı UI", key="ui_loglevel")
            csv_flush_interval = st.number_input("CSV flush interval (s)", min_value=0, value=0, help="0=Locust varsayılanı. Büyük dosyalarda 5-10 sn faydalı olabilir.", key="ui_csv_flush_interval")
            stream_logs = st.checkbox("Canlı log akışı", value=True, help="Kapatırsanız performans artar, log dosyadan görülebilir.", key="ui_stream_logs")

        prof = st.expander("Profil (kaydet / yükle)", expanded=False)
        with prof:
            pcol1, pcol2 = st.columns([2, 1])
            with pcol1:
                existing = list_profiles()
                selected_profile = st.selectbox("Kayıtlı profiller", options=existing if existing else ["(profil yok)"] , index=0, key="ui_selected_profile")
            with pcol2:
                st.write(" ")
                st.write(" ")
                if existing:
                    if st.button("Profili Yükle", use_container_width=True):
                        pdata = load_profile(selected_profile)
                        if pdata:
                            # Push into session_state then rerun
                            for k, v in pdata.items():
                                st.session_state[f"ui_{k}"] = v
                            # selected locustfile key name mismatch
                            if "selected_file" in pdata:
                                st.session_state["ui_selected_file"] = pdata["selected_file"]
                            st.rerun()
                delcol = st.checkbox("Silmeyi onayla", value=False, key="ui_delete_confirm")
                if existing and st.button("Profili Sil", use_container_width=True, disabled=not st.session_state.get("ui_delete_confirm")):
                    delete_profile(selected_profile)
                    st.rerun()

            st.divider()
            name = st.text_input("Yeni profil adı", key="ui_profile_name")
            if st.button("Profili Kaydet", type="primary"):
                if not name.strip():
                    st.warning("Lütfen bir profil adı girin.")
                else:
                    pdata = {
                        "selected_file": st.session_state.get("ui_selected_file"),
                        "use_file_host": st.session_state.get("ui_use_file_host"),
                        "host": st.session_state.get("ui_host"),
                        "users": int(st.session_state.get("ui_users", 1)),
                        "spawn": int(st.session_state.get("ui_spawn", 1)),
                        "run_time": st.session_state.get("ui_run_time"),
                        "html_report": bool(st.session_state.get("ui_html_report", True)),
                        "csv_full_history": bool(st.session_state.get("ui_csv_full_history", True)),
                        "csv_prefix": st.session_state.get("ui_csv_prefix"),
                        "loglevel": st.session_state.get("ui_loglevel"),
                        "csv_flush_interval": int(st.session_state.get("ui_csv_flush_interval", 0)),
                        "stream_logs": bool(st.session_state.get("ui_stream_logs", True)),
                    }
                    save_profile(name, pdata)
                    st.success(f"Profil kaydedildi: {name}")

        # Matrix runner UI
        mx = st.expander("Parametre Taraması (Matrix)", expanded=False)
        with mx:
            st.caption("Virgülle ayırın. Örn: users=10,50 | spawn=2,5 | run-time=30s,1m")
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                mx_users_txt = st.text_input("Users listesi", value=f"{st.session_state.get('ui_users', default_users)}, {max(st.session_state.get('ui_users', default_users)*5, st.session_state.get('ui_users', default_users)+10)}")
            with mc2:
                base_spawn = int(st.session_state.get('ui_spawn', int(default_spawn)))
                mx_spawn_txt = st.text_input("Spawn listesi", value=f"{base_spawn}, {max(base_spawn*2, base_spawn+1)}")
            with mc3:
                mx_runtime_txt = st.text_input("Run-time listesi", value=f"{st.session_state.get('ui_run_time', default_run_time)}, 2m")

            mh1, mh2 = st.columns(2)
            with mh1:
                mx_use_file_host = st.checkbox("Host'u dosyadan al (matrix)", value=st.session_state.get('ui_use_file_host', use_file_host))
            with mh2:
                mx_hosts_txt = st.text_input("Host listesi (opsiyonel)", value=("" if mx_use_file_host else st.session_state.get('ui_host', host)))

            mopts1, mopts2 = st.columns(2)
            with mopts1:
                mx_html = st.checkbox("HTML raporu (matrix)", value=st.session_state.get('ui_html_report', html_report))
                mx_history = st.checkbox("CSV full history (matrix)", value=st.session_state.get('ui_csv_full_history', csv_full_history))
            with mopts2:
                mx_loglevel = st.selectbox("Log seviyesi (matrix)", options=["ERROR", "WARNING", "INFO"], index=1)
                mx_stream = st.checkbox("Canlı log akışı (matrix)", value=False)

            go_matrix = st.button("Matrix'i Çalıştır", type="primary")

            if go_matrix:
                # Build combinations
                users_list = _parse_list_int(mx_users_txt)
                spawn_list = _parse_list_int(mx_spawn_txt)
                rt_list = _parse_list_str(mx_runtime_txt)
                hosts_list = _parse_list_str(mx_hosts_txt) if not mx_use_file_host else [None]

                if not users_list or not spawn_list or not rt_list:
                    st.error("Users / Spawn / Run-time listeleri boş olamaz.")
                else:
                    import itertools
                    combos = list(itertools.product(users_list, spawn_list, rt_list, hosts_list))
                    MAX_COMBOS = 50
                    if len(combos) > MAX_COMBOS:
                        st.warning(f"Kombinasyon sayısı {len(combos)} > {MAX_COMBOS}. İlk {MAX_COMBOS} çalıştırılacak.")
                        combos = combos[:MAX_COMBOS]

                    progress = st.progress(0.0)
                    status = st.empty()
                    results = []
                    batch_id = datetime.utcnow().strftime("batch_%Y%m%d_%H%M%S")
                    for i, (uu, rr, rt, hh) in enumerate(combos, start=1):
                        status.info(f"Çalışıyor: users={uu}, spawn={rr}, rt={rt}, host={'dosya' if mx_use_file_host else (hh or host)}")
                        run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S") + f"_{i:02d}"
                        run_dir = RUNS_DIR / run_id
                        run_dir.mkdir(parents=True, exist_ok=True)

                        locustfile_path = BASE_DIR / selected_file
                        effective_host_matrix = None if mx_use_file_host else (hh or host)

                        proc, logfile, html_path, start, cmd = run_locust(
                            locustfile=locustfile_path,
                            host=effective_host_matrix,
                            users=int(uu),
                            spawn_rate=float(rr),
                            run_time=rt,
                            run_dir=run_dir,
                            csv_prefix=st.session_state.get("ui_csv_prefix", csv_prefix),
                            html_report=mx_html,
                            csv_full_history=mx_history,
                            loglevel=mx_loglevel,
                            csv_flush_interval=None,
                            stream_logs=mx_stream,
                        )
                        proc.wait()
                        ended = datetime.utcnow().isoformat()

                        # Save metadata
                        meta = {
                            "batch_id": batch_id,
                            "matrix_idx": i,
                            "locustfile": str(locustfile_path.relative_to(BASE_DIR)),
                            "use_file_host": bool(mx_use_file_host),
                            "typed_host": None if mx_use_file_host else (hh or host),
                            "effective_host": None if mx_use_file_host else (hh or host),
                            "users": int(uu),
                            "spawn_rate": float(rr),
                            "run_time": rt,
                            "csv_prefix": st.session_state.get("ui_csv_prefix", csv_prefix),
                            "html_report": bool(mx_html),
                            "csv_full_history": bool(mx_history),
                            "started_at": start,
                            "ended_at": ended,
                            "command": " ".join(cmd),
                        }
                        try:
                            (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                        except Exception:
                            pass

                        # Load summary
                        data_run = load_stats(run_dir, prefix=st.session_state.get("ui_csv_prefix", csv_prefix))
                        agg = aggregated_metrics(data_run.get("stats")) if data_run.get("stats") is not None else {}
                        results.append({
                            "run_id": run_id,
                            **meta,
                            **({k: agg.get(k) for k in ["requests", "failures", "success_rate", "median_ms", "p95_ms"]} if agg else {}),
                        })

                        progress.progress(i / len(combos))

                    status.success(f"Matrix tamamlandı. Toplam {len(results)} koşu.")
                    dfres = pd.DataFrame(results)
                    st.dataframe(dfres, use_container_width=True)

                    # Save batch summary
                    try:
                        (RUNS_DIR / f"{batch_id}_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                    except Exception:
                        pass

        dist = st.expander("Dağıtık Mod (Master/Worker)", expanded=False)
        with dist:
            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                dist_enabled = st.checkbox("Dağıtık modu kullan", value=False, key="ui_dist_enabled")
            with dc2:
                dist_workers = st.number_input("Worker sayısı", min_value=1, value=2, key="ui_dist_workers")
            with dc3:
                dist_stream = st.checkbox("Canlı log (master)", value=True, key="ui_dist_stream")

        run_btn = st.button(
            "Testi Başlat",
            type="primary",
            use_container_width=True,
            disabled=(which_locust() is None or not selected_file or (not use_file_host and not host)),
        )
        st.caption("Not: Zamanlayıcı etkinse, planlanan işleri bu sekme açıkken otomatik başlatır.")

        log_area = st.empty()
        status_area = st.empty()

        if run_btn:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            st.session_state["current_run_dir"] = str(run_dir)
            st.session_state["running"] = True

            locustfile_path = BASE_DIR / selected_file

            if st.session_state.get("ui_dist_enabled", False):
                # Distributed mode
                mproc, wprocs, logfile, html_path, start, cmd = run_locust_distributed(
                    locustfile=locustfile_path,
                    host=None if use_file_host else st.session_state.get("ui_host", host),
                    users=int(st.session_state.get("ui_users", users)),
                    spawn_rate=float(st.session_state.get("ui_spawn", spawn)),
                    run_time=st.session_state.get("ui_run_time", run_time),
                    run_dir=run_dir,
                    csv_prefix=st.session_state.get("ui_csv_prefix", csv_prefix),
                    html_report=bool(st.session_state.get("ui_html_report", html_report)),
                    csv_full_history=bool(st.session_state.get("ui_csv_full_history", csv_full_history)),
                    loglevel=st.session_state.get("ui_loglevel", loglevel),
                    csv_flush_interval=(int(st.session_state.get("ui_csv_flush_interval", csv_flush_interval)) or None),
                    stream_logs=bool(st.session_state.get("ui_dist_stream", True)),
                    workers=int(st.session_state.get("ui_dist_workers", 2)),
                )

                log_lines = []
                with st.spinner("Locust master/worker çalışıyor..."):
                    if st.session_state.get("ui_dist_stream", True) and mproc.stdout is not None:
                        last_update = 0.0
                        for line in mproc.stdout:
                            log_lines.append("MASTER: " + line.rstrip())
                            now = time.time()
                            if now - last_update > 0.25:
                                last_update = now
                                to_show = "\n".join(log_lines[-200:])
                                log_area.code(to_show)
                    mproc.wait()
                # try to ensure workers exit too
                for wp, wlog, _ in wprocs:
                    try:
                        wp.wait(timeout=3)
                    except Exception:
                        try:
                            wp.terminate()
                        except Exception:
                            pass
            else:
                proc, logfile, html_path, start, cmd = run_locust(
                    locustfile=locustfile_path,
                    host=None if use_file_host else st.session_state.get("ui_host", host),
                    users=int(st.session_state.get("ui_users", users)),
                    spawn_rate=float(st.session_state.get("ui_spawn", spawn)),
                    run_time=st.session_state.get("ui_run_time", run_time),
                    run_dir=run_dir,
                    csv_prefix=st.session_state.get("ui_csv_prefix", csv_prefix),
                    html_report=bool(st.session_state.get("ui_html_report", html_report)),
                    csv_full_history=bool(st.session_state.get("ui_csv_full_history", csv_full_history)),
                    loglevel=st.session_state.get("ui_loglevel", loglevel),
                    csv_flush_interval=(int(st.session_state.get("ui_csv_flush_interval", csv_flush_interval)) or None),
                    stream_logs=bool(st.session_state.get("ui_stream_logs", stream_logs)),
                )

                log_lines = []
                with st.spinner("Locust çalışıyor... Lütfen bekleyin"):
                    if stream_logs and proc.stdout is not None:
                        last_update = 0.0
                        for line in proc.stdout:
                            log_lines.append(line.rstrip())
                            now = time.time()
                            if now - last_update > 0.25:
                                last_update = now
                                to_show = "\n".join(log_lines[-200:])
                                log_area.code(to_show)
                    proc.wait()
                    if not stream_logs:
                        try:
                            tail = (logfile.read_text(encoding="utf-8", errors="ignore").splitlines())[-200:]
                            log_area.code("\n".join(tail))
                        except Exception:
                            pass

            st.session_state["running"] = False
            ended = datetime.utcnow().isoformat()

            # Save run metadata
            meta = {
                "locustfile": str(locustfile_path.relative_to(BASE_DIR)),
                "use_file_host": bool(use_file_host),
                "file_host": file_host_val,
                "typed_host": host if not use_file_host else None,
                "effective_host": effective_host,
                "users": int(users),
                "spawn_rate": float(spawn),
                "run_time": run_time,
                "csv_prefix": csv_prefix,
                "html_report": bool(html_report),
                "csv_full_history": bool(csv_full_history),
                "started_at": start,
                "ended_at": ended,
                "command": " ".join(cmd),
                "distributed": bool(st.session_state.get("ui_dist_enabled", False)),
                "workers": int(st.session_state.get("ui_dist_workers", 0)) if st.session_state.get("ui_dist_enabled", False) else 0,
            }
            try:
                (run_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            except Exception:
                pass

            status_area.success(f"Tamamlandı. Çalışma klasörü: {run_dir}")

        # Scheduler UI
        sched = st.expander("Zamanlayıcı (planlanmış koşular)", expanded=False)
        with sched:
            sc1, sc2 = st.columns([2, 1])
            with sc1:
                st.caption("Bir defalık ya da belirli aralıklarla koşu planlayın. Kayıtlı profil ya da mevcut form değerleri kullanılır.")
            with sc2:
                use_profile_for_sched = st.checkbox("Profilden doldur", value=False, key="ui_sched_use_profile")
            if use_profile_for_sched:
                profs = list_profiles()
                chosen = st.selectbox("Profil seçin", options=profs if profs else ["(profil yok)"])
                pdata = load_profile(chosen) if profs else None
            else:
                pdata = None

            st.markdown("— Oluştur —")
            mode = st.radio("Tür", options=["Bir defalık", "Aralıklı"], horizontal=True)
            if mode == "Bir defalık":
                dcol, tcol = st.columns(2)
                with dcol:
                    dt_date = st.date_input("Tarih")
                with tcol:
                    dt_time = st.time_input("Saat")
                sched_btn = st.button("Planı Kaydet")
                if sched_btn:
                    at_iso = datetime.combine(dt_date, dt_time).isoformat()
                    data = {
                        "id": datetime.utcnow().strftime("sch_%Y%m%d_%H%M%S"),
                        "enabled": True,
                        "type": "one_time",
                        "at": at_iso,
                        "created_at": datetime.utcnow().isoformat(),
                        "locustfile": (pdata.get("selected_file") if pdata else st.session_state.get("ui_selected_file")),
                        "use_file_host": (pdata.get("use_file_host") if pdata else st.session_state.get("ui_use_file_host")),
                        "host": (pdata.get("host") if pdata else st.session_state.get("ui_host")),
                        "users": int((pdata.get("users") if pdata else st.session_state.get("ui_users", 1))),
                        "spawn_rate": float((pdata.get("spawn") if pdata else st.session_state.get("ui_spawn", 1))),
                        "run_time": (pdata.get("run_time") if pdata else st.session_state.get("ui_run_time")),
                        "csv_prefix": (pdata.get("csv_prefix") if pdata else st.session_state.get("ui_csv_prefix")),
                        "html_report": bool((pdata.get("html_report") if pdata else st.session_state.get("ui_html_report", True))),
                        "csv_full_history": bool((pdata.get("csv_full_history") if pdata else st.session_state.get("ui_csv_full_history", True))),
                        "loglevel": (pdata.get("loglevel") if pdata else st.session_state.get("ui_loglevel")),
                    }
                    save_schedule(data)
                    st.success("Zamanlanmış görev kaydedildi.")
            else:
                mins = st.number_input("Her kaç dakikada bir?", min_value=1, value=60)
                sched_btn2 = st.button("Aralıklı Planı Kaydet")
                if sched_btn2:
                    data = {
                        "id": datetime.utcnow().strftime("sch_%Y%m%d_%H%M%S"),
                        "enabled": True,
                        "type": "interval",
                        "every_minutes": int(mins),
                        "created_at": datetime.utcnow().isoformat(),
                        "locustfile": (pdata.get("selected_file") if pdata else st.session_state.get("ui_selected_file")),
                        "use_file_host": (pdata.get("use_file_host") if pdata else st.session_state.get("ui_use_file_host")),
                        "host": (pdata.get("host") if pdata else st.session_state.get("ui_host")),
                        "users": int((pdata.get("users") if pdata else st.session_state.get("ui_users", 1))),
                        "spawn_rate": float((pdata.get("spawn") if pdata else st.session_state.get("ui_spawn", 1))),
                        "run_time": (pdata.get("run_time") if pdata else st.session_state.get("ui_run_time")),
                        "csv_prefix": (pdata.get("csv_prefix") if pdata else st.session_state.get("ui_csv_prefix")),
                        "html_report": bool((pdata.get("html_report") if pdata else st.session_state.get("ui_html_report", True))),
                        "csv_full_history": bool((pdata.get("csv_full_history") if pdata else st.session_state.get("ui_csv_full_history", True))),
                        "loglevel": (pdata.get("loglevel") if pdata else st.session_state.get("ui_loglevel")),
                    }
                    save_schedule(data)
                    st.success("Aralıklı görev kaydedildi.")

            st.markdown("— Kayıtlı Planlar —")
            sch_files = list_schedules()
            if not sch_files:
                st.info("Kayıtlı plan yok.")
            else:
                for sp in sch_files:
                    try:
                        sch = json.loads(sp.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    c1, c2, c3, c4, c5 = st.columns([2,2,2,2,2])
                    with c1:
                        st.write(f"ID: {sch.get('id')}")
                        st.write(f"Tür: {sch.get('type')}")
                    with c2:
                        if sch.get('type') == 'one_time':
                            st.write(f"Zaman: {sch.get('at')}")
                        else:
                            st.write(f"Her {sch.get('every_minutes')} dk")
                        st.write(f"Son Çalışma: {sch.get('last_run_at','-')}")
                    with c3:
                        st.write(f"Locustfile: {sch.get('locustfile')}")
                        st.write(f"Host: {('dosya' if sch.get('use_file_host') else (sch.get('host') or '-'))}")
                    with c4:
                        en = st.checkbox("Etkin", value=bool(sch.get('enabled', True)), key=f"sch_en_{sch.get('id')}")
                        if en != bool(sch.get('enabled', True)):
                            sch['enabled'] = bool(en)
                            sp.write_text(json.dumps(sch, indent=2), encoding='utf-8')
                            st.experimental_rerun()
                    with c5:
                        if st.button("Sil", key=f"sch_del_{sch.get('id')}"):
                            delete_schedule(sch.get('id'))
                            st.experimental_rerun()

    with tabs[1]:
        st.subheader("Raporları Gör")
        runs = list_runs()
        if not runs:
            st.info("Henüz bir çalışma yok. 'Test Çalıştır' sekmesinden başlatın.")
        else:
            run_opts = [str(p.relative_to(BASE_DIR)) for p in runs]
            sel = st.selectbox("Çalışma seçin", run_opts)
            selected_run = BASE_DIR / sel
            data = load_stats_cached(str(selected_run), "stats", run_signature(selected_run))

            # Show metadata if exists
            meta_path = selected_run / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    cols = st.columns(4)
                    cols[0].metric("Etkin Host", meta.get("effective_host", "-"))
                    cols[1].metric("Kullanıcı", str(meta.get("users", "-")))
                    cols[2].metric("Spawn/s", str(meta.get("spawn_rate", "-")))
                    cols[3].metric("Süre", meta.get("run_time", "-"))
                    st.caption(f"Locustfile: {meta.get('locustfile')} | Başlangıç: {meta.get('started_at')} | Bitiş: {meta.get('ended_at')}")
                except Exception:
                    pass

            if "stats" in data and not data["stats"].empty:
                render_summary_from_stats(data["stats"])
                # Threshold hints
                th_p95 = _env_int("THRESHOLD_P95_MS")
                th_success = _env_float("THRESHOLD_SUCCESS_RATE")
                agg = aggregated_metrics(data["stats"])
                if agg:
                    msgs = []
                    if th_p95 is not None and not pd.isna(agg.get("p95_ms")):
                        if agg["p95_ms"] <= th_p95:
                            msgs.append(f"p95 OK: {agg['p95_ms']:.0f} ms ≤ {th_p95} ms")
                        else:
                            msgs.append(f"p95 AŞILDI: {agg['p95_ms']:.0f} ms > {th_p95} ms")
                    if th_success is not None and not pd.isna(agg.get("success_rate")):
                        if agg["success_rate"] >= th_success:
                            msgs.append(f"Başarı OK: {agg['success_rate']:.2f}% ≥ {th_success}%")
                        else:
                            msgs.append(f"Başarı DÜŞÜK: {agg['success_rate']:.2f}% < {th_success}%")
                    if msgs:
                        if any("AŞILDI" in m or "DÜŞÜK" in m for m in msgs):
                            st.error(" | ".join(msgs))
                        else:
                            st.success(" | ".join(msgs))
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

            # Compare runs
            st.divider()
            st.subheader("Koşu Karşılaştırma")
            others = [o for o in run_opts if o != sel]
            if not others:
                st.info("Karşılaştırmak için başka koşu yok.")
            else:
                other_sel = st.selectbox("Karşılaştırılacak çalışma", others)
                other_run = BASE_DIR / other_sel
                data2 = load_stats_cached(str(other_run), "stats", run_signature(other_run))
                if "stats" not in data2 or data2["stats"].empty:
                    st.warning("Diğer koşu için istatistik bulunamadı.")
                else:
                    a1 = aggregated_metrics(data["stats"]) if "stats" in data else {}
                    a2 = aggregated_metrics(data2["stats"]) if "stats" in data2 else {}
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption(f"Seçili: {sel}")
                        st.metric("İstek", f"{a1.get('requests','-')}")
                        st.metric("Başarı %", f"{a1.get('success_rate', float('nan')):.2f}%" if a1 else "-")
                        st.metric("p95 (ms)", f"{a1.get('p95_ms', float('nan')):.0f}" if a1 else "-")
                        st.metric("Median (ms)", f"{a1.get('median_ms', float('nan')):.0f}" if a1 else "-")
                    with c2:
                        st.caption(f"Diğer: {other_sel}")
                        st.metric("İstek", f"{a2.get('requests','-')}")
                        st.metric("Başarı %", f"{a2.get('success_rate', float('nan')):.2f}%" if a2 else "-")
                        st.metric("p95 (ms)", f"{a2.get('p95_ms', float('nan')):.0f}" if a2 else "-")
                        st.metric("Median (ms)", f"{a2.get('median_ms', float('nan')):.0f}" if a2 else "-")

                    st.divider()
                    st.caption("Endpoint Bazlı Karşılaştırma")
                    try:
                        end_a = normalize_endpoint_stats(data.get("stats"))
                        end_b = normalize_endpoint_stats(data2.get("stats"))
                        if end_a.empty and end_b.empty:
                            st.info("Endpoint istatistikleri bulunamadı.")
                        else:
                            # Align by endpoint name
                            end_a = end_a.set_index("name")
                            end_b = end_b.set_index("name")
                            all_names = sorted(set(end_a.index.tolist()) | set(end_b.index.tolist()))
                            comp = pd.DataFrame(index=all_names)
                            # Suffix with A/B
                            for col in ["requests", "failures", "success_rate", "median_ms", "p95_ms"]:
                                comp[f"{col}_A"] = end_a[col] if col in end_a.columns else pd.NA
                                comp[f"{col}_B"] = end_b[col] if col in end_b.columns else pd.NA
                            # Deltas (B - A)
                            comp["p95_delta_ms"] = comp["p95_ms_B"].astype(float) - comp["p95_ms_A"].astype(float)
                            comp["median_delta_ms"] = comp["median_ms_B"].astype(float) - comp["median_ms_A"].astype(float)
                            comp["success_delta_%"] = comp["success_rate_B"].astype(float) - comp["success_rate_A"].astype(float)
                            comp = comp.reset_index().rename(columns={"index": "endpoint"})

                            # Sort by worst p95 regression first
                            comp_sorted = comp.sort_values("p95_delta_ms", ascending=False)
                            st.dataframe(comp_sorted, use_container_width=True, height=360)
                    except Exception:
                        st.warning("Endpoint karşılaştırması oluşturulurken bir sorun oluştu.")

            # Failure/Exception Analysis for selected run
            st.divider()
            st.subheader("Hata Analizi ve Sorunlu Endpointler")
            try:
                failures_df = data.get("failures") if "failures" in data else None
                stats_df = data.get("stats") if "stats" in data else None
                exceptions_df = data.get("exceptions") if "exceptions" in data else None

                cols = st.columns(2)
                with cols[0]:
                    if failures_df is not None and not failures_df.empty:
                        fdf = failures_df.copy()
                        # Normalize
                        name_col = "Name" if "Name" in fdf.columns else ("name" if "name" in fdf.columns else None)
                        err_col = "Error" if "Error" in fdf.columns else ("error" if "error" in fdf.columns else None)
                        occ_col = "Occurrences" if "Occurrences" in fdf.columns else ("occurrences" if "occurrences" in fdf.columns else None)
                        if name_col and err_col and occ_col:
                            by_name = fdf.groupby(name_col)[occ_col].sum().sort_values(ascending=False).head(10)
                            fig_bar = px.bar(by_name, title="En Çok Hata Alan Endpointler (Occurrences)")
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.info("Failures CSV beklenen kolonları içermiyor.")
                    else:
                        st.info("Hata (failures.csv) verisi yok.")
                with cols[1]:
                    if exceptions_df is not None and not exceptions_df.empty:
                        edf = exceptions_df.copy()
                        # Expect columns Count, Message
                        cnt_col = "Count" if "Count" in edf.columns else ("count" if "count" in edf.columns else None)
                        msg_col = "Message" if "Message" in edf.columns else ("message" if "message" in edf.columns else None)
                        if cnt_col and msg_col:
                            edf2 = edf.sort_values(cnt_col, ascending=False).head(10)[[cnt_col, msg_col]]
                            st.dataframe(edf2.rename(columns={cnt_col: "Count", msg_col: "Message"}), use_container_width=True)
                        else:
                            st.info("Exceptions CSV beklenen kolonları içermiyor.")
                    else:
                        st.info("Exception verisi yok.")

            except Exception:
                st.warning("Hata analizi oluşturulurken bir sorun oluştu.")

    with tabs[2]:
        st.subheader("Genel Dashboard")

        def _first_col(df: pd.DataFrame, candidates: list[str]):
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        def _try_number(val, default=None):
            try:
                if pd.isna(val):
                    return default
                return float(val)
            except Exception:
                return default

        # Collect run summaries
        summaries = []
        for r in list_runs():
            meta = {}
            mp = r / "metadata.json"
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            data = load_stats_cached(str(r), "stats", run_signature(r))
            agg = None
            if "stats" in data and not data["stats"].empty:
                sdf = data["stats"]
                name_col = _first_col(sdf, ["Name", "name"]) or "Name"
                agg_rows = sdf[sdf[name_col].astype(str).str.lower() == "aggregated"] if name_col in sdf.columns else pd.DataFrame()
                if not agg_rows.empty:
                    agg = agg_rows.iloc[0]

            if agg is None:
                continue

            # Extract values with multiple possible column names
            req_count = _try_number(agg.get("Request Count", agg.get("Requests", 0)), 0)
            fail_count = _try_number(agg.get("Failure Count", agg.get("Failures", 0)), 0)
            median = _try_number(agg.get("50%ile", agg.get("Median Response Time", agg.get("50%", None))))
            p95 = _try_number(agg.get("95%ile", agg.get("95%", None)))

            # Average RPS from history if available
            avg_rps = None
            if "history" in data and not data["history"].empty:
                hdf = data["history"]
                rps_col = _first_col(hdf, ["Requests/s", "RPS", "requests/s"]) 
                if rps_col:
                    try:
                        avg_rps = float(hdf[rps_col].astype(float).mean())
                    except Exception:
                        avg_rps = None

            summaries.append(
                {
                    "run_id": r.name,
                    "path": str(r),
                    "locustfile": meta.get("locustfile"),
                    "host": meta.get("effective_host") or meta.get("typed_host") or meta.get("file_host"),
                    "started_at": meta.get("started_at"),
                    "ended_at": meta.get("ended_at"),
                    "users": meta.get("users"),
                    "spawn_rate": meta.get("spawn_rate"),
                    "run_time": meta.get("run_time"),
                    "requests": req_count,
                    "failures": fail_count,
                    "success_rate": (1 - fail_count / req_count) * 100 if req_count and req_count > 0 else None,
                    "median_ms": median,
                    "p95_ms": p95,
                    "avg_rps": avg_rps,
                }
            )

        if not summaries:
            st.info("Henüz metadata'lı bir çalışma bulunamadı. Yeni bir test çalıştırın.")
        else:
            df = pd.DataFrame(summaries)
            # Parse datetime for filtering/trends
            if "started_at" in df.columns:
                try:
                    df["started_at_dt"] = pd.to_datetime(df["started_at"], errors="coerce")
                    df["started_date"] = df["started_at_dt"].dt.date
                except Exception:
                    df["started_at_dt"] = pd.NaT
                    df["started_date"] = pd.NaT

            # Filters
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                locust_opts = sorted([x for x in df["locustfile"].dropna().unique().tolist()])
                sel_locust = st.multiselect("Locustfile filtresi", options=locust_opts, default=locust_opts)
            with c2:
                host_opts = sorted([x for x in df["host"].dropna().unique().tolist()])
                sel_host = st.multiselect("Host filtresi", options=host_opts, default=host_opts)
            with c3:
                group_by = st.selectbox("Gruplama", options=["Host", "Locustfile", "Host+Locustfile"], index=0)

            # Date range filter
            if df["started_at_dt"].notna().any():
                min_d = pd.to_datetime(df["started_at_dt"].min()).date()
                max_d = pd.to_datetime(df["started_at_dt"].max()).date()
                dr = st.date_input("Tarih aralığı", value=(min_d, max_d))
            else:
                dr = None

            fdf = df.copy()
            if sel_locust:
                fdf = fdf[fdf["locustfile"].isin(sel_locust)]
            if sel_host:
                fdf = fdf[fdf["host"].isin(sel_host)]
            if dr and isinstance(dr, tuple) and len(dr) == 2:
                start_d, end_d = dr
                try:
                    mask = (fdf["started_date"] >= start_d) & (fdf["started_date"] <= end_d)
                    fdf = fdf[mask]
                except Exception:
                    pass

            if fdf.empty:
                st.warning("Seçilen filtrelerle eşleşen çalışma yok.")
            else:
                # Summary KPIs (weighted where applicable)
                total_runs = len(fdf)
                total_req = float(fdf["requests"].fillna(0).sum())
                total_fail = float(fdf["failures"].fillna(0).sum())
                overall_success = (1 - total_fail / total_req) * 100 if total_req > 0 else None

                # Weighted p95 by requests as an approximation
                def weighted_avg(series, weights):
                    try:
                        s = series.astype(float)
                        w = weights.astype(float)
                        if w.sum() == 0:
                            return None
                        return float((s * w).sum() / w.sum())
                    except Exception:
                        return None

                weighted_p95 = weighted_avg(fdf["p95_ms"].fillna(0), fdf["requests"].fillna(0))

                k = st.columns(4)
                k[0].metric("Koşu Sayısı", str(total_runs))
                k[1].metric("Toplam İstek", f"{int(total_req)}")
                k[2].metric("Başarı Oranı", f"{overall_success:.1f}%" if overall_success is not None else "-")
                k[3].metric("Ağırlıklı p95 (ms)", f"{weighted_p95:.0f}" if weighted_p95 is not None else "-")

                # Trend charts
                st.divider()
                st.caption("Trend Grafikleri")
                if fdf["started_at_dt"].notna().any():
                    t1, t2 = st.columns(2)
                    with t1:
                        try:
                            figt1 = px.line(
                                fdf.sort_values("started_at_dt"),
                                x="started_at_dt",
                                y="p95_ms",
                                color=("host" if group_by in ["Host", "Host+Locustfile"] else "locustfile"),
                                markers=True,
                                title="p95 (ms) Trendi",
                            )
                            st.plotly_chart(figt1, use_container_width=True)
                        except Exception:
                            pass
                    with t2:
                        try:
                            figt2 = px.line(
                                fdf.sort_values("started_at_dt"),
                                x="started_at_dt",
                                y="success_rate",
                                color=("host" if group_by in ["Host", "Host+Locustfile"] else "locustfile"),
                                markers=True,
                                title="Başarı Oranı (%) Trendi",
                            )
                            figt2.update_yaxes(range=[0,100])
                            st.plotly_chart(figt2, use_container_width=True)
                        except Exception:
                            pass

                # Grouping
                if group_by == "Host":
                    gkey = ["host"]
                elif group_by == "Locustfile":
                    gkey = ["locustfile"]
                else:
                    gkey = ["host", "locustfile"]

                g = (
                    fdf.groupby(gkey).agg(
                        requests=("requests", "sum"),
                        failures=("failures", "sum"),
                        avg_p95=("p95_ms", "mean"),
                        avg_median=("median_ms", "mean"),
                        avg_rps=("avg_rps", "mean"),
                        runs=("run_id", "count"),
                    ).reset_index()
                )
                g["success_rate_%"] = (1 - g["failures"] / g["requests"]) * 100

                # Apply thresholds if provided
                th_p95 = _env_int("THRESHOLD_P95_MS")
                th_success = _env_float("THRESHOLD_SUCCESS_RATE")
                if th_p95 is not None:
                    g["p95_ok"] = g["avg_p95"].apply(lambda x: bool(x <= th_p95) if pd.notna(x) else None)
                if th_success is not None:
                    g["success_ok"] = g["success_rate_%"].apply(lambda x: bool(x >= th_success) if pd.notna(x) else None)

                st.divider()
                st.caption("Gruplanmış özet")
                st.dataframe(g, use_container_width=True)

                st.divider()
                c1, c2 = st.columns(2)
                with c1:
                    try:
                        fig = px.bar(g.sort_values("avg_p95", ascending=False),
                                     x=("host" if "host" in g.columns else "locustfile"),
                                     y="avg_p95",
                                     color="success_rate_%",
                                     title="p95 (ms) ve Başarı Oranı")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                with c2:
                    try:
                        fig2 = px.bar(g.sort_values("requests", ascending=False),
                                      x=("host" if "host" in g.columns else "locustfile"),
                                      y="requests",
                                      color="failures",
                                      title="İstek Hacmi ve Hatalar")
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception:
                        pass

                # Optional: Top problematic endpoints across filtered runs (approx by total failures)
                try:
                    st.divider()
                    st.caption("Top Sorunlu Endpointler (filtreye göre)")
                    rows = []
                    for r in list_runs():
                        meta = {}
                        mp = r / "metadata.json"
                        if mp.exists():
                            try:
                                meta = json.loads(mp.read_text(encoding="utf-8"))
                            except Exception:
                                meta = {}
                        host_m = meta.get("effective_host") or meta.get("typed_host") or meta.get("file_host")
                        locust_m = meta.get("locustfile")
                        if sel_locust and locust_m not in sel_locust:
                            continue
                        if sel_host and host_m not in sel_host:
                            continue
                        data_r = load_stats_cached(str(r), "stats", run_signature(r))
                        if "stats" in data_r and not data_r["stats"].empty:
                            end_df = normalize_endpoint_stats(data_r["stats"])  # includes failures/requests
                            if not end_df.empty:
                                end_df = end_df.copy()
                                end_df["host"] = host_m
                                end_df["locustfile"] = locust_m
                                rows.append(end_df)
                    if rows:
                        edfall = pd.concat(rows, ignore_index=True)
                        grp = edfall.groupby(["name"]).agg(
                            requests=("requests", "sum"),
                            failures=("failures", "sum"),
                        ).reset_index()
                        grp["fail_rate_%"] = (grp["failures"] / grp["requests"]) * 100
                        grp = grp.sort_values(["fail_rate_%", "failures"], ascending=False).head(15)
                        st.dataframe(grp, use_container_width=True)
                    else:
                        st.info("Seçili filtrelere göre endpoint verisi bulunamadı.")
                except Exception:
                    pass

    with tabs[3]:
        st.subheader("Geçmiş Çalışmalar")
        runs = list_runs()
        if not runs:
            st.info("Kayıtlı çalışma yok.")
        else:
            for r in runs:
                any_stats = any(r.glob("*_stats.csv"))
                html_exists = (r / "report.html").exists()
                st.write(f"- {r.name}  | CSV: {'✅' if any_stats else '❌'} | HTML: {'✅' if html_exists else '❌'}  | Klasör: {r}")

    with tabs[4]:
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
