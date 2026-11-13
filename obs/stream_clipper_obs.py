"""OBS script: make highlight clips as soon as the stream stops.

Load it in OBS: Tools -> Scripts -> "+" -> pick this file.
(OBS needs a 64-bit Python 3 set in Tools -> Scripts -> Python Settings.)

When the stream stops:
  * no local recording (the normal case) -> start
        python stream_clipper.py --stream-ended --stream-start ...
    which finds the stream's Twitch VOD, waits for Twitch to finalise it,
    downloads it and clips it;
  * a recording that covered the whole stream -> wait for that recording to
    stop (with auto-record, a second later) and clip the local file instead.
    A recording that was stopped or started mid-stream is incomplete, so the
    VOD is used then.

The clipper runs as a separate background process, so OBS never freezes.
Output goes to the log file set below. This file only uses the standard
library: numpy etc. are needed by the Python that runs stream_clipper.py.
"""
from __future__ import annotations

import datetime as dt
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import obspython as obs

HERE = Path(__file__).resolve().parent
DEFAULT_CLIPPER = HERE.parent / "stream_clipper.py"
DEFAULT_LOG = Path.home() / ".cache" / "stream_clipper" / "obs.log"
RECORDING_GRACE = dt.timedelta(seconds=60)  # a recording starting this soon after the stream counts as covering it

settings = {
    "clipper": str(DEFAULT_CLIPPER),
    "python": "",                # empty = auto-detect
    "extra_args": "",            # e.g. --captions --lang fr
    "log": str(DEFAULT_LOG),
    "use_recording": True,       # use a local recording when it covers the whole stream
    "enabled": True,
}

state = {
    "stream_start": None,        # aware datetime of the current/last stream start
    "recording_start": None,     # aware datetime of the current recording start
    "rec_covers_stream": False,  # current recording has been running since the stream started
    "waiting_for_recording": False,  # stream stopped; clip the file once the recording stops
}


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def say(msg: str) -> None:
    print(f"[stream_clipper] {msg}")  # shows in Tools -> Scripts -> Script Log


# ----------------------------------------------------------------------------
# Launching the clipper
# ----------------------------------------------------------------------------
VENV_DIRS = (HERE.parent / ".venv", Path.home() / ".venvs" / "stream_clipper")


def find_python() -> str | None:
    """The Python that runs stream_clipper.py: the one set in the settings, else the
    project's .venv (or ~/.venvs/stream_clipper), else OBS's own Python
    (inside OBS sys.executable is obs64.exe, so look next to sys.base_prefix)."""
    if settings["python"]:
        return settings["python"]
    for venv in VENV_DIRS:
        for name in ("Scripts/python.exe", "bin/python"):
            if (venv / name).is_file():
                return str(venv / name)
    for base in (sys.base_prefix, sys.prefix):
        for name in ("python.exe", "bin/python3", "bin/python"):
            cand = Path(base) / name
            if cand.is_file():
                return str(cand)
    return shutil.which("python") or shutil.which("python3")


def iso(t: dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


def launch(clipper_args: list[str]) -> None:
    """Start stream_clipper.py in the background with these arguments."""
    if not settings["enabled"]:
        return
    python = find_python()
    if not python:
        say("no Python found - set it in the script settings")
        return
    clipper = Path(settings["clipper"])
    if not clipper.is_file():
        say(f"stream_clipper.py not found at {clipper} - fix it in the script settings")
        return
    cmd = [python, str(clipper), *clipper_args, *shlex.split(settings["extra_args"])]

    log_path = Path(settings["log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")  # inherited by the child, closed below
    logf.write(f"\n===== {now():%Y-%m-%d %H:%M:%S}  {subprocess.list2cmdline(cmd)}\n")
    logf.flush()
    kw = {}
    if os.name == "nt":  # no console window; keep running if OBS closes
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    try:
        subprocess.Popen(cmd, cwd=str(clipper.parent), stdout=logf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, env=env, **kw)
        say(f"clipping started - progress in {log_path}")
    except OSError as e:
        say(f"could not start the clipper: {e}")
    finally:
        logf.close()


def clip_vod(stream_start: dt.datetime | None) -> None:
    args = ["--stream-ended"]
    if stream_start:
        args += ["--stream-start", iso(stream_start)]
    say("stream stopped - will clip its Twitch VOD")
    launch(args)


def clip_recording(recording: str | None, rec_start, stream_start) -> None:
    if not recording or not Path(recording).is_file():
        say(f"recording file not found ({recording!r}) - using the Twitch VOD instead")
        clip_vod(stream_start)
        return
    args = [recording]
    if rec_start:
        args += ["--recording-start", iso(rec_start)]
    if stream_start:
        args += ["--stream-start", iso(stream_start)]
    say("stream stopped - will clip the local recording")
    launch(args)


# ----------------------------------------------------------------------------
# OBS events
# ----------------------------------------------------------------------------
def on_event(event) -> None:
    if event == obs.OBS_FRONTEND_EVENT_STREAMING_STARTED:
        state["stream_start"] = now()
        # a recording already running when the stream starts covers it
        state["rec_covers_stream"] = obs.obs_frontend_recording_active()
        state["waiting_for_recording"] = False

    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STARTED:
        t = now()
        state["recording_start"] = t
        start = state["stream_start"]
        state["rec_covers_stream"] = bool(
            obs.obs_frontend_streaming_active() and start and t - start <= RECORDING_GRACE)

    elif event == obs.OBS_FRONTEND_EVENT_STREAMING_STOPPED:
        if (settings["use_recording"] and state["rec_covers_stream"]
                and obs.obs_frontend_recording_active()):
            state["waiting_for_recording"] = True  # the file isn't finished yet
            say("stream stopped - waiting for the recording to finish")
        else:
            clip_vod(state["stream_start"])

    elif event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED:
        if state["waiting_for_recording"]:
            state["waiting_for_recording"] = False
            clip_recording(obs.obs_frontend_get_last_recording(),
                           state["recording_start"], state["stream_start"])
        # stopped mid-stream: the file is incomplete, the VOD will be used
        state["rec_covers_stream"] = False
        state["recording_start"] = None


def clip_latest_vod_now(props, prop) -> bool:
    """Button: clip the latest Twitch VOD right away (handy to test the setup)."""
    launch(["--vod-delay", "0", "--stream-ended"])
    return False


def clip_last_recording_now(props, prop) -> bool:
    """Button: clip the last local recording right away."""
    rec = obs.obs_frontend_get_last_recording()
    if rec and Path(rec).is_file():
        launch([rec])
    else:
        say("no recording found")
    return False


# ----------------------------------------------------------------------------
# OBS script API
# ----------------------------------------------------------------------------
def script_description() -> str:
    return ("<b>stream_clipper</b><br>Makes highlight clips as soon as you stop streaming, "
            "from your Twitch VOD (or from your local recording if it covers the whole stream).")


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_bool(props, "enabled", "Clip automatically when the stream stops")
    obs.obs_properties_add_bool(props, "use_recording",
                                "Use the local recording if it covers the whole stream")
    obs.obs_properties_add_path(props, "clipper", "stream_clipper.py", obs.OBS_PATH_FILE,
                                "Python script (*.py)", str(DEFAULT_CLIPPER.parent))
    obs.obs_properties_add_path(props, "python", "Python (empty = auto: venv, else OBS's Python)", obs.OBS_PATH_FILE,
                                "Python (python.exe python3 python)", None)
    obs.obs_properties_add_text(props, "extra_args", "Extra flags (e.g. --captions --lang fr)",
                                obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_path(props, "log", "Log file", obs.OBS_PATH_FILE_SAVE,
                                "Log (*.log *.txt)", None)
    obs.obs_properties_add_button(props, "clip_vod", "Clip latest VOD now", clip_latest_vod_now)
    obs.obs_properties_add_button(props, "clip_rec", "Clip last recording now",
                                  clip_last_recording_now)
    return props


def script_defaults(data) -> None:
    obs.obs_data_set_default_bool(data, "enabled", True)
    obs.obs_data_set_default_bool(data, "use_recording", True)
    obs.obs_data_set_default_string(data, "clipper", str(DEFAULT_CLIPPER))
    obs.obs_data_set_default_string(data, "python", "")
    obs.obs_data_set_default_string(data, "extra_args", "")
    obs.obs_data_set_default_string(data, "log", str(DEFAULT_LOG))


def script_update(data) -> None:
    settings.update(
        enabled=obs.obs_data_get_bool(data, "enabled"),
        use_recording=obs.obs_data_get_bool(data, "use_recording"),
        clipper=obs.obs_data_get_string(data, "clipper") or str(DEFAULT_CLIPPER),
        python=obs.obs_data_get_string(data, "python"),
        extra_args=obs.obs_data_get_string(data, "extra_args"),
        log=obs.obs_data_get_string(data, "log") or str(DEFAULT_LOG),
    )


def script_load(data) -> None:
    obs.obs_frontend_add_event_callback(on_event)
    say(f"loaded - python: {find_python()}")
