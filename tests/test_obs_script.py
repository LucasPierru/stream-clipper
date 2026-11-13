"""The OBS script's event logic, with a fake `obspython` module and a fake clock."""
import datetime as dt
import importlib
import sys
import types

import pytest

from conftest import ROOT

EVENTS = dict(OBS_FRONTEND_EVENT_RECORDING_STARTED=1, OBS_FRONTEND_EVENT_RECORDING_STOPPED=2,
              OBS_FRONTEND_EVENT_STREAMING_STARTED=3, OBS_FRONTEND_EVENT_STREAMING_STOPPED=4)


class FakeOBS:
    def __init__(self, tmp_path, monkeypatch):
        self.streaming = self.recording = False
        self.recording_file = tmp_path / "rec.mkv"
        self.recording_file.write_bytes(b"x")
        mod = types.ModuleType("obspython")
        vars(mod).update(EVENTS)
        mod.obs_frontend_streaming_active = lambda: self.streaming
        mod.obs_frontend_recording_active = lambda: self.recording
        mod.obs_frontend_get_last_recording = lambda: str(self.recording_file)
        self.callbacks = []
        mod.obs_frontend_add_event_callback = self.callbacks.append
        monkeypatch.setitem(sys.modules, "obspython", mod)
        monkeypatch.syspath_prepend(str(ROOT / "obs"))
        sys.modules.pop("stream_clipper_obs", None)
        self.script = importlib.import_module("stream_clipper_obs")
        self.script.settings.update(python=sys.executable, log=str(tmp_path / "obs.log"))
        self.clock = dt.datetime(2026, 11, 12, 20, 0).astimezone()
        monkeypatch.setattr(self.script, "now", lambda: self.clock)
        self.launched = []
        monkeypatch.setattr(self.script.subprocess, "Popen",
                            lambda cmd, **kw: self.launched.append(cmd[2:]))
        self.script.script_load(None)

    def tick(self, seconds):
        self.clock += dt.timedelta(seconds=seconds)

    def fire(self, name, **flags):
        for k, v in flags.items():
            setattr(self, k, v)
        self.callbacks[0](EVENTS["OBS_FRONTEND_EVENT_" + name])

    def take(self):
        out, self.launched = self.launched, []
        return out


@pytest.fixture
def obs(tmp_path, monkeypatch):
    return FakeOBS(tmp_path, monkeypatch)


def test_stream_without_recording_clips_the_vod(obs):
    start = obs.clock
    obs.fire("STREAMING_STARTED", streaming=True)
    obs.tick(3600)
    obs.fire("STREAMING_STOPPED", streaming=False)
    assert obs.take() == [["--stream-ended", "--stream-start", start.isoformat(timespec="seconds")]]


def test_auto_record_waits_for_the_file(obs):
    obs.fire("STREAMING_STARTED", streaming=True)
    obs.tick(1)
    obs.fire("RECORDING_STARTED", recording=True)
    obs.tick(3600)
    obs.fire("STREAMING_STOPPED", streaming=False)
    assert obs.take() == []                         # file not finished yet
    obs.fire("RECORDING_STOPPED", recording=False)
    (cmd,) = obs.take()
    assert cmd[0] == str(obs.recording_file)
    assert "--recording-start" in cmd and "--stream-start" in cmd


def test_recording_restarted_mid_stream_uses_vod(obs):
    obs.fire("STREAMING_STARTED", streaming=True)
    obs.fire("RECORDING_STARTED", recording=True)
    obs.tick(600)
    obs.fire("RECORDING_STOPPED", recording=False)
    assert obs.take() == []                         # still live: never clip mid-stream
    obs.tick(120)
    obs.fire("RECORDING_STARTED", recording=True)   # too late to cover the stream
    obs.tick(600)
    obs.fire("STREAMING_STOPPED", streaming=False)
    assert obs.take()[0][0] == "--stream-ended"
    obs.fire("RECORDING_STOPPED", recording=False)
    assert obs.take() == []


def test_recording_without_stream_is_ignored(obs):
    obs.fire("RECORDING_STARTED", recording=True)
    obs.tick(60)
    obs.fire("RECORDING_STOPPED", recording=False)
    assert obs.take() == []


def test_use_recording_off_and_disabled(obs):
    obs.script.settings["use_recording"] = False
    obs.fire("STREAMING_STARTED", streaming=True)
    obs.fire("RECORDING_STARTED", recording=True)
    obs.fire("STREAMING_STOPPED", streaming=False)
    assert obs.take()[0][0] == "--stream-ended"
    obs.script.settings["enabled"] = False
    obs.fire("STREAMING_STARTED", streaming=True)
    obs.fire("STREAMING_STOPPED", streaming=False)
    assert obs.take() == []


def test_extra_flags_and_buttons(obs):
    obs.script.settings["extra_args"] = '--captions --lang fr --out "D:/My Clips"'
    obs.script.clip_latest_vod_now(None, None)
    assert obs.take() == [["--vod-delay", "0", "--stream-ended", "--captions", "--lang", "fr",
                           "--out", "D:/My Clips"]]


def test_find_python_prefers_project_venv(obs, tmp_path, monkeypatch):
    obs.script.settings["python"] = ""
    venv = tmp_path / ".venv"
    exe = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    monkeypatch.setattr(obs.script, "VENV_DIRS", (venv,))
    assert obs.script.find_python() == str(exe)
