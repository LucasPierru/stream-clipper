# stream_clipper

**Stop streaming in OBS, and get vertical highlight clips of your Twitch stream a few minutes later, ready to post as YouTube Shorts or TikToks.**

- **Finds the moments by itself:** spots where you suddenly get louder (yelling, laughing, hype). Optionally also uses your Twitch stream markers, timestamps you choose, or chat activity.
- **Cuts 9:16 clips:** 1080×1920, with the full gameplay frame over a blurred background. Optionally with burned-in captions (Whisper).
- **Runs on your own PC.** No server, no account other than your own Twitch app, and nothing is ever uploaded: you review the clips and post the ones you like.

Clips go to `~/Videos/StreamClips/<stream date>/` (a second stream that day goes to `<date>_2/`). Each folder gets a `highlights.md` listing every clip, with a link to that moment in your VOD.

> **Use it for your own streams.** Downloading other people's VODs to repost them can break Twitch's terms and their copyright.

## Requirements

- Windows, macOS or Linux, with [OBS Studio](https://obsproject.com/) for the automatic mode
- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on your PATH (Windows: `winget install Gyan.FFmpeg`)
- A Twitch account with **Store past broadcasts** turned on (Twitch → Settings → Stream)

## Setup (once)

### 1. Install

```bash
git clone https://github.com/<you>/stream-clipper.git
cd stream-clipper
python -m venv .venv
```

Install into the venv: `pip install -e .` for the clipper only, or `pip install -e ".[captions]"` to also get `--captions` (faster-whisper, a large download):

| Shell | Command |
|---|---|
| cmd / PowerShell | `.venv\Scripts\python -m pip install -e ".[captions]"` |
| Git Bash | `.venv/Scripts/python -m pip install -e ".[captions]"` |
| macOS / Linux | `.venv/bin/python -m pip install -e ".[captions]"` |

The OBS script finds `.venv` by itself.

> If the project folder is synced by OneDrive or Dropbox, mark `.venv` as "always keep on this device" so the sync tool never swaps its files for online-only placeholders. If the venv ever breaks, delete it and repeat this step.

### 2. Create a Twitch app

The clipper uses the Twitch API to find your VOD. Every user creates their own app; it's free.

1. Go to <https://dev.twitch.tv/console/apps> (Twitch requires two-factor authentication on your account for this) and click **Register Your Application**:
   - **Name:** anything unique, for example `yourname-clipper`
   - **OAuth Redirect URLs:** `http://localhost:3000`
   - **Category:** Application Integration
   - **Client Type:** Confidential
2. Copy `.env.example` to `.env`, and fill in:
   - `TWITCH_CHANNEL`: your channel name, as in `twitch.tv/<name>`
   - `TWITCH_CLIENT_ID`: from **Manage** on your app
   - `TWITCH_CLIENT_SECRET`: from **New Secret** on the same page

`.env` is gitignored. Keep it private.

### 3. Set up OBS

1. OBS → Tools → Scripts → **Python Settings**: point it to a regular 64-bit Python install, **not** the venv (OBS can't load a venv; the script itself needs no packages). On Windows, typically `C:\Users\<you>\AppData\Local\Programs\Python\Python3xx`.
2. Tools → Scripts → **+** → pick `obs/stream_clipper_obs.py`. The Script Log should say `loaded - python: ...\.venv\...`.
3. Optional, in the script's settings: extra flags, for example `--markers --captions --lang en`.
4. Test: press **Clip latest VOD now** and follow the log at `~/.cache/stream_clipper/obs.log`.

That's it. From now on, **Stop Streaming** is all it takes: clips appear about 5–15 minutes later, mostly spent downloading the VOD.

## How it works

When you stop streaming, the OBS script starts the clipper in the background. The clipper finds that stream's VOD through the Twitch API, waits 2 minutes for Twitch to finish it, downloads it, and makes the clips. Full details, diagrams and failure handling: [docs/auto-clip-flow.md](docs/auto-clip-flow.md).

- The clipper runs as its own process, so OBS doesn't lag, and the clipping continues if you close OBS.
- Each stream is clipped only once.
- **If you also record locally** (optional): when the recording covers the whole stream, the local file is used instead of the VOD. That means no download, and no Twitch music muting. The clips still link to the right moment in the VOD.

## How it picks moments

- **Audio spikes (always on):** where you suddenly get louder than the last 5 minutes. It adjusts to changes in game volume, and ignores silence (muted mic, BRB screen).
- **`--markers`:** a clip at every stream marker you add while live, for example from Twitch's Stream Manager or a Stream Deck. Run `python stream_clipper.py --twitch-login` once first: it saves a Twitch token to `.env`, which then renews itself.
- **`--at H:MM:SS`:** a clip at a time you choose (manual runs).
- **`--chat`** (off by default): bursts of chat activity. Needs [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader) on your PATH, and skips itself if chat has fewer than 50 messages.

## Manual runs

Activate the venv first (`.venv\Scripts\activate` in cmd/PowerShell, `source .venv/Scripts/activate` in Git Bash, `source .venv/bin/activate` on macOS/Linux). In Git Bash, use forward slashes: `\` is an escape character there.

```bash
python stream_clipper.py                                    # latest VOD of TWITCH_CHANNEL
python stream_clipper.py https://www.twitch.tv/videos/123456789
python stream_clipper.py "D:/Recordings/2026-11-12 19-02-11.mkv"  # a local video file
python stream_clipper.py --dry-run                          # only list the moments it found
python stream_clipper.py --help                             # every option
```

After `pip install -e .`, `stream-clipper` works as a command too.

## Tuning

| Problem | Fix |
|---|---|
| Too many weak clips | `--threshold 4` or `--max-clips 5` |
| Missing moments | `--threshold 2` |
| Clips cut off the build-up | `--peak-position 0.8` |
| Clips cut off the reaction | `--peak-position 0.5` |
| You want different clip lengths | `--clip-length 45` |
| You prefer a full-height crop over the blurred background | `--crop center` |
| You want 16:9 clips | `--format horizontal` |

Downloaded VODs and the audio analysis are cached in `~/.cache/stream_clipper`, so re-running to tune settings is fast. Add `--delete-vod` to free the space once the clips are made. Other folders: `CLIPPER_OUT` and `CLIPPER_CACHE` in `.env`.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests run offline: the Twitch API and OBS are faked, and the end-to-end tests generate a test video with ffmpeg (skipped if ffmpeg isn't installed). GitHub Actions runs them on Linux and Windows for every push.

`stream_clipper.py` is just the launcher (`python -m clipper` works too). The code lives in `clipper/`:

| Module | What it does |
|---|---|
| `config.py` | `.env` loading, folders, tuning defaults |
| `source.py` | local file / VOD URL → video on disk (yt-dlp), finding a stream's VOD |
| `detect.py` | audio + chat signals, picking the moments |
| `twitch.py` | Twitch API, tokens, `--twitch-login`, markers |
| `render.py` | ffmpeg cutting, vertical layout, Whisper captions |
| `output.py` | date folders, `highlights.md` / `.json` |
| `pipeline.py` | one stream → clips |
| `state.py` | `processed_streams.json`, so each stream is clipped once |
| `cli.py` | command-line options |

`obs/stream_clipper_obs.py` is the OBS script. OBS runs scripts inside its own process, so it only reacts to OBS events and starts `stream_clipper.py` as a separate process; the heavy work never runs inside OBS. It uses only the Python standard library.

## License

[MIT](LICENSE)
