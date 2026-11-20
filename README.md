# stream_clipper

**Stop streaming in OBS, and get vertical highlight clips of your Twitch stream a few minutes later, ready to post as YouTube Shorts or TikToks.**

- **Finds the moments by itself:** spots where you suddenly get louder (yelling, laughing, hype). Optionally also uses your Twitch stream markers, timestamps you choose, or chat activity.
- **Cuts 9:16 clips that fill the screen:** 1080×1920, with your webcam on top and the gameplay below when it finds a webcam overlay, or the gameplay filling the whole frame when it doesn't. Optionally with burned-in captions (Whisper).
- **Optional, for YouTube:** the whole VOD cut into 30–60 min parts at natural moments (`--parts`), and music like in-game radio taken out while keeping your voice (`--remove-music`).
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
git clone https://github.com/LucasPierru/stream-clipper.git
cd stream-clipper
python -m venv .venv
```

Activate the venv. Your prompt then starts with `(.venv)`:

| Shell            | Command                          |
| ---------------- | -------------------------------- |
| cmd              | `.venv\Scripts\activate`         |
| PowerShell       | `.venv\Scripts\Activate.ps1`     |
| Git Bash         | `source .venv/Scripts/activate`  |
| macOS / Linux    | `source .venv/bin/activate`      |

> PowerShell may refuse to run the script ("running scripts is disabled"). Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use cmd.

Then install the clipper. Add `[captions]` to also get `--captions` (faster-whisper, a large download):

```bash
python -m pip install -e .               # clipper only
python -m pip install -e ".[captions]"   # clipper + captions
```

For `--remove-music`, see [Music removal](#music-removal---remove-music): it needs PyTorch, and the right build for your graphics card has to be installed first.

Activate the venv again in every new terminal before running the clipper by hand. The OBS script doesn't need this: it finds `.venv` by itself.

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

- **Audio spikes (always on):** where you suddenly get louder than the last 5 minutes. It adjusts to changes in game volume, and ignores silence (a silent "starting soon" screen, muted mic, BRB).
- **`--markers`:** a clip at every stream marker you add while live, for example from Twitch's Stream Manager or a Stream Deck. Run `python stream_clipper.py --twitch-login` once first: it saves a Twitch token to `.env`, which then renews itself.
- **`--at H:MM:SS`:** a clip at a time you choose (manual runs).
- **`--chat`** (off by default): bursts of chat activity. Needs [TwitchDownloaderCLI](https://github.com/lay295/TwitchDownloader) on your PATH, and skips itself if chat has fewer than 50 messages.

## Clip layout

| `--layout` | What you get |
|---|---|
| `auto` (default) | Webcam on top, gameplay below, if a webcam overlay is found. Otherwise `fill`. |
| `split` | Webcam on top, gameplay below, for every clip. |
| `fill` | The gameplay cropped to fill the whole 9:16 frame (the middle of the screen). |
| `blur` | The whole 16:9 frame in the middle, over a blurred copy of itself. |

How the webcam is found: the clipper samples 40 frames across the stream and looks for a face that stays in the same place (game characters move around, so they're ignored). It then finds the edges of the overlay: a straight line at the same place in every frame, where your camera picture meets a different picture (the gameplay), so the pixels on each side change independently of each other. Starting from your face, the first such line on each side is the webcam's edge. This works in a dark room and when you move a lot, and a panel attached to the webcam, such as a translucent chat box under or next to it, is left out: its own outer edge has the same (dimmed) gameplay on both sides. The edge of your desk or a shelf isn't mistaken for it either, because both sides of it are your room. The log says what it found, for example `webcam found: 460x340 at (0,160)`.

- Each clip is checked: if the webcam isn't on screen during that clip (a BRB or full-screen scene), that clip uses `fill`. A change in your room's lighting (the sun coming up, a lamp switched on) doesn't count: the overlay's border is still there.
- If the detection gets it wrong, set the position yourself with `--cam X,Y,W,H`, in pixels of the original video (for 1080p: 0–1920 and 0–1080).
- The gameplay part is a crop of the middle of the screen, moved aside if it would overlap the webcam.

### Vertical, horizontal or both

`--format vertical` (default) makes the 9:16 clips above. `--format horizontal` makes 16:9 clips of the whole frame instead, for example for a YouTube compilation. `--format both` makes both from the same moments: `01_1h23m45s.mp4` (9:16) and `01_1h23m45s_horizontal.mp4` (16:9), with the same captions, and `highlights.md` lists both. Cutting takes about twice as long. For OBS runs, put `--format both` in the script's "Extra flags".

## Full VOD in parts, without the music (for YouTube)

### Parts: `--parts`

Cuts the whole VOD into parts of 30–60 minutes in `<stream date>/parts/`, named `part_01_0h00m00s.mp4`, `part_02_0h39m34s.mp4` and so on. A `parts.md` lists each part's start time (with a link into the VOD), length, and why it was cut there.

Each cut is placed somewhere between 30 and 60 minutes after the previous one, at the most natural moment in that window:

- **a fade to black or a loading screen** (counts the most, the longer the better): games fade out between missions, when you sleep, on a time skip or a loading screen. The part then starts right as the picture comes back. Overlays such as your webcam are ignored, and a scene that is simply dark (night) doesn't count.
- **never in the middle of a cutscene**: cutscenes are recognised by their black bars at the top and bottom. Cutting where one starts or ends is fine.
- **not while heading into a mission**: a long cutscene (30 s or more) usually opens or closes a mission, so it avoids the 3 minutes before one (driving there, walking up to the character), and prefers not to cut in the 5 minutes after one (often the mission it opened).
- **quiet audio**: you aren't in the middle of a sentence
- **a scene change**: the picture changes completely (counts a little)
- **never inside a highlight clip**

A single black frame (opening the phone, a camera cut) doesn't count as a fade. When nothing in the window stands out, it falls back to a quiet moment outside cutscenes, and the part runs as long as allowed. The last part is between 15 and 75 minutes, so it's never a short leftover. It can't truly know that a mission is over, so check `parts.md` and change a cut by hand if needed.

Finding the fades means looking at the whole video at 5 frames per second: a few minutes for a long VOD (about 4 min for 3 h), cached for re-runs. The cuts themselves are lossless copies of the video (no re-encoding), so they're fast and keep the full quality. Change the length with `--part-length 45-60` (minutes).

### Music removal: `--remove-music`

Twitch mutes only part of the copyrighted music in a VOD, and YouTube claims what's left. `--remove-music` finds every stretch with music (for example GTA radio while you drive) and keeps only the voices there. Everything else, including the game audio, stays as it was. It applies to everything the run makes: the parts, the clips and their captions. Highlight detection also runs on the music-free audio, so a loud song doesn't count as a moment.

How it keeps your voice: an AI model (Demucs) splits the audio into vocals and the rest. The vocals still contain the song's singer, so every half second of them is checked by a voice-recognition model that learns **your voice** from the parts of the stream without music: your voice is kept, every other voice is muted, which removes the lyrics. The log shows how much was kept and muted.

Limits:
- It can't tell copyrighted music from royalty-free music, so all music goes, including a game's own soundtrack.
- In those stretches, the game sounds go too, and so do game characters talking over the music. (`KEEP_GAME_SPEECH = True` keeps clear speech, but then many lyrics get through too: a singer's voice on its own often sounds like speech to the model.)
- While you talk over a song, the singer under your voice stays (quietly).
- Parts that Twitch already muted stay silent.

The models are downloaded on first use (about 1.5 GB). With an NVIDIA graphics card it takes a few minutes for a long VOD; on the CPU it works but can take longer than the stream itself. The music-free audio is cached (`CACHE_DIR/<key>_nomusic.m4a`), so re-runs on a kept VOD reuse it.

**Setup, once, with the venv active:**

1. Install PyTorch with GPU support: on [pytorch.org](https://pytorch.org/get-started/locally/), pick Windows, Pip, Python and the newest CUDA version, and run the command it shows with `torch torchaudio`. It looks like `python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130`.
2. `python -m pip install -e ".[music]"`
3. Check that the GPU is used: `python -c "import torch; print(torch.cuda.is_available())"` should print `True`. The log line `music removal on the GPU (...)` confirms it during a run.

Without step 1, pip installs the CPU-only PyTorch, which works but is slow.

**For OBS runs**, put `--parts --remove-music` in the script's "Extra flags" (with `--format both` if you want that too).

## Manual runs

Activate the venv first (see [Install](#1-install)). In Git Bash, use forward slashes: `\` is an escape character there.

```bash
python stream_clipper.py                                    # latest VOD of TWITCH_CHANNEL
python stream_clipper.py https://www.twitch.tv/videos/123456789
python stream_clipper.py "D:/Recordings/2026-11-12 19-02-11.mkv"  # a local video file
python stream_clipper.py --dry-run                          # only list the moments it found
python stream_clipper.py --help                             # every option
```

After `pip install -e .`, `stream-clipper` works as a command too.

## Tuning

| Problem                                          | Fix                                |
| ------------------------------------------------ | ---------------------------------- |
| Too many weak clips                              | `--threshold 4` or `--max-clips 5` |
| Missing moments                                  | `--threshold 2`                    |
| Clips cut off the build-up                       | `--peak-position 0.8`              |
| Clips cut off the reaction                       | `--peak-position 0.5`              |
| You want shorter or longer clips (default 60 s)  | `--clip-length 30`                 |
| The webcam isn't detected, or the box is wrong   | `--cam X,Y,W,H`                    |
| You prefer the whole frame over a blurred copy   | `--layout blur`                    |
| You want 16:9 clips                              | `--format horizontal`              |
| You want both 9:16 and 16:9 clips                | `--format both`                    |
| You want the whole VOD in parts for YouTube      | `--parts` (`--part-length 30-60`)  |
| Music gets your YouTube uploads claimed          | `--remove-music`                   |

To try settings without cutting anything, use `--dry-run`. Downloaded VODs are deleted once the clips are made; add `--keep-vod` to keep one in `~/.cache/stream_clipper` while you re-run to tune settings (the audio analysis is always cached, it's tiny). Local video files are never deleted. Other folders: `CLIPPER_OUT` and `CLIPPER_CACHE` in `.env`.

### Default settings (`clipper/config.py`)

The defaults live as constants in the "Tuning defaults" part of `clipper/config.py`. A flag overrides its constant for one run (in the OBS script, put flags in "Extra flags"). Editing the constant changes the default for every run, OBS runs included. Settings without a flag can only be changed there.

| Constant          | Default | Flag              | What it does |
| ----------------- | ------- | ----------------- | ------------ |
| `CLIP_LENGTH`     | `60`    | `--clip-length`   | Length of each clip, in seconds. |
| `PEAK_POSITION`   | `0.65`  | `--peak-position` | Where a detected moment sits in its clip, from 0 (start) to 1 (end). With 0.65 and 60 s clips, the clip starts 39 s before the moment and ends 21 s after it. |
| `MARKER_POSITION` | `0.85`  | none              | The same, for stream markers (`--markers`) and `--at` times. It's later in the clip because you press a marker just *after* the moment: with 60 s clips, the clip starts 51 s before the marker and ends 9 s after it. |
| `MAX_CLIPS`       | `8`     | `--max-clips`     | Most clips found automatically per stream. Marker and `--at` clips are always added on top. |
| `MIN_GAP`         | `None`  | `--min-gap`       | Minimum seconds between two moments. `None` means the clip length, so clips never overlap. |
| `THRESHOLD`       | `3.0`   | `--threshold`     | How strong a moment must be to become a clip. Lower gives more (weaker) clips, higher gives fewer. |
| `SKIP_START`      | `60`    | `--skip-start`    | Seconds ignored at the start of the stream (the "starting soon" screen). |
| `SKIP_END`        | `30`    | `--skip-end`      | Seconds ignored at the end (the outro). |
| `SILENCE_DB`      | `-50`   | none              | Seconds quieter than this (in dB) count as silence: they're never highlights and don't drag the "normal loudness" baseline down. |
| `HYPE_WORDS`      | `lul`, `pog`, ... | none    | Chat words that count extra with `--chat`. |
| `CAPTION_STYLE`   | white bold Arial | none     | Look of the burned-in `--captions` (font, size, outline, position). |
| `AUDIO_SR`        | `8000`  | none              | Sample rate for the loudness analysis. There's no need to change it. |
| `VIDEO_EXTS`      | `.mp4`, `.mkv`, ... | none  | File types recognised as video files. |
| `PART_MIN`, `PART_MAX` | `30*60`, `60*60` | `--part-length` | Length of the `--parts`, in seconds. |
| `BLACK_LEVEL`, `BLACK_FRAC` | `20`, `0.85` | none | `--parts`: a frame with this share of pixels darker than `BLACK_LEVEL` (0–255) is black (a fade or loading screen). The share leaves room for a webcam overlay; lower it if your overlays cover more of the screen. |
| `MUSIC_ON`, `MUSIC_OFF` | `0.35`, `0.20` | none | `--remove-music`: a music stretch starts when the music detector is this sure, and lasts while it stays above the second value. Raise them if talking gets treated as music. |
| `MUSIC_MIN`, `MUSIC_GAP`, `MUSIC_PAD` | `8`, `10`, `2` | none | Seconds: shorter music is left alone, closer stretches are merged, and each stretch is widened by this much. |
| `VOICE_SIM`       | `0.65`  | none              | `--remove-music`: how much a voice must sound like yours (0–1) to be kept. Lower it if your own voice gets cut, raise it if singers get through. |
| `KEEP_GAME_SPEECH` | `False` | none             | `--remove-music`: also keep other voices that are clearly speech (game characters) in the music stretches. Lets many lyrics through. |
| `SPEECH_KEEP`, `SPEECH_MUSIC_MAX` | `0.5`, `0.15` | none | With `KEEP_GAME_SPEECH`: other voices are kept when they're clearly speech (at least the first value) with little music in them (below the second). |

If you edit `clipper/config.py`, a later `git pull` may conflict with your edit. Flags in the OBS "Extra flags" box avoid that.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
```

The tests run offline: the Twitch API, OBS and the music-removal models are faked, and the end-to-end tests generate a test video with ffmpeg (skipped if ffmpeg isn't installed). GitHub Actions runs them on Linux and Windows for every push.

`stream_clipper.py` is just the launcher (`python -m clipper` works too). The code lives in `clipper/`:

| Module        | What it does                                                          |
| ------------- | --------------------------------------------------------------------- |
| `config.py`   | `.env` loading, folders, tuning defaults                              |
| `source.py`   | local file / VOD URL → video on disk (yt-dlp), finding a stream's VOD |
| `detect.py`   | audio + chat signals, picking the moments                             |
| `twitch.py`   | Twitch API, tokens, `--twitch-login`, markers                         |
| `facecam.py`  | webcam overlay detection (YuNet face model in `clipper/data/`)        |
| `render.py`   | ffmpeg cutting, layouts, Whisper captions                             |
| `music.py`    | `--remove-music`: music detection (AST) and voice separation (Demucs) |
| `parts.py`    | `--parts`: natural cut points, lossless VOD parts                     |
| `output.py`   | date folders, `highlights.md` / `.json`                               |
| `pipeline.py` | one stream → clips                                                    |
| `state.py`    | `processed_streams.json`, so each stream is clipped once              |
| `cli.py`      | command-line options                                                  |

`obs/stream_clipper_obs.py` is the OBS script. OBS runs scripts inside its own process, so it only reacts to OBS events and starts `stream_clipper.py` as a separate process; the heavy work never runs inside OBS. It uses only the Python standard library.

## License

[MIT](LICENSE)
