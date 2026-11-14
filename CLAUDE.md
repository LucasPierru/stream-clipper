# stream_clipper - project context

An open-source tool that turns a finished Twitch stream into vertical (9:16) highlight clips. It runs on the streamer's own PC, triggered by OBS when the stream stops. Users review and post the clips themselves, so the pipeline must **never upload anything**.

User-facing docs: `README.md` (setup, usage, tuning) and `docs/auto-clip-flow.md` (the automatic flow, diagrams, failure handling, design notes). Keep both up to date when behaviour changes.

## How it works

1. **Trigger: the OBS script `obs/stream_clipper_obs.py`** is the only automatic mode.
   - On `STREAMING_STOPPED` it launches `stream_clipper.py --stream-ended --stream-start ISO` as a detached background process (no window, output appended to `~/.cache/stream_clipper/obs.log`, path configurable). The clipper (`pipeline.ended_stream_args`) finds the archive VOD whose `created_at` is closest to the stream start (within 10 min; the newest if unknown), writes its stream id to `processed_streams.json` **before** the slow work, waits `--vod-delay` (120 s), then downloads and clips it.
   - It must stay a separate process: OBS runs scripts inside its own process, so heavy work there would freeze OBS, and closing OBS would kill it. Keep the script a thin, **stdlib-only** trigger. Inside OBS `sys.executable` is obs64.exe, so `find_python()` uses, in order: the Python set in its settings, `<project>/.venv`, `~/.venvs/stream_clipper`, then the python next to OBS's configured Python (`sys.base_prefix`). OBS's own Python setting must be a real install, not a venv (OBS loads its DLL).
   - Optional local recording: used only if it covers the whole stream (running at stream start or started ≤ 60 s after, and still running at stream stop). The script then waits for `RECORDING_STOPPED` (the file isn't finished before) and launches `stream_clipper.py <file> --recording-start ISO --stream-start ISO`. `link_recording()` matches the VOD and sets `Source.vod_offset` = stream start − recording start (VOD time = file time − offset), applied to Twitch links, `--markers` and `--chat`; clip names, the "Time" column and `--at` stay in file time. A recording stopped or started mid-stream → the VOD is used.
   - There is deliberately **no polling / `--watch` mode and no server** (removed: OBS knows exactly when the stream stops). Don't re-add one.
   - `processed_streams.json` (`clipper/state.py`) is re-read from disk on every check (each run is a separate process), so each stream is clipped only once.
2. **Source:** normally the VOD downloaded with yt-dlp into `CACHE_DIR`; a local recording when one covers the whole stream.
   - Manual runs can take a local file, a VOD URL, or `--channel` (the latest archive). A path that doesn't exist (no `://`) is a clean "file not found" error, not a yt-dlp URL.
   - The stream date comes from `--recording-start` for OBS recordings, the yt-dlp `timestamp` (start of the stream) for downloads, and the file mtime otherwise. `--date` overrides all of them.
   - Twitch VODs download as *fragmented* MP4 (moof/mdat chunks, no index): every ffmpeg open walks all chunk headers, ~12 s per seek on an 8 h VOD, which made webcam detection take ~10 min. `download()` checks the first top-level boxes (`is_fragmented_mp4`) and, if needed, rewrites the file once with `-c copy -movflags +faststart` (`defragment`; needs as much free disk again temporarily; on failure it continues with the slow original).
   - Downloaded VODs are deleted after a successful run unless `--keep-vod` (kept on `--dry-run` or failure). Local recordings are never deleted.
3. **Detection** (one ffmpeg pass decodes the audio to 8 kHz mono PCM; loudness is measured once per second and cached in a `.npy` file):
   - **Audio score (always on):** robust z-score (median/MAD) of loudness vs a rolling 5-minute median, plus 0.5 × the jump vs the previous 5 s. Seconds quieter than `SILENCE_DB` (-50 dB) count as zero, are **left out of the baseline** (`rolling_nanmedian`) and replaced by it before the level and jump are computed. Otherwise a silent "starting soon" screen drags the baseline down and normal talking right after it becomes the top highlights (seen on a real VOD: 5 min of digital silence → all clips at 4–6 min), and sound returning after silence (muted mic, BRB, DMCA-muted parts) looks like a jump. Smoothed over 3 s.
   - **Chat score (`--chat`, off by default** since many small streams have little chat): TwitchDownloaderCLI `chatdownload`, messages per second with extra weight for hype words, shifted back by `--chat-delay` (6 s), summed over 10 s windows and z-scored. Skips itself under 50 messages.
   - **Forced clips:** `--at H:MM:SS`, and `--markers` (Twitch stream markers through Helix; needs a user token).
4. **Picking:**
   - Clips are `--clip-length` 60 s. Greedy by highest score above `--threshold` (3.0), up to `--max-clips` (8), with `--min-gap` between peaks (default = the clip length, so clips never overlap each other or a marker).
   - The first `--skip-start` (60 s) and last `--skip-end` (30 s) are ignored.
   - The peak sits at 65% of the clip. Markers sit at 85%, because streamers press them after the moment happens.
   - Forced clips are always kept and don't count toward `--max-clips`.
5. **Cutting:**
   - ffmpeg with libx264 (veryfast, CRF 20) and AAC.
   - Output 1080×1920. `--layout auto` (default): if a webcam overlay is found and on screen during the clip → `split` (webcam panel on top, 25–42% of the height, gameplay crop below); otherwise `fill` (middle of the frame cropped to 9:16). Also `split` (always), `blur` (whole 16:9 frame over a blurred copy), `--format horizontal` (16:9). `--cam X,Y,W,H` (source pixels) skips detection. `split_geometry()` trims both crops to their panel's aspect ratio (never stretches) and nudges the gameplay crop off the webcam.
   - **Webcam detection** (`facecam.py`): 40 frames sampled across the stream at 640 px wide. YuNet (`cv2.FaceDetectorYN`, model vendored in `clipper/data/`, MIT; OpenCV 5 has no Haar cascades) finds faces; the webcam is a face at the same spot in ≥ 40% of frames and < 30% of the frame height (game characters move, full-screen cams are too big). The overlay edges are the strongest still→changing steps in the per-pixel std map around the face (the room barely changes, gameplay does), with sanity checks. The 640-px box is then **refined at full resolution** (`refine_box` on 20 native-size grey frames): each edge snaps to the exact step, half-changing (blended) edge pixels count as gameplay, and 1 px (2 px at 1080p) more is shaved, so no gameplay strip shows at the panel's edges once it's scaled up ~5x. Edges on the frame border stay put. Per clip, `cam_visible()` compares the cam area with its usual look (median grey difference < 30) on 3 frames; clips without it use `fill`. Verified on a real 5.8 h GTA RP VOD: exact overlay box, in ~6 s.
   - **Optional `--captions`:** faster-whisper transcribes each clip with word timestamps, grouped 3 words per line into an `.srt`, burned in with the `subtitles` filter. ffmpeg runs with `cwd` set to the output folder so the `.srt` path needs no escaping.
6. **Output:**
   - Clips go to `CLIPPER_OUT/<YYYY-MM-DD>/` (default `~/Videos/StreamClips`); a second stream on the same day goes to `<date>_2/`.
   - A `.source` file identifies the VOD. Re-running the same VOD reuses its folder and replaces its old `NN_*.mp4/.srt` files (other files are kept).
   - Files are named `01_1h23m45s.mp4`. Each folder also gets `highlights.md` (with timestamped Twitch links) and `highlights.json`.

## Configuration

`.env` (gitignored; template in `.env.example`) is read by a small built-in loader; real environment variables take priority. `CLIPPER_ENV` (a normal environment variable) can point to a different file. Keys:

- `TWITCH_CHANNEL`, `TWITCH_CLIENT_ID`, `TWITCH_CLIENT_SECRET` (each user creates their own Twitch app; never ship credentials)
- `TWITCH_TOKEN`, `TWITCH_REFRESH_TOKEN`: written by `--twitch-login`
- `CLIPPER_OUT`, `CLIPPER_CACHE`

## Twitch auth

- **App token** (client credentials): enough for `/users` and `/videos` (finding the VOD).
- **User token** (scope `user:read:broadcast`): only for `--markers`. `--twitch-login` runs a one-time authorization-code flow through a local server on `http://localhost:3000`, which must be the redirect URL of the user's dev.twitch.tv app. Tokens are saved to `.env`.
- `helix()` retries once with a fresh token on a 401; the user token renews itself with the refresh token.

## Code layout

`stream_clipper.py` is only a launcher; the code is in the `clipper/` package (also `python -m clipper`, or the `stream-clipper` command after `pip install -e .`). Dependencies are declared in `pyproject.toml` only: numpy, yt-dlp and opencv-python-headless, plus the `captions` extra (faster-whisper) and the `dev` extra (pytest). Twitch calls use urllib: no requests or dotenv.

- `config.py`: `load_env`/`save_env`/`env`, folders (`CACHE_DIR`, `DEFAULT_OUT`, `PROCESSED_FILE`) and all tuning constants
- `util.py`: `ClipperError`, `log`, `hms`, `twitch_t`, `parse_ts`, `run`
- `models.py`: `Source` (`.key` for caches / `.source` tag, `vod_offset`, `to_vod_time`/`from_vod_time`) and `Highlight`
- `source.py`: `resolve_source`, `local_source`, `latest_archive_url`, `download`, `match_archive`, `find_stream_vod`, `link_recording`, `probe_duration`
- `detect.py`: `rolling`, `rolling_nanmedian`, `robust_z`, `loudness_per_second`, `audio_score`, `chat_score`, `pick`
- `twitch.py`: `app_token`, `user_token`, `helix`, `twitch_login`, `user_id`, `recent_archives`, `markers`
- `facecam.py`: `Box`, `Facecam`, `video_size`, `grab_frames`, `detect_faces`, `face_cluster`, `cam_rect`, `refine_edge`, `refine_box`, `find_facecam`, `cam_visible`
- `render.py`: `split_geometry`, `video_filter`, `cut`, `load_whisper`, `transcribe`, `write_srt`
- `output.py`: `make_output_dir`, `clip_name`, `write_report`
- `pipeline.py`: `process` (= `ended_stream_args` for `--stream-ended`, or `link_recording` for a recording; then `find_highlights` + `plan_layouts` / `render_clips` + report)
- `state.py`: `ProcessedStreams`
- `cli.py`: `parse_args`, `parse_box`, `main`

Expected failures raise `ClipperError` (no `sys.exit` inside the code). `main()` prints `error: ...` and exits 1; for OBS runs that line lands in `obs.log`, so keep messages one-line and actionable.

## Tests

`python -m pytest` (after `pip install -e ".[dev]"`). `tests/conftest.py` points `.env`, cache and output to a temp folder and clears Twitch variables before `clipper` is imported, so tests never touch real settings or the network.

- `test_detect.py`: audio score and picking on synthetic loudness
- `test_util_output.py`: time helpers, `.env`, output folders, report links with VOD offset, SRT, filters, state file
- `test_twitch_flow.py`: VOD matching, `--stream-ended` claim/wait/dedupe, recording offset, with a fake Helix
- `test_facecam.py`: face clustering, overlay rectangle on a synthetic stillness map, full-resolution edge refinement, layout planning, `--cam`
- `test_obs_script.py`: the OBS event logic with a fake `obspython` module and clock
- `test_end_to_end.py`: the full CLI on a generated video, including a split layout (skipped without ffmpeg)

CI: `.github/workflows/tests.yml` runs them on Ubuntu and Windows, Python 3.10 and 3.12.

**Not yet verified against the real services:** yt-dlp downloading a real Twitch VOD end to end, Helix VOD lookup and markers, TwitchDownloaderCLI, faster-whisper transcription, the OBS script inside real OBS (event order, Python auto-detection, the process surviving OBS closing), and whether 120 s is enough for a VOD to be downloadable after a stream. `--twitch-login` has worked against real Twitch.

## Ideas / next steps

- Tune the threshold and peak position on real VODs, using `--dry-run`.
- Smarter gameplay crop for `fill`/`split` (follow the action instead of the middle).
- Sharper webcam panel: the overlay is small (~455 px wide in a 1080p stream) and is upscaled ~2.4x, so it looks a bit pixelated. Using `scale=...:flags=lanczos` for the `[cam]` branch in `render.video_filter` would be slightly crisper than the default bicubic (it can't add detail that isn't there).
- Spread clips out more: highlights can still bunch within a few minutes.
- Use an LLM on the transcript to rank the moments and suggest titles.
- Add a scene-change or kill-feed signal for gameplay with no voice.
- Snap clip edges to silences, so clips don't start in the middle of a word.
- A desktop notification when the clips are ready.
- Clean up old VODs in the cache automatically.
- A `--vod-offset` flag for manual runs on local recordings (right now only the OBS script sets the offset).
