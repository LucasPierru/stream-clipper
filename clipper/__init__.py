"""stream_clipper - turn a Twitch VOD into ready-to-post vertical highlight clips.

Module map (pipeline order):
  config    .env loading, defaults and paths
  util      logging, errors, time formatting, subprocess helper
  models    Source / Highlight dataclasses
  source    resolve a local file / VOD URL / channel into a video on disk
  detect    loudness + chat signals, and picking the highlight moments
  twitch    Helix API, OAuth (app + user tokens), stream markers
  render    ffmpeg cutting, vertical layout, Whisper captions
  output    date folders and the highlights.md/.json report
  state     processed_streams.json (each stream is clipped once)
  pipeline  process(): one VOD -> clips
  cli       argument parsing and main()
"""
