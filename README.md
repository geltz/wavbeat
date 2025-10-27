## wavbeat

Turns any audio file into a beat-locked loop. Constant-speed chops with legato crossfades, 4-on-the-floor kicks, sparse swung hats, optional backbeat claps, scheduled LP/BP sweeps, and glue reverb.  

Drag-and-drop GUI with sliders for BPM, bars, subdivision, global speed, chop-rate, hat density, and clap deviation; auto-saves numbered WAVs.   

## Parameters

* `input_file` (positional): input audio file.
* `--bpm INT` (default **120**): tempo.
* `--rate FLOAT` (default **1.0**): playback rate (0.1..2.0) for **chops only**; kick/hat/clap unaffected.
* `--reverb FLOAT` (default **1.0**): glue reverb 0..2 (0 disables).
* `--speed FLOAT` (default **1.0**): global speed on final mix.
* `--bars INT` (default **auto 4..16**): exact bars to render.
* `--subdiv INT` (default **1**): grid per beat (1=quarters, 2=eighths, 4=sixteenths).
* `--hat_density FLOAT` (default **0.60**): hat density (lower = sparser).
* `--clap` (flag): add claps on backbeats.
* `--clap_dev_prob FLOAT` (default **0.10**): probability of bonus deviating clap (0..1).
* `--clap_dev_ms FLOAT` (default **22.0**): max absolute deviation (ms) for bonus clap.


