## wavbeat

Convert audio to beat-locked loops with constant chops, legato crossfades, 4/4 kicks, swung hats, optional backbeat claps, scheduled LP/BP sweeps, and glue reverb.  

Drag-and-drop GUI offers sliders for BPM, bars, subdivision, global speed, chop-rate, hat density, and clap deviation; auto-saves numbered WAVs.  

## Parameters

### Required

| Argument | Description |
| :--- | :--- |
| `input_file` (positional) | The input audio file. |

### Optional

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--bpm INT` | **120** | Sets the tempo. |
| `--rate FLOAT` | **1.0** | Playback rate (0.1–2.0) applied to **chops only**; kick, hat, and clap are unaffected. |
| `--reverb FLOAT` | **1.0** | Sets the glue reverb level (0–2, where 0 disables it). |
| `--speed FLOAT` | **1.0** | Sets the global speed multiplier on the final mix. |
| `--bars INT` | **auto 4–16** | Defines the exact number of bars to render. |
| `--subdiv INT` | **1** | Sets the grid subdivision per beat (1=quarters, 2=eighths, 4=sixteenths). |
| `--hat_density FLOAT` | **0.60** | Controls the hat density (lower values result in sparser hats). |
| `--clap_dev_prob FLOAT` | **0.10** | Probability (0–1) of a bonus, deviating clap. |
| `--clap_dev_ms FLOAT` | **22.0** | Maximum absolute deviation (in milliseconds) for the bonus clap. |
| `--clap` | Adds claps on backbeats. |
