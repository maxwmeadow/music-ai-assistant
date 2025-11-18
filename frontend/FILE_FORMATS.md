# File Formats Documentation

This document describes all import/export file formats supported by Music AI Assistant.

---

## Table of Contents

1. [Project Files (.maa)](#project-files-maa)
2. [MIDI Files (.mid)](#midi-files-mid)
3. [Audio Export (.wav)](#audio-export-wav)
4. [Version Compatibility](#version-compatibility)

---

## Project Files (.maa)

### Overview

`.maa` (Music AI Assistant) files are JSON-formatted project files that store the complete state of a project. They can also use the `.json` extension.

### Format

```json
{
  "version": "1.0.0",
  "created": "2024-11-16T10:30:00.000Z",
  "modified": "2024-11-16T10:45:00.000Z",
  "metadata": {
    "title": "My Song",
    "tempo": 120,
    "key": "C",
    "timeSignature": "4/4"
  },
  "tracks": [
    {
      "id": "melody",
      "instrument": "piano/grand_piano_k",
      "notes": [
        {
          "pitch": 60,
          "start": 0.0,
          "duration": 0.5,
          "velocity": 80
        }
      ],
      "samples": null
    }
  ],
  "settings": {
    "loopEnabled": false,
    "loopStart": 0,
    "loopEnd": 0,
    "trackVolumes": {
      "melody": 0.8,
      "drums": 0.9
    }
  },
  "dsl": "tempo(120)\ntrack(\"melody\") {\n  instrument(\"piano/grand_piano_k\")\n  note(\"C4\", 0.5, 0.8)\n}",
  "ir": { ... }
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Format version (currently "1.0.0") |
| `created` | string | Yes | ISO 8601 timestamp of creation |
| `modified` | string | Yes | ISO 8601 timestamp of last modification |
| `metadata.title` | string | Yes | Project title |
| `metadata.tempo` | number | Yes | BPM (beats per minute) |
| `metadata.key` | string | Yes | Musical key (e.g., "C", "Am") |
| `metadata.timeSignature` | string | Yes | Time signature (e.g., "4/4", "3/4") |
| `tracks` | array | Yes | Array of track objects |
| `settings` | object | Yes | Playback and mixer settings |
| `dsl` | string | Yes | Raw DSL code |
| `ir` | object | No | Intermediate Representation (optional) |

### Track Object Structure

```json
{
  "id": "track_name",
  "instrument": "category/instrument_name",
  "notes": [
    {
      "pitch": 60,      // MIDI note number (0-127)
      "start": 0.0,     // Start time in seconds
      "duration": 0.5,  // Duration in seconds
      "velocity": 80    // Velocity (0-127)
    }
  ],
  "samples": null      // For percussion/sample-based tracks
}
```

### Import Behavior

When importing a `.maa` file:
- All project state is restored
- DSL code is loaded into the editor
- Tracks are parsed and displayed in the piano roll
- Mixer settings (volumes, mute/solo) are restored
- Loop region settings are restored

### Export Behavior

When exporting a `.maa` file:
- Current DSL code is saved
- All tracks with their notes are saved
- Mixer settings are captured
- IR representation is optionally included
- File is downloaded as `{title}_{timestamp}.maa`

---

## MIDI Files (.mid)

### Overview

Standard MIDI (Musical Instrument Digital Interface) files are supported for import and export. This allows interoperability with other Digital Audio Workstations (DAWs).

### Supported Features

**Import:**
- ✓ Multi-track MIDI files (Type 1)
- ✓ Tempo extraction
- ✓ Time signature extraction
- ✓ Note pitch, timing, velocity, duration
- ✓ General MIDI program changes
- ✓ Up to 1000+ notes per track

**Export:**
- ✓ Multi-track export (one track per IR track)
- ✓ Tempo metadata
- ✓ Time signature metadata
- ✓ General MIDI program mapping
- ✓ Note pitch, timing, velocity, duration

### Limitations

**Import Limitations:**
- Key signature is not imported (defaults to C)
- Control changes (CC) are ignored
- Pitch bend is ignored
- MIDI Type 2 files (multiple sequences) are not supported
- Channel 10 (drums) notes are skipped during import

**Export Limitations:**
- Sample-based tracks (drums) are not exported
- Only note tracks are exported
- Expression/dynamics beyond velocity are not exported
- No support for MIDI CC messages

### Instrument Mapping

#### Export (App → MIDI)

| App Instrument | General MIDI Program | GM Instrument Name |
|----------------|---------------------|-------------------|
| piano/* | 0 | Acoustic Grand Piano |
| guitar/* | 24 | Acoustic Guitar (nylon) |
| bass/* | 32-39 | Bass instruments |
| synth/* | 80 | Synth Lead |
| pad/* | 88 | Synth Pad |

#### Import (MIDI → App)

| General MIDI Program | App Instrument |
|---------------------|----------------|
| 0-7 (Piano) | piano/grand_piano_k |
| 24-31 (Guitar) | guitar/rjs_guitar_new_strings |
| 32-39 (Bass) | bass/bass_synth |
| 80-87 (Synth Lead) | piano/grand_piano_k (fallback) |
| Other | piano/grand_piano_k (default) |

### Import Behavior

When importing a MIDI file:
1. File is parsed using `@tonejs/midi` library
2. Tempo and time signature extracted from header
3. Each non-empty track is converted to an IR track
4. MIDI notes converted to IR notes (pitch, start, duration, velocity)
5. IR is compiled to DSL and loaded into editor
6. Piano roll displays imported notes

### Export Behavior

When exporting to MIDI:
1. IR is converted to MIDI using `@tonejs/midi`
2. One MIDI track per IR track with notes
3. Tempo and time signature written to header
4. Instrument mapped to General MIDI program
5. File downloaded as `{title}.mid`

---

## Audio Export (.wav)

### Overview

Render your project to a WAV audio file for sharing or further processing in other software.

### Format Details

- **Format:** WAV (Waveform Audio File Format)
- **Sample Rate:** 44.1 kHz (standard CD quality)
- **Bit Depth:** 16-bit (determined by Tone.Recorder)
- **Channels:** Stereo (2 channels)
- **Encoding:** PCM (uncompressed)

### Export Process

1. Compile DSL to executable Tone.js code
2. Calculate project duration from DSL
3. Set up Tone.Recorder connected to master output
4. Execute code to schedule all notes and samples
5. Start recording and playback
6. Wait for duration + 2 second buffer (for release tails)
7. Stop recording and download file

### Duration Calculation

The export automatically calculates duration by:
- Parsing all `note()` and `chord()` calls in DSL
- Finding the latest end time
- Adding 2-second buffer for reverb/release tails
- Defaulting to 10 seconds if no notes found

### Limitations

- Maximum recommended duration: 5 minutes (browser memory limits)
- No real-time effects (export renders what's scheduled)
- No support for MP3 export (WAV only)
- Browser must support `Tone.Recorder` API

### Export Behavior

When exporting audio:
1. User clicks "Export Audio"
2. Progress bar shows recording status (0-100%)
3. After completion, file downloads as `{title}.wav`
4. Transport and recorder are cleaned up

---

## Version Compatibility

### Current Version: 1.0.0

The Music AI Assistant uses semantic versioning for project files.

### Version Format

`MAJOR.MINOR.PATCH`

- **MAJOR:** Incompatible changes (breaking format changes)
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes (backward compatible)

### Compatibility Rules

**Import:**
- Files with version `1.x.x` are compatible
- Files with version `2.x.x` or higher will be rejected
- Validation error shown if version incompatible

**Export:**
- Always exports with current version (1.0.0)
- Includes both `created` and `modified` timestamps
- Forward compatibility not guaranteed

### Future Format Changes

If the format changes in the future:

**Backward Compatible (Minor/Patch):**
- New optional fields added
- Existing fields retain meaning
- Old files still load correctly

**Breaking Changes (Major):**
- Required field changes
- Field type changes
- Structural reorganization
- Requires migration tool

### Migration Strategy

For future major version upgrades:
1. Export all projects to current format
2. Upgrade application
3. Use migration tool (if provided)
4. Re-import projects

---

## Example Files

### Minimal Project File

```json
{
  "version": "1.0.0",
  "created": "2024-11-16T12:00:00.000Z",
  "modified": "2024-11-16T12:00:00.000Z",
  "metadata": {
    "title": "Minimal Example",
    "tempo": 120,
    "key": "C",
    "timeSignature": "4/4"
  },
  "tracks": [],
  "settings": {
    "loopEnabled": false,
    "loopStart": 0,
    "loopEnd": 0,
    "trackVolumes": {}
  },
  "dsl": "tempo(120)"
}
```

### Full Project Example

See `examples/` directory (if available) for complete project examples.

---

## Best Practices

### Saving Projects

1. **Save frequently** - Use Ctrl+S to export project regularly
2. **Use descriptive titles** - Makes finding projects easier
3. **Version your work** - Include version numbers in titles (e.g., "Song v1", "Song v2")
4. **Backup important projects** - Keep copies in cloud storage

### Importing Files

1. **Check file size** - Very large MIDI files (>10MB) may be slow to import
2. **Validate MIDI files** - Test in another MIDI player if import fails
3. **Note count limits** - Projects with >5000 notes may have performance issues
4. **Confirm before overwriting** - Import replaces current project

### Exporting Audio

1. **Compile first** - Always compile DSL before exporting audio
2. **Check duration** - Verify audio length is appropriate
3. **Close other tabs** - Free up browser memory for large exports
4. **Test playback** - Play in app before exporting to verify sound

---

## Troubleshooting

### Project Import Issues

**Error: "Invalid JSON format"**
- File is corrupted or not a valid JSON file
- Try opening in text editor to check syntax
- Re-export from backup if available

**Error: "Missing required field: version"**
- File is not a valid Music AI Assistant project
- May be a different application's format

**Error: "Incompatible project version"**
- Project was created with a newer/older version
- Check application version and update if needed

### MIDI Import Issues

**Error: "Failed to import MIDI"**
- File may be corrupted
- Try opening in another MIDI player first
- Check file is actually a MIDI file (not renamed)

**No notes appear after import**
- MIDI file may only contain control changes
- Check tracks in MIDI editor before importing
- Ensure tracks are not muted in original file

**Wrong instruments**
- MIDI program numbers may not map perfectly
- Manually change instruments after import
- Check General MIDI instrument mapping table

### Audio Export Issues

**Error: "No music to export"**
- Compile DSL code first
- Ensure executable code is generated
- Check for compilation errors

**Export takes too long**
- Project may be very long (>5 minutes)
- Close other browser tabs to free memory
- Consider exporting in shorter sections

**No sound in exported file**
- Verify playback works in app first
- Check mixer volumes (may be muted)
- Ensure notes are actually playing

---

## Technical Details

### Libraries Used

- **@tonejs/midi** - MIDI parsing and generation
- **Tone.js** - Audio synthesis and recording
- **Native File API** - File reading and downloading

### Browser Compatibility

- Chrome 90+ (recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

### Performance Considerations

- Project files: <1MB for typical projects
- MIDI files: Can import files up to 50MB
- Audio export: Limited by browser memory (~5 minutes max)

---

## See Also

- [DSL Reference](./DSL_REFERENCE.md) - Music DSL syntax
- [Architecture](../CLAUDE.md) - System architecture overview
- [API Documentation](./API.md) - Backend API endpoints
