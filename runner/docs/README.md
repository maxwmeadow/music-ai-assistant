# Music Synthesis System v2.0

## Architecture Overview

This is a complete refactor of the music synthesis system with clear separation of concerns, improved performance, and better developer experience.

### Core Principles

1. **Modular Design** - Each component has a single, well-defined responsibility
2. **Performance First** - Caching, batching, and lazy loading throughout
3. **Developer Friendly** - Clear APIs, comprehensive error handling, debugging tools
4. **Scalable** - Support for complex compositions with chords and polyphony

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ MusicPlayer │  │ DevToolkit   │  │ Custom UI/Logic  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Compilation Layer                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              MusicCompiler                           │  │
│  │  • Compiles DSL or IR to playback instructions       │  │
│  │  • Coordinates all subsystems                        │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌───────────────────┐ ┌──────────────┐ ┌────────────────────┐
│ InstrumentFactory │ │   Scheduler  │ │   SampleCache     │
│                   │ │              │ │                   │
│ • Creates synths  │ │ • Schedules  │ │ • Caches mappings │
│ • Loads samples   │ │   notes      │ │ • Manages URLs    │
│ • Manages pools   │ │ • Handles    │ │ • Batches loads   │
│                   │ │   chords     │ │                   │
└───────────────────┘ └──────────────┘ └────────────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                    ┌──────────────────┐
                    │    Tone.js       │
                    │  Audio Engine    │
                    └──────────────────┘
```

---

## Component Guide

### 1. SampleCache.js
**Purpose**: Centralized sample and mapping management

**Key Features**:
- Caches instrument mappings to avoid redundant fetches
- Deduplicates concurrent requests
- Builds optimized sample URL maps
- Supports preloading for critical instruments

**Usage**:
```javascript
import sampleCache from './SampleCache.js';

await sampleCache.loadCatalog();
const mapping = await sampleCache.getMapping('piano/grand_piano_k');
const { urls, baseUrl } = sampleCache.buildSamplerUrls(mapping, 'piano/grand_piano_k');
```

**Performance Benefits**:
- Mapping loaded once, reused everywhere
- Browser caching enabled (`force-cache`)
- Concurrent request deduplication

---

### 2. InstrumentFactory.js
**Purpose**: Smart instrument creation and management

**Key Features**:
- Creates both synthesized and sampled instruments
- Polyphonic pools for chord support
- Instrument caching and reuse
- Voice allocation for simultaneous notes

**Usage**:
```javascript
import instrumentFactory from './InstrumentFactory.js';

// Create single instrument
const piano = await instrumentFactory.createInstrument('piano/grand_piano_k');

// Create polyphonic pool (8 voices)
const pianoPool = await instrumentFactory.createPolyphonicPool('piano/grand_piano_k', 8);

// Play chord
pianoPool.triggerChord(['C4', 'E4', 'G4'], '1n', Tone.now(), 0.8);
```

**Polyphonic Support**:
- Each voice can play independently
- Round-robin voice allocation
- Automatic voice management
- No note cutting/stealing

---

### 3. MusicScheduler.js
**Purpose**: Advanced note and event scheduling

**Key Features**:
- Precise timing using Tone.Transport
- Native chord support
- Tempo changes
- Offline rendering for export

**Usage**:
```javascript
import MusicScheduler from './MusicScheduler.js';

const scheduler = new MusicScheduler();

// Schedule single note
scheduler.scheduleNote(instrument, 'C4', 0.5, '0:0', 0.8);

// Schedule chord
scheduler.scheduleChord(instrument, ['C4', 'E4', 'G4'], 1.0, '0:0', 0.8);

// Start playback
await scheduler.start(120); // 120 BPM
```

---

### 4. MusicCompiler.js
**Purpose**: Compiles DSL or IR to executable music

**Key Features**:
- Supports both DSL and IR formats
- Automatic instrument loading
- Chord detection and handling
- WAV export

**Usage**:
```javascript
import MusicCompiler from './MusicCompiler.js';

const compiler = new MusicCompiler();
await compiler.initialize();

// From DSL
await compiler.compileDSL(dslCode);

// From IR
await compiler.compileIR(irData);

// Start playback
await compiler.scheduler.start(120);

// Export
await compiler.exportWAV();
```

---

### 5. MusicPlayer.js
**Purpose**: High-level client integration

**Key Features**:
- Simple API for common use cases
- Server validation integration
- Auto-stop after duration
- Playback controls

**Usage**:
```javascript
import MusicPlayer from './MusicPlayer.js';

const player = new MusicPlayer('http://localhost:5001');
await player.initialize();

// Play DSL
await player.playDSL(dslCode);

// Play IR
await player.playIR(irData);

// Controls
player.pause();
player.resume();
player.stop();

// Export
await player.exportWAV();
```

---

### 6. DevToolkit.js
**Purpose**: Development and debugging utilities

**Key Features**:
- Performance profiling
- Pattern generation
- Validation testing
- Diagnostic reports

**Usage**:
```javascript
import devToolkit from './DevToolkit.js';

// Enable debug mode
devToolkit.enableDebug();

// Profile instrument load
const profile = await devToolkit.profileInstrumentLoad(
  'piano/grand_piano_k',
  sampleCache,
  instrumentFactory
);

// Test patterns
const scale = devToolkit.generateTestPattern('scale');
const chord = devToolkit.generateTestPattern('chord');

// Validate
const dslTest = devToolkit.testDSL(dslCode);
const irTest = devToolkit.testIR(irData);

// Export diagnostics
devToolkit.exportDiagnostics(compiler);
```

---

## Server API (server-v2.js)

### Endpoints

#### POST /compile
Compile DSL or IR to playback instructions
```json
{
  "format": "dsl",  // or "ir"
  "data": "..."
}
```

#### GET /instruments
List available instruments with optional filtering
```
GET /instruments?category=piano&search=grand
```

#### POST /validate/dsl
Validate DSL syntax

#### POST /validate/ir
Validate IR format

#### GET /dev/stats
Development statistics and metrics

---

## Migration Guide

### From Old System

**Old approach** (generator.js):
```javascript
const generator = new DSLGenerator();
const dsl = generator.generate(parsedData);
const toneCode = generator.compileDSLToToneJS(dsl);
// Execute toneCode...
```

**New approach**:
```javascript
const compiler = new MusicCompiler();
await compiler.initialize();
await compiler.compileIR(irData);
await compiler.scheduler.start(120);
```

### Key Improvements

1. **No more string-based code generation** - Direct object manipulation
2. **Automatic caching** - Instruments and samples cached intelligently
3. **Native chord support** - No workarounds needed
4. **Better error handling** - Clear error messages and recovery
5. **Performance monitoring** - Built-in profiling tools

---

## Performance Optimizations

### Sample Loading
- ✅ Mappings cached globally
- ✅ Concurrent request deduplication
- ✅ Browser cache enabled
- ✅ Optional preloading for critical sounds

### Instrument Management
- ✅ Instrument reuse via caching
- ✅ Polyphonic pools reduce overhead
- ✅ Lazy loading - only create what's needed
- ✅ Proper disposal prevents memory leaks

### Scheduling
- ✅ Tone.Transport for precise timing
- ✅ Batch event scheduling
- ✅ Efficient chord handling
- ✅ Minimal CPU during playback

---

## Chord Support

### IR Format
```javascript
{
  tracks: [{
    notes: [
      { pitch: [60, 64, 67], duration: 1.0, velocity: 0.8 } // C major chord
    ]
  }]
}
```

### DSL Format
```
track("piano") {
  instrument("piano/grand_piano_k")
  chord(["C4", "E4", "G4"], 1.0, 0.8)
}
```

### Automatic Voice Allocation
The polyphonic pool handles simultaneous notes automatically - no manual voice management needed.

---

## Debugging Tips

### Enable Debug Mode
```javascript
import devToolkit from './DevToolkit.js';
devToolkit.enableDebug();
```

### Check Stats
```javascript
// Compiler stats
console.log(compiler.getStats());

// Cache stats
console.log(sampleCache.getStats());

// Factory stats
console.log(instrumentFactory.getStats());
```

### Profile Performance
```javascript
const profile = await devToolkit.profileInstrumentLoad(
  'piano/grand_piano_k',
  sampleCache,
  instrumentFactory
);
```

### Export Diagnostics
```javascript
devToolkit.exportDiagnostics(compiler);
// Downloads diagnostics-{timestamp}.json
```

---

## Next Steps

### Recommended Improvements

1. **Add Web Workers** - Offload sample processing
2. **Implement Streaming** - For very large instruments
3. **Add MIDI Support** - Export/import MIDI files
4. **Visual Editor** - GUI for composition
5. **Effects Chain** - Reverb, delay, compression
6. **Mix Down** - Multi-track mixing and mastering

### File Structure
```
project/
├── client/
│   ├── SampleCache.js
│   ├── InstrumentFactory.js
│   ├── MusicScheduler.js
│   ├── MusicCompiler.js
│   ├── MusicPlayer.js
│   └── DevToolkit.js
├── server/
│   ├── server-v2.js
│   ├── parser.js (legacy)
│   └── generator.js (legacy)
├── samples/
│   └── [instrument folders]
├── catalog.json
└── README.md
```

---

## License

Your existing license applies.

## Support

For issues or questions, check the DevToolkit diagnostics first, then review error logs.