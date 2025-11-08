# Week 7 (Nov 8–14): Production Deployment & Model Integration

## Team Overview

| Member | Role | Focus Areas |
|--------|------|-------------|
| **Max** | Deployment + Beatbox2drums | Production infrastructure, HPCC model integration, deployment optimization |
| **Carson** | Import/Export + Polish | MIDI/JSON import/export, file handling, UI refinements |
| **Ayaan** | Bug Fixes + Performance | Critical bug resolution, performance optimization, UX improvements |
| **Steve** | Instrument System + Testing | Audio bug fixes, sample pack expansion, integration testing |

**Critical Deadlines This Week:**
- **11/9:** Deployment stable, Beatbox2drums integrated
- **11/15:** Import/Export functional

**Strategy:** Focus on production readiness - deployment stability, model integration, and core features working reliably. This week bridges the gap between development features and production-ready product.

---

## MAX - Deployment & Beatbox2drums Integration

### Mission
Get the application deployed to a reliable production environment and integrate the beatbox2drums model from HPCC training. These are critical blockers for demo and user testing.

### Key Libraries
- Docker, docker-compose
- Cloud platform CLIs (Vercel, Railway, AWS)
- PyTorch model loading
- FastAPI deployment

---

### Task 1: Deployment Platform Evaluation
**Goal:** Choose optimal platform for production deployment

**Platforms to Test:**
1. **Vercel** (Frontend + Serverless API)
   - Pros: Free tier, great Next.js integration, fast CDN
   - Cons: Serverless function timeout limits, cold starts for ML models

2. **Railway** (Full Stack)
   - Pros: Simple deployment, Docker support, persistent processes
   - Cons: Pricing, resource limits on free tier

3. **AWS** (EC2 + S3 + CloudFront)
   - Pros: Full control, scalable, reliable
   - Cons: Complex setup, costs, requires DevOps knowledge

4. **Render** (Current - Performance Issues)
   - Re-evaluate: Can we optimize or is migration necessary?

**Testing Checklist:**
- [ ] Deploy test version to each platform
- [ ] Load test with audio processing requests
- [ ] Measure cold start times for ML inference
- [ ] Calculate estimated monthly costs
- [ ] Test CORS and networking between services
- [ ] Verify model checkpoint loading (135MB file)

**Deliverable:** `DEPLOYMENT_COMPARISON.md` with recommendation

---

### Task 2: Production Docker Optimization
**Goal:** Reduce image size and startup time

**Current Issues:**
- Large model checkpoints increasing image size
- Slow cold starts on free tiers
- Memory constraints on model loading

**Optimization Strategies:**
1. **Multi-stage Docker builds**
   - Separate build and runtime stages
   - Only include necessary files in final image

2. **Model checkpoint optimization**
   - Use Git LFS or external storage (S3, R2)
   - Download on first startup, cache locally
   - Consider model quantization if needed

3. **Python dependency optimization**
   - Remove unused packages
   - Use lightweight base image (python:3.11-slim)
   - Cache pip packages in Docker layer

**Files to Update:**
- `backend/Dockerfile`
- `runner/Dockerfile`
- `docker-compose.yml`
- `docker-compose.prod.yml` (new)

**Testing:** Measure image size before/after, cold start time

---

### Task 3: Environment Configuration Management
**Goal:** Proper environment variable handling for production

**Create Configuration Files:**
- `.env.example` - Template with all required variables
- `.env.production` - Production values (gitignored)
- `backend/config.py` - Centralized config loading

**Required Environment Variables:**
```bash
# Backend
BACKEND_URL=https://api.your-domain.com
RUNNER_URL=https://runner.your-domain.com
FRONTEND_URL=https://your-domain.com
MODEL_CHECKPOINT_PATH=backend/checkpoints/combined_hum2melody_full.pth
DATABASE_PATH=backend/training_data.db
ALLOWED_ORIGINS=https://your-domain.com

# Runner
BACKEND_URL=https://api.your-domain.com
CDN_BASE_URL=https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev

# Frontend
NEXT_PUBLIC_API_URL=https://api.your-domain.com
NEXT_PUBLIC_RUNNER_URL=https://runner.your-domain.com
```

**Validation:** Add startup checks to verify all required env vars are set

---

### Task 4: Beatbox2drums Model Integration from HPCC
**Goal:** Transfer trained model and integrate into backend API

**Steps:**

1. **Download Model from HPCC**
   - SSH into HPCC cluster
   - Locate trained checkpoint in `hpcc_training/checkpoints/`
   - Download via SCP or SFTP
   - Verify checkpoint integrity (file size, loadable)

2. **Create Model Loader**
   - File: `backend/models/beatbox2drums_loader.py`
   - Similar pattern to `combined_model_loader.py`
   - Handle checkpoint loading errors gracefully

3. **Create Inference Wrapper**
   - File: `backend/inference/beatbox_predictor.py`
   - Similar to `MelodyPredictor` pattern
   - Methods: `predict_from_audio()`, `predict_from_audio_RAW()`

4. **Update ModelServer**
   - File: `backend/model_server.py`
   - Add `beatbox_predictor` initialization
   - Add `predict_drums()` method
   - Handle model loading failures with mock fallback

5. **Update /beatbox2drums Endpoint**
   - File: `backend/main.py`
   - Connect to `model_server.predict_drums()`
   - Test with sample beatbox audio

**Testing Checklist:**
- [ ] Model loads without errors
- [ ] Inference produces valid IR format
- [ ] Endpoint returns 200 with valid response
- [ ] Audio preprocessing works correctly
- [ ] Results sound reasonable when played

**Deliverable:** Working `/beatbox2drums` endpoint with real model

---

### Task 5: Model Performance Optimization
**Goal:** Reduce inference latency and memory usage

**Optimization Techniques:**

1. **Model Quantization** (if needed)
   - Convert float32 to float16 or int8
   - Test accuracy trade-offs
   - Measure speedup

2. **Batch Size Optimization**
   - Find optimal batch size for inference
   - Balance between speed and memory

3. **CPU vs GPU Inference**
   - Test inference on CPU (production likely CPU-only)
   - Optimize for CPU if no GPU available
   - Consider ONNX conversion for faster CPU inference

4. **Preprocessing Optimization**
   - Cache CQT parameters
   - Optimize librosa operations
   - Consider parallel processing

**Metrics to Track:**
- Inference time (ms)
- Memory usage (MB)
- CPU utilization (%)
- Model size (MB)

**Target:** <2 seconds for hum2melody, <3 seconds for beatbox2drums

---

### Task 6: Health Checks and Monitoring
**Goal:** Add observability to production deployment

**Implementation:**

1. **Enhanced Health Endpoints**
   - `GET /health` - Basic health check
   - `GET /health/detailed` - Include model status, disk space, memory
   - `GET /metrics` - Prometheus-compatible metrics (optional)

2. **Logging Configuration**
   - Use Python `logging` module consistently
   - Log levels: DEBUG (dev), INFO (prod)
   - Include timestamps, request IDs
   - File: `backend/logging_config.py`

3. **Error Tracking**
   - Add error context (user action, inputs)
   - Log stack traces for 500 errors
   - Consider Sentry integration (optional)

4. **Performance Monitoring**
   - Log inference times
   - Track endpoint response times
   - Monitor model loading success/failure

---

### Task 7: Database Migration for Production
**Goal:** Prepare SQLite database for production use

**Tasks:**

1. **Database Initialization Script**
   - File: `backend/init_db.py`
   - Creates tables if not exist
   - Handles migrations safely
   - Run on startup

2. **Backup Strategy**
   - Periodic SQLite backups to S3/R2
   - Retention policy (keep last 7 days)
   - Restoration documentation

3. **Data Privacy Considerations**
   - Audio upload storage duration
   - User data handling
   - GDPR compliance (if applicable)

**Note:** If scaling beyond SQLite, consider PostgreSQL migration path

---

### Task 8: Production Deployment
**Goal:** Deploy to chosen platform and verify stability

**Deployment Steps:**

1. **Frontend Deployment**
   - Build optimized production bundle
   - Configure API URLs
   - Deploy to platform
   - Test static asset loading

2. **Backend Deployment**
   - Push Docker image to registry
   - Configure environment variables
   - Deploy with health checks
   - Verify model checkpoint accessible

3. **Runner Deployment**
   - Deploy Node.js service
   - Configure CDN URLs
   - Test DSL compilation

4. **DNS and SSL**
   - Configure custom domain (if available)
   - Enable HTTPS
   - Update CORS policies

**Smoke Testing:**
- [ ] All services return 200 on /health
- [ ] Frontend loads and displays UI
- [ ] Audio recording works
- [ ] Hum2melody generates valid output
- [ ] Beatbox2drums generates valid output
- [ ] Playback works with generated audio
- [ ] No console errors

---

### Task 9: Load Testing
**Goal:** Verify production can handle concurrent users

**Testing Approach:**

1. **Tools:** Apache Bench, k6, or Artillery

2. **Test Scenarios:**
   - 10 concurrent audio uploads
   - 50 requests/second to /health
   - Large audio file (10MB)
   - Sustained load for 5 minutes

3. **Metrics to Monitor:**
   - Response times (p50, p95, p99)
   - Error rate
   - CPU/Memory usage
   - Request throughput

**Document:** Performance limits and recommended scaling thresholds

---

### Task 10: Deployment Documentation
**Goal:** Document deployment process and operations

**Files to Create:**

1. **`DEPLOYMENT.md`**
   - Deployment instructions for each platform
   - Environment variable reference
   - Troubleshooting common issues

2. **`OPERATIONS.md`**
   - How to check service health
   - How to view logs
   - How to deploy updates
   - Rollback procedures

3. **`ARCHITECTURE.md`** (Update)
   - Production architecture diagram
   - Service dependencies
   - Data flow

---

### Integration Points
- **Model checkpoints must be accessible in production**
- **Environment variables consistent across services**
- **CORS configured for production URLs**
- **Database persists across deployments**

### Key Decisions
1. Which platform to use for production?
2. Model checkpoint storage strategy (bundle vs download)?
3. Scaling strategy for future growth?
4. Cost vs performance trade-offs?

---

## CARSON - Import/Export & UI Polish

### Mission
Enable users to import/export projects in standard formats (MIDI, JSON) and polish the editing features from Week 6. This makes the tool interoperable with other DAWs and ensures professional feel.

### Key Libraries
- `@tonejs/midi` for MIDI parsing/generation
- File I/O (FileReader, Blob, URL.createObjectURL)
- JSON schema validation

---

### Task 1: Project JSON Export
**Goal:** Save complete project state to JSON file

**Data Structure:**
```typescript
interface ProjectFile {
  version: string;           // "1.0.0"
  created: string;           // ISO timestamp
  modified: string;          // ISO timestamp
  metadata: {
    title: string;
    tempo: number;
    key: string;
    timeSignature: string;
  };
  tracks: Track[];
  settings: {
    loopEnabled: boolean;
    loopStart: number;
    loopEnd: number;
  };
  dsl: string;               // Raw DSL code
}
```

**Implementation:**

1. **Export Function**
   - File: `frontend/src/lib/export.ts`
   - Function: `exportProject(data: ProjectFile): void`
   - Generate JSON with indentation
   - Create Blob and trigger download
   - Filename: `{title}_{timestamp}.maa` (Music AI Assistant)

2. **UI Integration**
   - Add "Export Project" button
   - Use download icon (Lucide: Download)
   - Show toast notification on success

**Testing:**
- [ ] Exported JSON is valid
- [ ] File downloads correctly
- [ ] All project data included

---

### Task 2: Project JSON Import
**Goal:** Load saved project files

**Implementation:**

1. **Import Function**
   - File: `frontend/src/lib/import.ts`
   - Function: `importProject(file: File): Promise<ProjectFile>`
   - Parse JSON with error handling
   - Validate schema
   - Return typed project data

2. **UI Integration**
   - Add "Import Project" button (Upload icon)
   - File input accepting `.maa` and `.json`
   - Show loading state during parse
   - Restore all state (code, tempo, tracks, loop settings)
   - Confirm before overwriting current project

3. **Error Handling**
   - Invalid JSON format
   - Missing required fields
   - Version incompatibility
   - Corrupted file data

**Testing:**
- [ ] Valid files load correctly
- [ ] Invalid files show error
- [ ] All state restored properly
- [ ] UI updates after import

---

### Task 3: MIDI Export
**Goal:** Export tracks to standard MIDI format

**Implementation:**

1. **IR to MIDI Conversion**
   - File: `frontend/src/lib/midi-export.ts`
   - Function: `exportToMIDI(ir: IR): Uint8Array`
   - Use `@tonejs/midi` library
   - Convert tracks to MIDI tracks
   - Set tempo and time signature
   - Handle percussion track (channel 10)

2. **Mapping:**
   - IR Note → MIDI Note
     - `pitch` → MIDI note number
     - `start` → MIDI ticks
     - `duration` → MIDI ticks
     - `velocity` → MIDI velocity (0-127)
   - IR Track → MIDI Track
     - Track name from `id`
     - Program change for instrument (map to General MIDI)

3. **General MIDI Instrument Mapping**
   - Piano → Program 0 (Acoustic Grand Piano)
   - Bass → Program 32-39 (Bass instruments)
   - Drums → Channel 10 (Standard MIDI percussion)
   - Guitar → Program 24-31
   - Default: Program 0

4. **UI Integration**
   - Add "Export MIDI" button
   - Export current IR to MIDI file
   - Filename: `{title}.mid`

**Testing:**
- [ ] MIDI file opens in DAW (GarageBand, Logic, Ableton)
- [ ] Notes play at correct pitch and timing
- [ ] Tempo preserved
- [ ] Multiple tracks separated correctly

---

### Task 4: MIDI Import
**Goal:** Import MIDI files into the application

**Implementation:**

1. **MIDI to IR Conversion**
   - File: `frontend/src/lib/midi-import.ts`
   - Function: `importFromMIDI(file: File): Promise<IR>`
   - Parse MIDI using `@tonejs/midi`
   - Extract tempo from MIDI header
   - Convert MIDI tracks to IR tracks
   - Convert MIDI notes to IR notes

2. **Mapping Challenges:**
   - MIDI ticks → seconds (use PPQ and tempo)
   - MIDI velocity (0-127) → IR velocity (0-1)
   - Program changes → instrument selection
   - Multiple tempo changes → use first or average?

3. **UI Integration**
   - Add "Import MIDI" button
   - File input accepting `.mid`, `.midi`
   - Show track preview before import
   - Option to merge with current project or replace

4. **Edge Cases:**
   - MIDI files with no notes
   - Very dense MIDI (1000+ notes)
   - MIDI Type 2 files (multiple sequences)
   - Non-standard MIDI events

**Testing:**
- [ ] Import simple MIDI file
- [ ] Notes appear in piano roll
- [ ] Playback sounds correct
- [ ] Tempo matches original
- [ ] Handle large MIDI files (>10k notes)

---

### Task 5: Audio Export (WAV/MP3)
**Goal:** Export rendered audio to file

**Implementation:**

1. **Tone.js Recording**
   - Use `Tone.Recorder` to capture output
   - Record playback to WAV blob
   - Trigger download

2. **Function:**
   - File: `frontend/src/lib/audio-export.ts`
   - Function: `exportAudio(ir: IR): Promise<Blob>`
   - Start transport
   - Record to WAV
   - Stop after duration
   - Return Blob

3. **UI Integration**
   - Add "Export Audio" button (Music icon)
   - Show recording progress
   - Estimate duration from IR
   - Download WAV file

4. **Optional:** MP3 conversion
   - Use lamejs or Web Audio API
   - Requires client-side encoding (slower)

**Testing:**
- [ ] WAV file plays in media player
- [ ] Audio matches playback
- [ ] Full duration captured
- [ ] Stereo/mono correct

---

### Task 6: Editing Feature Polish from Week 6
**Goal:** Refine undo/redo, multi-select, copy/paste

**Bug Fixes and Improvements:**

1. **Undo/Redo Refinements**
   - Debounce history pushes (don't save every keystroke)
   - Limit history size (max 50 states)
   - Clear redo stack on new action
   - Show undo/redo in UI (grayed when unavailable)

2. **Multi-Select Improvements**
   - Box select visual feedback (dotted border)
   - Show selection count ("5 notes selected")
   - Deselect all with Escape key
   - Invert selection option (Ctrl+I)

3. **Copy/Paste Enhancements**
   - Paste at mouse position (not just playhead)
   - Duplicate with Ctrl+D (copy + paste immediately)
   - Paste relative to original timing
   - Visual feedback during paste

4. **Velocity Editor Tweaks**
   - Velocity values displayed on hover
   - Percentage display (0-100%)
   - Gradient fill for visual appeal
   - Sync scroll with piano roll

---

### Task 7: Keyboard Shortcuts Improvements
**Goal:** Complete and polish keyboard shortcut system

**Shortcuts to Implement:**
```
Ctrl+Z       - Undo
Ctrl+Y       - Redo
Ctrl+C       - Copy
Ctrl+V       - Paste
Ctrl+X       - Cut
Ctrl+A       - Select All
Ctrl+D       - Duplicate
Delete       - Delete selected
Escape       - Deselect / Cancel
Space        - Play/Pause
Ctrl+S       - Save Project
Ctrl+O       - Open Project
Ctrl+E       - Export MIDI
Ctrl+Shift+E - Export Audio
?            - Show shortcuts
```

**Implementation:**
- Centralize keyboard handling in custom hook
- Prevent default browser shortcuts
- Display shortcuts in modal (? key)
- Visual hints in UI (show shortcut in tooltip)

---

### Task 8: File Format Documentation
**Goal:** Document all import/export formats

**File:** `frontend/FILE_FORMATS.md`

**Content:**
- **JSON Project Format** - Full schema with examples
- **MIDI Format** - What's supported, limitations
- **Audio Export** - Formats, sample rates, channels
- **Version Compatibility** - How we handle format changes

**Include:** Example files for each format

---

### Task 9: Import/Export UI Component
**Goal:** Create unified import/export interface

**Component:** `frontend/src/components/FileMenu.tsx`

**Features:**
- Dropdown menu with all import/export options
- Icons for each file type
- Keyboard shortcuts displayed
- Recent files list (localStorage)
- "New Project" option (confirm if unsaved changes)

**Design:**
- Clean, organized layout
- Group by import vs export
- Visual file type icons
- Tooltips explaining each option

---

### Task 10: Testing and Documentation
**Goal:** Comprehensive testing of all file operations

**Test Matrix:**
| Operation | Format | Test Case | Expected Result |
|-----------|--------|-----------|-----------------|
| Export | JSON | Valid project | Downloads .maa file |
| Import | JSON | Valid file | Restores project |
| Import | JSON | Invalid file | Shows error |
| Export | MIDI | Single track | Opens in DAW |
| Export | MIDI | Multi-track | Tracks separated |
| Import | MIDI | Simple melody | Displays in piano roll |
| Export | WAV | Full mix | Plays correctly |

**Deliverables:**
- All test cases passing
- Example files for documentation
- User guide section in README

---

### Integration Points
- **File operations integrate with Steve's save/load**
- **MIDI import creates tracks (uses Steve's track management)**
- **Export uses current IR from playback state**
- **Keyboard shortcuts work with Ayaan's UX improvements**

### Key Decisions
1. Should MIDI import merge or replace current project?
2. Audio export: WAV only or support MP3?
3. Project file extension: `.maa`, `.json`, or both?
4. Version compatibility strategy for future format changes?

---

## AYAAN - Critical Bug Fixes & Performance

### Mission
Eliminate critical bugs, optimize performance for production, and ensure the application feels responsive and professional. Focus on issues that would block demo or user testing.

### Key Tools
- Browser DevTools (Performance, Memory, Network tabs)
- React DevTools Profiler
- Console monitoring
- User testing feedback

---

### Task 1: Critical Bug Triage
**Goal:** Identify and prioritize all blocking issues

**Process:**

1. **Bug Inventory**
   - Review `frontend/BUGS.md` (if exists from Week 6)
   - Manual testing of all workflows
   - Check browser console for errors
   - Test on different browsers

2. **Severity Classification:**
   - **P0 (CRITICAL):** Blocks demo, crashes app, data loss
   - **P1 (HIGH):** Major UX issue, workaround exists
   - **P2 (MEDIUM):** Minor annoyance, edge case
   - **P3 (LOW):** Polish, nice-to-have

3. **Known Issues to Check:**
   - Audio recording failures
   - Model inference errors
   - Playback synchronization issues
   - UI rendering glitches
   - Memory leaks
   - State management bugs

**Deliverable:** Updated `BUGS.md` with priorities and assignments

---

### Task 2: Audio Recording Bug Fixes
**Goal:** Ensure reliable audio capture on all browsers

**Common Issues:**

1. **Microphone Permission Denied**
   - Add clear permission request UI
   - Show helpful error message
   - Test permission state handling

2. **Audio Not Recording**
   - Check MediaRecorder API support
   - Verify audio stream active
   - Test different browsers

3. **Recording Stops Prematurely**
   - Check event handlers
   - Verify stop conditions
   - Test long recordings (>30s)

4. **Audio Quality Issues**
   - Verify sample rate settings
   - Check for clipping
   - Test with different microphones

**Testing Matrix:**
| Browser | Record 5s | Record 30s | Permission | Format |
|---------|-----------|------------|------------|--------|
| Chrome  | ✓ | ✓ | ✓ | ✓ |
| Firefox | ✓ | ✓ | ✓ | ✓ |
| Safari  | ? | ? | ? | ? |
| Edge    | ✓ | ✓ | ✓ | ✓ |

---

### Task 3: Piano Roll Performance Optimization
**Goal:** Smooth interaction with 500+ notes

**Performance Issues:**

1. **Rendering Lag**
   - Profile canvas rendering with DevTools
   - Implement dirty rectangle rendering (only redraw changed areas)
   - Debounce mouse move events
   - Use `requestAnimationFrame` for smooth updates

2. **Note Selection Lag**
   - Optimize hit detection algorithm
   - Use spatial indexing for large note counts
   - Cache calculations

3. **Zoom/Pan Performance**
   - Throttle zoom events
   - Batch canvas redraws
   - Optimize coordinate transformations

**Optimization Techniques:**
- Offscreen canvas for static elements
- Reduce redraw frequency
- Use Web Workers for heavy calculations (if needed)

**Target Performance:**
- 60 FPS during interaction
- No lag with 1000+ notes
- Smooth zoom/pan

---

### Task 4: Memory Leak Detection
**Goal:** Prevent memory growth during extended use

**Testing Process:**

1. **Memory Profiling**
   - Use Chrome DevTools Memory tab
   - Record heap snapshots
   - Identify detached DOM nodes
   - Check for retained event listeners

2. **Common Leak Sources:**
   - Tone.js objects not disposed
   - Canvas contexts not cleared
   - Event listeners not removed
   - Intervals/timeouts not cleared
   - Large arrays accumulating

3. **Test Scenario:**
   - Play/stop 50 times
   - Create/delete 100 notes
   - Import/export 10 times
   - Check memory growth

**Fix Pattern:**
- Use `useEffect` cleanup functions
- Dispose Tone.js objects: `synth.dispose()`
- Remove event listeners on unmount
- Clear intervals/timeouts

---

### Task 5: State Management Cleanup
**Goal:** Ensure consistent and predictable state updates

**Issues to Fix:**

1. **State Synchronization**
   - DSL code vs IR representation
   - Piano roll vs code editor
   - Playhead position tracking
   - Track mute/solo state

2. **Race Conditions**
   - Async state updates
   - Multiple simultaneous API calls
   - Playback state changes

3. **State Reset Issues**
   - New project doesn't clear all state
   - Import doesn't fully reset
   - Transport state persists incorrectly

**Approach:**
- Centralize state updates
- Use reducer pattern for complex state
- Add state validation
- Document state flow

---

### Task 6: Error Handling Improvements
**Goal:** Graceful failure with helpful messages

**Error Categories:**

1. **Network Errors**
   ```
   ❌ "Failed to connect"
   ✅ "Cannot reach backend server. Check that all services are running."
   ```

2. **Audio Processing Errors**
   ```
   ❌ "Error 500"
   ✅ "Audio processing failed. Try a shorter recording (max 30 seconds)."
   ```

3. **Model Inference Errors**
   ```
   ❌ "Prediction failed"
   ✅ "Could not generate melody. Make sure you're humming clearly."
   ```

4. **Playback Errors**
   ```
   ❌ "Transport error"
   ✅ "Playback failed. Check that samples are loaded."
   ```

**Implementation:**
- Catch errors at API boundary
- Map error codes to user messages
- Show actionable next steps
- Log technical details to console

---

### Task 7: Loading State Consistency
**Goal:** Visual feedback for all async operations

**Components Needing Loading States:**

1. **Audio Processing**
   - Show spinner during upload
   - Display progress (if possible)
   - Disable button during processing
   - Timeout after 30 seconds

2. **DSL Compilation**
   - Show "Compiling..." state
   - Disable play during compilation
   - Handle compilation errors

3. **Project Import**
   - Show loading during file parse
   - Disable UI during state restoration

4. **Instrument Loading**
   - Show sample loading progress
   - Display which instruments are ready

**UI Pattern:**
```tsx
{isLoading ? (
  <button disabled>
    <Spinner /> Processing...
  </button>
) : (
  <button onClick={handleClick}>
    Generate Melody
  </button>
)}
```

---

### Task 8: UI Responsiveness Testing
**Goal:** Ensure app works on different screen sizes

**Breakpoints to Test:**
- 1920x1080 (desktop)
- 1366x768 (laptop)
- 1024x768 (small laptop)
- 768x1024 (tablet portrait)
- 375x667 (mobile - stretch goal)

**Components to Check:**
- Code editor resizing
- Piano roll usability
- Mixer panel layout
- Timeline visibility
- Button spacing

**Issues to Fix:**
- Text overflow
- Overlapping elements
- Unreadable font sizes
- Inaccessible controls

---

### Task 9: Accessibility Improvements
**Goal:** Make app usable with keyboard only

**Keyboard Navigation:**
- [ ] Tab through all controls
- [ ] Enter/Space activates buttons
- [ ] Escape closes modals
- [ ] Arrow keys navigate piano roll
- [ ] Focus indicators visible

**Screen Reader Support:**
- Add ARIA labels to controls
- Provide text alternatives for icons
- Announce state changes
- Label form inputs

**Visual Accessibility:**
- Sufficient color contrast
- No color-only indicators
- Focus outlines visible
- Text readable at 125% zoom

---

### Task 10: Performance Metrics Dashboard
**Goal:** Track app performance over time

**Metrics to Collect:**

1. **API Response Times**
   - /hum2melody average latency
   - /beatbox2drums average latency
   - /run compilation time

2. **Client Performance**
   - Time to interactive
   - First contentful paint
   - Piano roll FPS
   - Memory usage over time

3. **User Actions**
   - Notes created per session
   - Playback frequency
   - Error rate

**Implementation:**
- Log metrics to console (dev)
- Send to backend (optional)
- Create `PERFORMANCE.md` report

**Deliverable:** Performance baseline for future optimization

---

### Task 11: Cross-Browser Testing
**Goal:** Verify compatibility on major browsers

**Test Workflow on Each Browser:**
1. Record audio → Generate melody
2. Edit notes in piano roll
3. Play/pause/stop
4. Export project
5. Import project

**Browsers:**
- Chrome (primary)
- Firefox
- Safari (if Mac available)
- Edge

**Document:**
- Feature support matrix
- Browser-specific bugs
- Workarounds needed

---

### Task 12: Production Readiness Checklist
**Goal:** Ensure app is demo-ready

**Checklist:**
- [ ] No console errors on load
- [ ] No console errors during usage
- [ ] All API calls succeed
- [ ] Loading states on all async operations
- [ ] Error messages are helpful
- [ ] No broken UI elements
- [ ] Tooltips on all controls
- [ ] Keyboard shortcuts work
- [ ] Performance acceptable (no lag)
- [ ] Memory stable (no leaks)
- [ ] Works in Chrome and Firefox
- [ ] Mobile layout acceptable (stretch)

---

### Integration Points
- **Test Max's deployment thoroughly**
- **Test Carson's import/export features**
- **Test Steve's instrument system improvements**
- **Coordinate bug fixes with team**

### Key Decisions
1. Which bugs are blockers for 11/9 deadline?
2. What's the minimum acceptable performance?
3. Which browsers are must-support vs nice-to-have?
4. When to stop fixing bugs and ship?

---

## STEVE - Instrument System & Integration Testing

### Mission
Fix audio bugs in instrument system, expand sample library, and perform comprehensive integration testing to ensure all features work together seamlessly.

### Key Libraries
- Tone.js audio engine
- Sample loading and caching
- Integration testing tools

---

### Task 1: Audio Bug Investigation
**Goal:** Identify and document all audio-related issues

**Known Issues from Project Status:**
- Audio loading failures
- Playback glitches
- Sample missing errors
- Volume/mixing problems

**Investigation Steps:**

1. **Sample Loading Errors**
   - Check CDN accessibility
   - Verify `mapping.json` files
   - Test loading across all instruments
   - Check for 404 errors in Network tab

2. **Playback Issues**
   - Audio dropouts
   - Timing synchronization
   - Clicks/pops
   - Stuck notes

3. **Mixing Problems**
   - Volume inconsistencies
   - Panning issues
   - Mute/solo not working
   - Clipping/distortion

**Deliverable:** `AUDIO_BUGS.md` with reproduction steps

---

### Task 2: Sample Cache Improvements
**Goal:** Reliable and efficient sample loading

**File:** `runner/SampleCache.js`

**Improvements:**

1. **Retry Logic**
   - Retry failed loads 3 times
   - Exponential backoff
   - Fallback to default instrument

2. **Loading Progress**
   - Track samples loaded vs total
   - Expose progress API
   - Frontend can show loading bar

3. **Error Handling**
   - Detailed error messages
   - Log missing samples
   - Graceful degradation

4. **Preloading Strategy**
   - Preload common instruments (piano, bass, drums)
   - Lazy load specialty instruments
   - Cancel unnecessary loads

**Testing:**
- [ ] All instruments load successfully
- [ ] Failed loads retry automatically
- [ ] Progress tracking accurate
- [ ] Error messages helpful

---

### Task 3: Instrument Mapping Verification
**Goal:** Ensure all instruments have valid CDN mappings

**Process:**

1. **Audit Existing Instruments**
   - List all instruments referenced in codebase
   - Check CDN for each instrument folder
   - Verify `mapping.json` format
   - Test sample paths

2. **Missing Instruments**
   - Document which instruments lack samples
   - Find or create alternatives
   - Update documentation

3. **Mapping.json Schema**
   ```json
   {
     "baseUrl": "https://cdn.../instrument/category/",
     "samples": {
       "C3": ["v1.mp3", "v2.mp3", "v3.mp3"],
       "D3": ["v1.mp3", "v2.mp3", "v3.mp3"]
     }
   }
   ```

**Deliverable:** Complete instrument inventory in `INSTRUMENTS.md`

---

### Task 4: Sample Pack Expansion
**Goal:** Add more instrument options for variety

**Priority Instruments:**
1. **Percussion**
   - Kick, snare, hi-hat variants
   - Cymbals, toms
   - Electronic drum sounds

2. **Bass**
   - Synth bass
   - Acoustic bass
   - Sub bass

3. **Melodic**
   - Electric piano
   - Strings (violin, cello)
   - Brass (trumpet, sax)

4. **Textures**
   - Pads
   - Ambient sounds
   - Effects

**Sources:**
- Free sample libraries (Freesound, LMMS)
- Record custom samples
- Use synthesis (Tone.js built-in)

**Process:**
1. Find/create samples
2. Organize into velocity layers
3. Upload to CDN (Cloudflare R2)
4. Create `mapping.json`
5. Test in application

---

### Task 5: Volume and Mixing Fixes
**Goal:** Consistent audio levels across instruments

**Issues to Fix:**

1. **Volume Normalization**
   - Analyze peak levels of all samples
   - Normalize to -6dB
   - Apply gain compensation

2. **Mixer Panel Improvements**
   - File: `frontend/src/components/MixerPanel.tsx`
   - Add VU meters (visual level indicators)
   - Add pan controls
   - Add solo/mute indicators

3. **Default Mix Settings**
   - Set sensible default volumes
   - Drums: 0.8
   - Melody: 0.9
   - Bass: 0.7
   - Accompaniment: 0.6

4. **Master Output**
   - Add master volume control
   - Add limiter to prevent clipping
   - Monitor master level

**Testing:**
- [ ] No clipping at default levels
- [ ] All instruments audible
- [ ] Consistent loudness
- [ ] Mute/solo work correctly

---

### Task 6: Tone.js Resource Management
**Goal:** Prevent audio context issues and resource leaks

**Issues:**

1. **AudioContext Suspension**
   - Handle browser autoplay policies
   - Resume context on user interaction
   - Show "Click to enable audio" message

2. **Instrument Disposal**
   - Dispose unused Tone.js instruments
   - Clear sample buffers
   - Release audio nodes

3. **Transport Cleanup**
   - Stop transport on page unload
   - Clear scheduled events
   - Dispose all synths

**Pattern:**
```typescript
useEffect(() => {
  const synth = new Tone.Sampler(...);

  return () => {
    synth.dispose(); // Cleanup
  };
}, []);
```

---

### Task 7: Integration Testing Framework
**Goal:** Automated testing of critical workflows

**Test Workflows:**

1. **End-to-End: Hum to Playback**
   - Upload audio file
   - Call /hum2melody
   - Receive IR
   - Compile DSL
   - Play audio
   - Verify sound output

2. **End-to-End: Edit and Export**
   - Load project
   - Edit notes
   - Export MIDI
   - Verify MIDI file

3. **Multi-Track Playback**
   - Create 5 tracks
   - Play simultaneously
   - Verify synchronization

**Implementation:**
- Use Playwright or Cypress for E2E tests
- Or manual testing with detailed checklist
- Document test results

---

### Task 8: Sample Playback Testing
**Goal:** Verify all instruments play correctly

**Test Matrix:**

| Instrument | Loads | Plays | Correct Pitch | Velocity Layers |
|------------|-------|-------|---------------|-----------------|
| Piano (Grand) | ✓ | ✓ | ✓ | ✓ |
| Bass (Acoustic) | ? | ? | ? | ? |
| Drums (Kit) | ? | ? | N/A | ? |
| Strings | ? | ? | ? | ? |
| Synth | ? | ? | ? | ? |

**For Each Instrument:**
1. Load in app
2. Play low note (C2)
3. Play middle note (C4)
4. Play high note (C6)
5. Test soft velocity (0.3)
6. Test loud velocity (0.9)
7. Verify pitch accuracy
8. Check for artifacts

---

### Task 9: Performance Testing with Multiple Tracks
**Goal:** Ensure smooth playback with complex arrangements

**Test Scenarios:**

1. **10 Tracks, 50 Notes Each**
   - Create project
   - Play from start
   - Monitor audio dropouts
   - Check CPU usage

2. **Dense Polyphony**
   - 20 simultaneous notes
   - Verify no voice stealing issues
   - Check for clicks/pops

3. **Long Playback**
   - 5-minute project
   - Continuous playback
   - Monitor memory growth
   - Check for drift/sync issues

**Metrics:**
- CPU usage < 50%
- No audio dropouts
- No drift over time
- Memory stable

---

### Task 10: Documentation Updates
**Goal:** Complete and accurate audio system documentation

**Files to Update:**

1. **`INSTRUMENTS.md`** (New)
   - List of all available instruments
   - CDN paths
   - How to add new instruments
   - Troubleshooting guide

2. **`AUDIO_ARCHITECTURE.md`** (New)
   - How Tone.js is integrated
   - Sample loading flow
   - Mixing architecture
   - Transport management

3. **`TROUBLESHOOTING.md`** (Update)
   - Common audio issues
   - Browser compatibility
   - Permission problems
   - Performance tips

---

### Task 11: User Acceptance Testing
**Goal:** Validate with real users (team members)

**UAT Process:**

1. **Test Participants**
   - Ask Carson, Ayaan, Max to test
   - Provide test script
   - Collect feedback

2. **Test Script:**
   - Record hummed melody
   - Generate melody
   - Add beatbox drums
   - Edit in piano roll
   - Change instruments
   - Adjust mixing
   - Export MIDI
   - Rate experience 1-5

3. **Feedback Collection**
   - What worked well?
   - What was confusing?
   - Any bugs encountered?
   - Feature requests?

**Deliverable:** UAT feedback report

---

### Task 12: Integration Testing Checklist
**Goal:** Verify all features work together

**Checklist:**

**Frontend ↔ Backend:**
- [ ] Audio upload works
- [ ] Hum2melody returns valid IR
- [ ] Beatbox2drums returns valid IR
- [ ] Error handling works

**Backend ↔ Runner:**
- [ ] IR forwarding works
- [ ] DSL compilation succeeds
- [ ] Compiled code executes

**Frontend ↔ Runner:**
- [ ] Direct DSL compilation
- [ ] Sample loading works
- [ ] CDN accessible

**Feature Interactions:**
- [ ] Import project → Play works
- [ ] Edit notes → Export MIDI preserves edits
- [ ] Undo → Playback reflects changes
- [ ] Loop → Playback loops correctly
- [ ] Tempo change → Playback tempo updates

---

### Integration Points
- **Test Max's deployed services**
- **Test Carson's import/export with different instruments**
- **Coordinate with Ayaan on performance issues**
- **Validate all features work in production environment**

### Key Decisions
1. Which instrument sample packs are priorities?
2. Audio quality vs file size trade-offs?
3. When to use Tone.js synthesis vs samples?
4. How to handle missing instruments gracefully?

---

## Deliverables Summary

### Max (Deployment & Beatbox)
- Production deployment on chosen platform (Vercel/Railway/AWS)
- `DEPLOYMENT_COMPARISON.md` - Platform evaluation
- `DEPLOYMENT.md` - Deployment instructions
- `OPERATIONS.md` - Operations guide
- `docker-compose.prod.yml` - Production Docker config
- Beatbox2drums model integrated and working
- `backend/models/beatbox2drums_loader.py`
- `backend/inference/beatbox_predictor.py`
- Performance metrics documented

### Carson (Import/Export & Polish)
- `frontend/src/lib/export.ts` - Project JSON export
- `frontend/src/lib/import.ts` - Project JSON import
- `frontend/src/lib/midi-export.ts` - MIDI export
- `frontend/src/lib/midi-import.ts` - MIDI import
- `frontend/src/lib/audio-export.ts` - WAV export
- `frontend/src/components/FileMenu.tsx` - File operations UI
- `frontend/FILE_FORMATS.md` - Format documentation
- Polished editing features from Week 6
- Example project and MIDI files

### Ayaan (Bug Fixes & Performance)
- `BUGS.md` - Updated with current status
- `PERFORMANCE.md` - Performance metrics report
- All P0/P1 bugs fixed
- Memory leaks eliminated
- Loading states on all operations
- Error messages improved
- Cross-browser compatibility verified
- Production readiness checklist complete

### Steve (Instruments & Testing)
- `AUDIO_BUGS.md` - Audio issues documented
- `INSTRUMENTS.md` - Complete instrument inventory
- `AUDIO_ARCHITECTURE.md` - Audio system docs
- Sample cache improvements in `runner/SampleCache.js`
- New sample packs uploaded to CDN
- Mixer panel improvements in `MixerPanel.tsx`
- Integration testing results
- UAT feedback report

---

## Success Metrics

### Minimum Viable (Required for 11/9 Deadline)
- ✓ Application deployed to production and accessible via URL
- ✓ Hum2melody works end-to-end in production
- ✓ Beatbox2drums model integrated and functional
- ✓ No P0 bugs (app doesn't crash)
- ✓ Basic import/export working (JSON projects)
- ✓ All instruments load and play
- ✓ Performance acceptable (no major lag)

### Target (Strong Progress for 11/15)
- ✓ Stable production deployment with <2s load time
- ✓ MIDI import/export fully functional
- ✓ Audio export (WAV) working
- ✓ All P1 bugs fixed
- ✓ Memory usage stable
- ✓ Expanded sample library (10+ instruments)
- ✓ Cross-browser tested (Chrome, Firefox, Edge)
- ✓ Comprehensive documentation complete
- ✓ Integration testing passing
- ✓ Performance metrics baseline established

### Stretch Goals
- ✓ MP3 export working
- ✓ Mobile layout functional
- ✓ Automated E2E tests
- ✓ Performance optimizations (60 FPS piano roll)
- ✓ Advanced mixing features (EQ, reverb)
- ✓ User analytics/telemetry
- ✓ Model inference <1s on production
- ✓ CDN optimization (edge caching)

---

## Week 7 Schedule

### Days 1-2 (Nov 8-9) - CRITICAL DEADLINE
**Focus:** Deployment + Beatbox Integration

**Max:**
- Day 1: Platform evaluation and selection
- Day 2: Production deployment and beatbox model integration

**Carson:**
- Day 1: Project JSON export/import
- Day 2: Testing import/export, bug fixes

**Ayaan:**
- Day 1: Critical bug triage and P0 fixes
- Day 2: Audio recording bugs, error handling

**Steve:**
- Day 1: Audio bug investigation
- Day 2: Sample cache improvements, testing

**Team Meeting:** End of Day 2 - Verify 11/9 deadline met

---

### Days 3-4 (Nov 10-11) - Feature Completion
**Focus:** Import/Export + Testing

**Max:**
- Day 3: Deployment optimization, monitoring
- Day 4: Performance testing, documentation

**Carson:**
- Day 3: MIDI export/import
- Day 4: Audio export, polish editing features

**Ayaan:**
- Day 3: Performance optimization (piano roll)
- Day 4: Memory leak fixes, state management

**Steve:**
- Day 3: Sample pack expansion
- Day 4: Mixing improvements, volume fixes

---

### Days 5-6 (Nov 12-13) - Integration & Polish
**Focus:** Testing + Documentation

**Max:**
- Day 5: Load testing, monitoring setup
- Day 6: Final documentation, ops guide

**Carson:**
- Day 5: File menu UI, keyboard shortcuts
- Day 6: Format documentation, testing

**Ayaan:**
- Day 5: Cross-browser testing
- Day 6: Production readiness checklist

**Steve:**
- Day 5: Integration testing
- Day 6: UAT with team, documentation

---

### Day 7 (Nov 14) - Final Review
**All Team:**
- Integration testing together
- Demo rehearsal
- Documentation review
- Plan Week 8 (Arranger focus)

---

## Communication Protocol

### Daily Standups (15 min)
- **Time:** 10 AM
- **Format:** Async or sync (team preference)
- **Share:**
  - What I did yesterday
  - What I'm doing today
  - Any blockers

### Status Updates
- Update task status in shared doc
- Mark tasks complete when done
- Flag blockers immediately

### Code Reviews
- All PRs reviewed within 4 hours
- Focus on integration points
- Test locally before approving

### Deployment Coordination
- Announce before deploying
- Verify health checks pass
- Monitor for errors post-deploy

---

## Risks and Mitigation

### Risk 1: Deployment Platform Issues
**Impact:** High - Blocks demo
**Likelihood:** Medium
**Mitigation:**
- Evaluate multiple platforms early
- Have fallback ready (keep Render working)
- Start migration Day 1

### Risk 2: Beatbox Model Integration Fails
**Impact:** Medium - Feature missing but not critical
**Likelihood:** Low
**Mitigation:**
- Test model loading locally first
- Have mock fallback ready
- Allocate full day for debugging

### Risk 3: Performance Issues in Production
**Impact:** High - Poor user experience
**Likelihood:** Medium
**Mitigation:**
- Load test early
- Optimize Docker images
- Consider CDN for static assets

### Risk 4: Import/Export Format Bugs
**Impact:** Low - Can iterate on format
**Likelihood:** Medium
**Mitigation:**
- Start with simple format
- Version the format
- Test with edge cases

### Risk 5: Team Member Blocked
**Impact:** Medium - Work delays
**Likelihood:** Low
**Mitigation:**
- Daily standups catch blockers
- Cross-functional knowledge sharing
- Help each other actively

---

## Key Research Questions

1. **Deployment:** Which platform offers best price/performance for ML workloads?
2. **Performance:** Can we achieve <1s model inference on production CPUs?
3. **Import/Export:** Should we support versioned file formats from Day 1?
4. **Audio:** What's the best strategy for sample caching (memory vs reload)?
5. **Testing:** What's the minimum viable integration test coverage?

---

## Next Week Preview (Week 8)

**Focus Areas:**
- Arranger model training and integration (Max)
- Advanced editing features (Carson)
- UI/UX polish and analytics (Ayaan)
- Performance optimization and scaling (Steve)

**Goal:** Complete Arranger feature and prepare for final demo
