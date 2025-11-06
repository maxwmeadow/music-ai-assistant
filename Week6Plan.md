# Week 6 (Nov 1–6): DAW Features & Checkpoint 2 Demo

## Team Overview

| Member | Role | Focus Areas |
|--------|------|-------------|
| **Max** | Models + Demo Video | AI backend, model training, demo production |
| **Carson** | Editing Tools | Undo/redo, multi-select, copy/paste, velocity editor |
| **Ayaan** | QA + Polish | Testing, bug fixes, UI polish, tooltips |
| **Steve** | Transport + Tracks | Playback controls, loop region, track management |

**Hardware:** Max's HPCC GPUs for model training  
**Strategy:** Feature branch development → Daily integration testing → Frontend production-ready while Max handles AI backend

---

## AYAAN - QA Lead & UX Polish

### Mission
Ensure the product works reliably and feels professional. Systematically test every workflow, fix bugs, add visual feedback, and make the UI self-explanatory. Critical for Checkpoint 2 demo success.

### Key Libraries
- React testing utilities
- Browser DevTools (Console, Network, Performance)
- `grep` for code analysis

---

### Task 1: Setup Testing Framework
**Goal:** Create systematic testing infrastructure

**Deliverable:** `frontend/TESTING.md`
- **Structure:** Organized by workflow (Audio Recording, Piano Roll, Timeline, etc.)
- **Format:** Checkbox lists for each test case
- **Content:** Clear pass/fail criteria

**Steps:**
1. Define all major user workflows
2. Break each workflow into testable steps
3. Document expected vs actual behavior format
4. Create bug reporting template

---

### Task 2: Systematic Workflow Testing
**Goal:** Verify complete user journeys work end-to-end

**Key Workflows to Test:**
1. **Audio → Melody:** Record → Generate → Edit → Play
2. **Piano Roll Editing:** Create → Drag → Resize → Delete notes
3. **Timeline Editing:** Navigate, zoom, markers
4. **Mixer Controls:** Solo, mute, volume adjustments
5. **DSL Editing:** Type code → Compile → Play

**Documentation Pattern:**
- **Bug ID and title**
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Priority level** (CRITICAL/HIGH/MEDIUM/LOW)
- **File location and line number**

**Deliverable:** `frontend/BUGS.md` with prioritized bug list

---

### Task 3: Edge Case Testing
**Goal:** Find breaking conditions and boundary issues

**Test Categories:**
- **Empty states:** 0 tracks, 0 notes, empty project
- **Minimal data:** Single note, single track
- **Heavy load:** 1000+ notes, 15+ tracks
- **Extreme values:** Very short/long notes, out-of-range MIDI
- **Rapid interaction:** Double-clicking, rapid button presses
- **Large files:** Audio >10MB, <1 second

**Approach:**
- Systematically try each edge case
- Document unexpected behaviors
- Note performance degradation points

---

### Task 4: Tooltips and Help Text
**Goal:** Make every UI element self-explanatory

**Implementation:** Add `title` attribute to all interactive elements

**Coverage Areas:**
- All buttons (record, play, generate, compile, etc.)
- Mixer controls (solo, mute, volume sliders)
- Timeline controls (zoom, markers, playhead)
- Piano roll tools (select, draw, delete)

**Content Guidelines:**
- Describe what the button does
- Mention keyboard shortcut if applicable
- Keep under 10 words

---

### Task 5: Loading States
**Goal:** Provide visual feedback during async operations

**Components Needing Loading States:**
- Audio processing (Generate Melody)
- DSL compilation
- Instrument sample loading
- Project save/load

**Implementation Approach:**
- Add loading state variable (`isProcessing`)
- Disable button during operation
- Show spinner or "Processing..." text
- Use `try/finally` to ensure state resets

**Visual Pattern:**
- Spinner animation (CSS keyframes)
- Disabled button styling
- Clear text ("Processing...", "Loading...", "Saving...")

---

### Task 6: Error Message Improvement
**Goal:** Replace generic errors with helpful, actionable messages

**Error Categories:**
1. **Backend Connection:** "Cannot connect to backend. Is the server running?"
2. **Audio Processing:** "Audio processing failed. Try a shorter recording."
3. **DSL Compilation:** "DSL compilation failed. Check your syntax on line X."
4. **File Operations:** "Failed to load project. File may be corrupted."

**Pattern:**
- Catch specific error types
- Provide context about what failed
- Suggest solution when possible
- Avoid technical jargon

---

### Task 7: Console Cleanup
**Goal:** Remove debug statements and fix React warnings

**Tasks:**
1. Find all `console.log` statements: `grep -r "console.log" frontend/src/`
2. Remove or comment out debug statements
3. Fix React warnings (keys, hooks rules, etc.)
4. Verify no errors in browser console

**Goal:** Clean console for production demo

---

### Task 8: Cross-Browser Testing
**Goal:** Ensure compatibility across major browsers

**Test Matrix:**
- **Chrome** (primary target)
- **Firefox**
- **Safari** (if Mac available)
- **Edge**

**Document:**
- Features that work
- Performance differences
- Browser-specific bugs
- Severity assessment

**Deliverable:** Browser compatibility section in `TESTING.md`

---

### Task 9: Performance Testing
**Goal:** Verify app handles heavy load without lag

**Test Scenarios:**
- Create project with 10 tracks, 100 notes each
- Rapid note creation/deletion
- Continuous playback with many tracks
- Piano roll with 1000+ notes

**Monitoring:**
- UI responsiveness (lag when dragging?)
- Playback quality (stuttering?)
- Browser memory usage
- Use Chrome DevTools Performance tab

**Document:** Performance bottlenecks and reproduction steps

---

### Task 10: Visual Polish Pass
**Goal:** Ensure professional visual appearance

**Checklist:**
- [ ] Buttons have hover states
- [ ] Disabled buttons look disabled (grayed out, `cursor: not-allowed`)
- [ ] Active states are visible (selected track, playing button)
- [ ] No text overlap or clipping
- [ ] No unintended scrollbars
- [ ] Consistent color scheme
- [ ] Even spacing and alignment
- [ ] Icons properly sized and aligned

**Fix Issues:** Adjust CSS/Tailwind classes for consistency

---

### Task 11: Bug Prioritization
**Goal:** Organize bugs by impact for efficient fixing

**Categories in `BUGS.md`:**
- **CRITICAL:** Blocks demo, crashes app
- **HIGH:** Significantly impacts UX
- **MEDIUM:** Minor annoyance, workaround exists
- **LOW:** Polish item, nice to fix

**Process:** Sort bugs, tackle CRITICAL/HIGH first

---

### Task 12: Bug Fixing
**Goal:** Resolve critical and high-priority bugs

**Workflow:**
1. Reproduce bug reliably
2. Fix issue
3. Test fix thoroughly
4. Commit with descriptive message: `"Fix: [Bug description] (#ID)"`
5. Verify no regressions

---

### Integration Points
- **Test Carson's editing features** as they're added
- **Test Steve's transport features** as they're added
- **Ensure all new features have tooltips and loading states**
- **Verify no regressions** in existing features after each merge

### Testing Strategy
1. Test on real devices, not just dev environment
2. Document every bug with clear reproduction steps
3. Fix critical bugs before adding polish
4. Use browser DevTools for debugging
5. Test keyboard navigation (Tab, Enter, Space)

---

## CARSON - Core Editing Features

### Mission
Build essential music editing tools that make the DAW productive. Undo/redo gives users confidence to experiment, multi-select enables efficient editing, velocity control adds musical expression, and quantize ensures timing precision.

### Key Libraries
- React hooks (`useState`, `useCallback`, `useEffect`, `useRef`)
- Canvas API for rendering
- Keyboard event handling

---

### Task 1: Setup Feature Branch
**Goal:** Create isolated development environment

**Commands:**
```bash
git checkout dev
git pull origin dev
git checkout -b feature/editing-tools
```

**Setup:** Create test files for development

---

### Task 2: History System
**Goal:** Implement undo/redo functionality

**Hook:** `useHistory(initialCode)`
- **Parameters:** Initial code string
- **Returns:** 
  - `pushHistory(code)` - Add state to history
  - `undo()` - Go back one state
  - `redo()` - Go forward one state  
  - `canUndo` - Boolean
  - `canRedo` - Boolean

**Data Structure:** Array of history entries with timestamps

**Implementation Considerations:**
- How to handle branching history (undo → edit → new branch)?
- When to push to history (every change vs debounced)?
- Maximum history size?

---

### Task 3: Integrate History into App
**Goal:** Connect history system to main application state

**Integration Points:**
- Code editor changes
- Piano roll note edits
- Track modifications

**Keyboard Shortcuts:**
- Ctrl+Z / Cmd+Z → Undo
- Ctrl+Y / Cmd+Y → Redo
- Ctrl+Shift+Z / Cmd+Shift+Z → Redo (alternative)

**Challenge:** Re-parse DSL and update UI when undoing/redoing

---

### Task 4: Multi-Select in Piano Roll
**Goal:** Allow selecting multiple notes with box selection

**State Requirements:**
- Set of selected note indices
- Box selection coordinates (start, end)
- Is box selecting flag

**Interactions:**
- Click + drag empty space → Box select
- Shift + click note → Add to selection
- Click note → Select (clear others if no Shift)
- Click empty → Clear selection

**Visual Feedback:**
- Draw selection box during drag
- Highlight selected notes (different color/border)

**Research:** Canvas mouse event handling for drag operations

---

### Task 5: Copy/Paste Implementation
**Goal:** Enable duplicating and moving note groups

**State Requirements:**
- Clipboard (array of notes)
- Clipboard origin time

**Operations:**
- **Copy (Ctrl+C):** Store selected notes
- **Paste (Ctrl+V):** Insert at playhead with time offset
- **Cut (Ctrl+X):** Copy + delete
- **Delete (Del/Backspace):** Remove selected notes

**Challenge:** Preserve relative timing when pasting

---

### Task 6: Velocity Editor
**Goal:** Visual velocity editing below piano roll

**Component:** Secondary canvas showing velocity bars

**Rendering:**
- Bar height represents velocity (0-1 → 0-100px)
- Each note gets a velocity bar
- Selected notes highlighted

**Interaction:**
- Click/drag to adjust velocity
- Find note at cursor position
- Calculate velocity from y-position (inverted: top=1.0, bottom=0.0)

**Research:** Canvas layering and coordinated scrolling

---

### Task 7: Quantize Function
**Goal:** Snap notes to rhythmic grid

**Function:** `quantizeNotes(notes, gridValue)`
- **Input:** Note array, grid size (1.0=whole, 0.25=quarter, etc.)
- **Returns:** Quantized note array

**Algorithm:**
1. Calculate grid size in seconds from BPM
2. Round start times to nearest grid point
3. Round durations to grid multiples
4. Ensure minimum duration

**UI:**
- Dropdown for grid value selection
- Button to apply quantize
- Apply to selected notes (or all if none selected)

---

### Task 8: Keyboard Shortcuts Modal
**Goal:** Display available keyboard shortcuts

**Component:** Modal dialog showing shortcut reference

**Content:**
- Undo/Redo
- Select All / Copy / Paste / Delete
- Play/Pause
- Show shortcuts (? key)

**Interaction:**
- Press "?" to open
- Click outside or close button to dismiss
- Escape key to close

---

### Task 9: Selection Helpers
**Goal:** Additional selection operations

**Operations to Implement:**
- **Select All (Ctrl+A):** Select all notes in current track
- **Duplicate (Ctrl+D):** Copy + paste in place (or offset)
- **Deselect:** Click empty area

---

### Task 10: Testing
**Goal:** Verify all editing features work correctly

**Test Checklist:**

**Undo/Redo:**
- [ ] Create note → Undo → Note disappears
- [ ] Undo → Redo → Note reappears
- [ ] Edit note → Undo → Returns to original
- [ ] Buttons disabled appropriately
- [ ] Keyboard shortcuts work

**Multi-Select:**
- [ ] Box selection works
- [ ] Shift+click adds to selection
- [ ] Selected notes highlighted
- [ ] Click empty clears selection

**Copy/Paste:**
- [ ] Copy preserves note data
- [ ] Paste at playhead works
- [ ] Relative timing preserved
- [ ] Pasted notes selected

**Velocity:**
- [ ] Bars display correctly
- [ ] Editing updates note velocity
- [ ] Changes reflected in playback

**Quantize:**
- [ ] Snaps to grid correctly
- [ ] Different grid sizes work
- [ ] Only affects selected notes (if any selected)

---

### Task 11: Commit and Push
**Goal:** Share completed features

```bash
git add .
git commit -m "Add editing features: undo/redo, multi-select, copy/paste, velocity, quantize"
git push origin feature/editing-tools
```

Create pull request for team review

---

### Integration Points
- **History integrates with page.tsx state**
- **Multi-select works with existing piano roll**
- **Copy/paste uses playhead time from playback hook**
- **Velocity changes trigger DSL update**

### Key Decisions
1. When to push to history (every change vs debounced)?
2. How to handle box selection across multiple tracks?
3. Should velocity editing affect multiple notes at once?
4. What's the default quantize grid size?

---

## STEVE - Transport Controls & Track Management

### Mission
Build playback workflow features and track organization tools. Pause/resume enables iterative editing, loop region focuses work on specific sections, metronome keeps time, and track management UI removes need for manual DSL editing.

### Key Libraries
- Tone.js for audio transport control
- React state management
- File I/O (FileReader, Blob, createElement)
- Lucide icons

---

### Task 1: Setup Feature Branch
**Goal:** Create isolated development environment

```bash
git checkout dev
git pull origin dev
git checkout -b feature/transport-tracks
```

---

### Task 2: Pause/Resume Functionality
**Goal:** Allow pausing and resuming playback

**State Requirements:**
- `isPlaying` - Currently playing flag
- `isPaused` - Paused (not stopped) flag
- `pausePosition` - Time when paused

**Operations:**
- **Play:** Start from beginning or resume from pause
- **Pause:** Stop temporarily, save position
- **Stop:** Reset to beginning

**Tone.js Methods to Use:**
- `Tone.Transport.start()`
- `Tone.Transport.pause()`
- `Tone.Transport.stop()`
- `Tone.Transport.seconds` (get/set position)

**Keyboard Shortcut:** Space bar for play/pause toggle

---

### Task 3: Loop Region
**Goal:** Enable looping specific sections

**State Requirements:**
- `loopEnabled` - Boolean flag
- `loopStart` - Start beat
- `loopEnd` - End beat

**Implementation:**
- Convert beats to seconds using tempo
- Set `Tone.Transport.loopStart` and `loopEnd`
- Set `Tone.Transport.loop = true`

**UI Controls:**
- Checkbox to enable/disable loop
- Number inputs for start/end beats
- Step by 0.25 (sixteenth notes)

---

### Task 4: Loop Markers on Timeline
**Goal:** Visualize loop region

**Rendering:**
- Shaded region between markers
- Vertical lines at start/end
- Different color for loop region

**Interaction:**
- Drag loop start marker
- Drag loop end marker
- Ensure start < end

**Canvas Drawing:**
- Calculate marker positions from beats
- Use `ctx.fillRect()` for region
- Use `ctx.strokeStyle` for markers

---

### Task 5: Track Management UI
**Goal:** Add/remove tracks without editing DSL

**Component:** `TrackManager`
- **Props:** `tracks`, `code`, `onCodeChange`

**Operations:**
- **Add Track:** Prompt for name and instrument, append DSL
- **Delete Track:** Confirm, remove from DSL using regex
- **Display:** Show track list with instrument info

**DSL Manipulation:**
- Parse track blocks with regex
- Generate new track DSL template
- Remove track blocks while preserving others

**Challenge:** Reliable regex for nested DSL structures

---

### Task 6: Metronome
**Goal:** Audible click track for timing reference

**Class:** `Metronome`
- **Methods:**
  - `start(tempo)` - Begin clicking
  - `stop()` - Stop clicking
  - `setVolume(db)` - Adjust volume
  - `dispose()` - Clean up

**Implementation:**
- Use `Tone.Synth` with short envelope
- Use `Tone.Loop` to trigger on beats
- Click on every quarter note ("4n")
- Set lower volume than music (-10 dB)

**UI:** Checkbox to enable/disable metronome

---

### Task 7: Tempo Control
**Goal:** Adjust tempo dynamically

**Component:** `TransportControls`
- **Props:** `tempo`, `onTempoChange`

**UI Elements:**
- Number input (40-240 BPM range)
- Increment/decrement buttons (+/- 1 BPM)
- Display "BPM" label

**Sync Requirements:**
- Update DSL code with new tempo
- Update `Tone.Transport.bpm.value` if playing
- Re-calculate loop markers if loop enabled

---

### Task 8: Project Save/Load
**Goal:** Persist and restore project state

**Save Function:**
- **Collect:** code, tracks, tempo, loop settings, timestamp
- **Format:** JSON
- **Action:** Create Blob, trigger download

**Load Function:**
- **Trigger:** File input
- **Read:** FileReader to parse JSON
- **Restore:** All state variables
- **Validate:** Handle invalid/corrupted files

**Error Handling:**
- Try/catch for JSON parsing
- User feedback via toast messages

---

### Task 9: Testing
**Goal:** Verify all transport and track features

**Test Checklist:**

**Pause/Resume:**
- [ ] Play → Pause → Resume continues correctly
- [ ] Play → Stop → Play restarts from beginning
- [ ] Space bar toggles play/pause
- [ ] Buttons disabled appropriately

**Loop:**
- [ ] Loop region plays correctly
- [ ] Markers visible on timeline
- [ ] Dragging markers works
- [ ] Loop disabled works

**Track Management:**
- [ ] Add track appears in UI and DSL
- [ ] Delete track removes from UI and DSL
- [ ] Multiple tracks work
- [ ] Edge case: Delete all tracks

**Metronome:**
- [ ] Click audible on beats
- [ ] Syncs with tempo changes
- [ ] Stops with playback

**Tempo:**
- [ ] UI updates DSL
- [ ] Playback tempo changes live
- [ ] Clamped to valid range
- [ ] +/- buttons work

**Save/Load:**
- [ ] Save downloads JSON
- [ ] Load restores all state
- [ ] Invalid file shows error

---

### Task 10: Commit and Push
**Goal:** Share completed features

```bash
git add .
git commit -m "Add transport and track features: pause, loop, track manager, metronome, tempo control, save/load"
git push origin feature/transport-tracks
```

Create pull request for review

---

### Integration Points
- **Pause/resume works with existing Tone.js playback**
- **Loop region renders on Timeline component**
- **Track manager modifies DSL code directly**
- **Tempo changes update both DSL and Tone.Transport**

### Key Decisions
1. Should loop markers be draggable or input-only?
2. How to handle track deletion with references in other tracks?
3. What metadata to include in project save files?
4. Should metronome have accent on downbeat?

---

## MAX - Model Integration & Demo Video

### Mission
Integrate arranger transformer, train on HPCC, create demo video showcasing end-to-end workflow. This is the capstone of Week 5 work and proves the system works.

### Key Libraries
- PyTorch for model training
- HPCC job submission scripts
- Video editing software
- Screen recording tools

---

### Task 1: Model Integration
**Goal:** Connect arranger model to backend API

**File:** `backend/models/arranger_transformer.py`

**Integration Steps:**
1. Import trained model checkpoint
2. Create inference endpoint
3. Handle input preprocessing
4. Return IR format output

**Endpoint:** `/api/generate-arrangement`
- **Input:** Melody/chord sequence (IR format)
- **Output:** Multi-instrument arrangement (IR format)

---

### Task 2: HPCC Training
**Goal:** Train model on full dataset with GPUs

**Tasks:**
1. Transfer preprocessed data to HPCC
2. Create SLURM job script
3. Configure GPU allocation
4. Monitor training progress
5. Download checkpoints

**Checkpointing Strategy:**
- Save every 5 epochs
- Save best validation model
- Keep last 3 checkpoints only (space constraints)

---

### Task 3: Model Evaluation
**Goal:** Verify model performance metrics

**Metrics to Calculate:**
- Training loss curve
- Validation loss curve
- Inference time (ms per sequence)
- Model size (MB)
- GPU memory usage

**Deliverable:** `METRICS.md` with performance stats

---

### Task 4: Demo Video Planning
**Goal:** Outline compelling demo narrative

**Structure (3-5 minutes):**
1. **Intro (30s):** Project overview, team, goals
2. **Problem (30s):** Why music arrangement is hard
3. **Solution (1min):** System architecture overview
4. **Demo (2-3min):** 
   - Record vocal melody
   - Generate melody from audio
   - Generate arrangement (bass, pads, counter)
   - Edit in piano roll
   - Play final result
5. **Technical (30s):** Brief model/tech mention
6. **Conclusion (30s):** What works, what's next

---

### Task 5: Demo Video Recording
**Goal:** Capture high-quality screen recording

**Tools:**
- Screen recorder (OBS, QuickTime, etc.)
- Audio recording for voiceover
- Music playback capture

**Tips:**
- Clean desktop/browser
- Prepare example sequences
- Rehearse workflow multiple times
- Record in segments (easier to edit)
- Capture audio separately for better quality

---

### Task 6: Video Editing
**Goal:** Produce polished final video

**Editing Tasks:**
- Trim and arrange clips
- Add title cards and text overlays
- Include team member names
- Add background music (low volume)
- Sync voiceover to visuals
- Add transitions between sections
- Export in HD (1080p)

**Deliverable:** `DEMO_VIDEO.mp4`

---

### Task 7: Backend Deployment
**Goal:** Ensure backend services are running and accessible

**Checklist:**
- [ ] Backend API deployed and responding
- [ ] Model checkpoint loaded correctly
- [ ] Frontend can connect to backend
- [ ] CORS configured properly
- [ ] Error handling in place

---

### Integration Points
- **Model uses Ayaan's IR format**
- **Backend serves model predictions**
- **Frontend displays generated arrangements**
- **Demo showcases full pipeline**

### Key Decisions
1. Which model checkpoint to use for demo?
2. Live demo vs pre-recorded sequences?
3. How much technical detail in video?
4. Background music for video?

---

## Deliverables Summary

### Max
- `backend/models/arranger_transformer.py` - Integrated model
- `backend/checkpoints/arranger_model.pth` - Trained checkpoint
- `DEMO_VIDEO.mp4` - 3-5 minute demo video
- `METRICS.md` - Performance metrics and stats

### Ayaan
- `frontend/TESTING.md` - Complete testing checklist with results
- `frontend/BUGS.md` - Bug tracker with priorities
- All UI controls have tooltips
- Loading states on all async operations
- Improved error messages throughout
- Clean console (no errors/warnings)

### Carson
- `frontend/src/hooks/useHistory.ts` - Undo/redo hook
- `frontend/src/components/KeyboardShortcuts.tsx` - Shortcuts modal
- Multi-select, copy/paste in PianoRoll
- Velocity editor in PianoRoll
- Quantize function
- All features tested and working

### Steve
- `frontend/src/components/TrackManager.tsx` - Track management UI
- `frontend/src/components/TransportControls.tsx` - Tempo control UI
- `frontend/src/lib/metronome.ts` - Metronome class
- Pause/resume playback
- Loop region with draggable markers
- Project save/load functionality
- All features tested and working

---

## Success Metrics

### Minimum Viable (Required for Checkpoint 2)
- ✓ All services deployed and accessible
- ✓ Hum2melody model generates valid melodies
- ✓ Demo video showcases end-to-end workflow
- ✓ Zero critical bugs (app doesn't crash)
- ✓ Undo/redo works for note edits
- ✓ Pause/resume playback functional
- ✓ UI has tooltips and loading states

### Target (Impressive for Checkpoint 2)
- ✓ Arranger model integrated and generating accompaniment
- ✓ Multi-select, copy/paste fully functional
- ✓ Loop region with visual markers
- ✓ Track management UI working
- ✓ Velocity editor functional
- ✓ Project save/load working
- ✓ Metronome and tempo control
- ✓ Cross-browser tested (Chrome, Firefox)
- ✓ Demo video is polished and exciting
- ✓ Performance metrics documented

### Stretch Goals
- ✓ All keyboard shortcuts working smoothly
- ✓ Mobile-responsive layout
- ✓ Quantize with multiple grid options
- ✓ Advanced error handling and recovery
- ✓ Performance optimized (no lag with 10+ tracks)

---

## Development Tips

### For Everyone
- **Feature branches:** Work in isolation, merge frequently
- **Daily integration:** Test together to catch conflicts early
- **Document decisions:** Especially UX and architecture choices
- **Ask early:** Don't wait until blocked

### Communication
- Daily standup: Progress, blockers, needs
- Share WIP code: Even incomplete implementations
- Define interfaces: Agree on data structures first
- Demo features: Show working features to team

### Debugging
- Browser DevTools are your friend
- Test on real devices
- Use console strategically (then remove)
- Test edge cases intentionally

### Performance
- Profile before optimizing
- Test with realistic data volumes
- Monitor memory usage
- Batch operations when possible

---

## Key Research Questions

1. **UX:** What keyboard shortcuts do users expect in a DAW?
2. **Performance:** How many notes before canvas rendering lags?
3. **Architecture:** Best way to sync Tone.Transport state with React?
4. **Testing:** What automated testing is worth the setup time?
5. **Polish:** What visual cues make a professional-feeling app?

Research these as you go and share findings with the team.