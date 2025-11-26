import { PianoRollNote, parseNotesFromDSL, updateDSLWithNewNotes, pitchToNote, snapToGrid, timeToX, xToTime, pitchToY, yToPitch, MIDI_MIN, MIDI_MAX, NOTE_HEIGHT, PIANO_WIDTH } from "./pianoRollHelpers";

interface HandlerDeps {
  dslCode: string;
  trackId: string;
  tempo: number;
  zoom: number;
  snapValue: number;
  snapEnabled: boolean;
  selectedNotes: Set<number>;
  setSelectedNotes: (notes: Set<number>) => void;
  onCodeChange: (newCode: string) => void;
  clipboard: PianoRollNote[];
  setClipboard: (notes: PianoRollNote[]) => void;
  clipboardOriginTime: number;
  setClipboardOriginTime: (time: number) => void;
  mousePosition: { x: number; y: number };
  setShowPastePreview: (show: boolean) => void;
}

export function createNoteHandlers(deps: HandlerDeps) {
  const {
    dslCode,
    trackId,
    tempo,
    zoom,
    snapValue,
    snapEnabled,
    selectedNotes,
    setSelectedNotes,
    onCodeChange,
    clipboard,
    setClipboard,
    clipboardOriginTime,
    setClipboardOriginTime,
    mousePosition,
    setShowPastePreview
  } = deps;

  const handleDeleteNote = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const filteredNotes = notes.filter((_, idx) => !selectedNotes.has(idx));
    updateDSLWithNewNotes(dslCode, trackId, filteredNotes, tempo, onCodeChange);
    setSelectedNotes(new Set());
  };

  const handleCopy = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const selectedNotesList = Array.from(selectedNotes).map(idx => notes[idx]);

    const earliestStart = Math.min(...selectedNotesList.map(n => n.start));

    setClipboard(selectedNotesList);
    setClipboardOriginTime(earliestStart);
  };

  const handleCut = () => {
    handleCopy();
    handleDeleteNote();
  };

  const handlePaste = () => {
    if (clipboard.length === 0) return;

    const notes = parseNotesFromDSL(dslCode, trackId, tempo);

    const pasteBeats = snapToGrid(xToTime(mousePosition.x, zoom), snapValue, snapEnabled);
    const timeOffset = pasteBeats - clipboardOriginTime;

    const newNotes = clipboard.map(note => ({
      ...note,
      start: note.start + timeOffset
    }));

    notes.push(...newNotes);
    updateDSLWithNewNotes(dslCode, trackId, notes, tempo, onCodeChange);

    const startIndex = notes.length - newNotes.length;
    const newSelection = new Set<number>();
    for (let i = 0; i < newNotes.length; i++) {
      newSelection.add(startIndex + i);
    }
    setSelectedNotes(newSelection);

    setShowPastePreview(true);
    setTimeout(() => setShowPastePreview(false), 300);
  };

  const handleQuantize = (gridValue: number) => {
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);

    const indicesToQuantize = selectedNotes.size > 0
      ? Array.from(selectedNotes)
      : notes.map((_, idx) => idx);

    indicesToQuantize.forEach(idx => {
      const note = notes[idx];
      note.start = Math.round(note.start / gridValue) * gridValue;
      note.duration = Math.max(gridValue, Math.round(note.duration / gridValue) * gridValue);
    });

    updateDSLWithNewNotes(dslCode, trackId, notes, tempo, onCodeChange);
  };

  const handleSelectAll = () => {
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const allIndices = new Set<number>();
    notes.forEach((_, idx) => allIndices.add(idx));
    setSelectedNotes(allIndices);
  };

  const handleDuplicate = () => {
    if (selectedNotes.size === 0) return;

    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const selectedNotesList = Array.from(selectedNotes).map(idx => notes[idx]);

    const latestEnd = Math.max(...selectedNotesList.map(n => n.start + n.duration));
    const offset = snapToGrid(latestEnd, snapValue, snapEnabled) - Math.min(...selectedNotesList.map(n => n.start));

    const duplicates = selectedNotesList.map(note => ({
      ...note,
      start: note.start + offset
    }));

    notes.push(...duplicates);
    updateDSLWithNewNotes(dslCode, trackId, notes, tempo, onCodeChange);

    const startIndex = notes.length - duplicates.length;
    const newSelection = new Set<number>();
    for (let i = 0; i < duplicates.length; i++) {
      newSelection.add(startIndex + i);
    }
    setSelectedNotes(newSelection);
  };

  const handleInvertSelection = () => {
    const notes = parseNotesFromDSL(dslCode, trackId, tempo);
    const newSelection = new Set<number>();

    notes.forEach((_, idx) => {
      if (!selectedNotes.has(idx)) {
        newSelection.add(idx);
      }
    });

    setSelectedNotes(newSelection);
  };

  return {
    handleDeleteNote,
    handleCopy,
    handleCut,
    handlePaste,
    handleQuantize,
    handleSelectAll,
    handleDuplicate,
    handleInvertSelection
  };
}