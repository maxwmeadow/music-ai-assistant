"""
DSL Context Extraction
Extract musical context (tempo, duration, key, chords) from existing DSL code
"""

import re
from typing import Optional, Dict, List


def extract_tempo(dsl_code: str) -> int:
    """
    Extract tempo from DSL code.

    Args:
        dsl_code: The DSL code string

    Returns:
        Tempo in BPM, defaults to 120 if not found
    """
    tempo_match = re.search(r'tempo\s*\(\s*(\d+)\s*\)', dsl_code)
    if tempo_match:
        return int(tempo_match.group(1))
    return 120  # Default tempo


def calculate_duration(dsl_code: str) -> float:
    """
    Calculate the total duration of the music by finding the last note end time.

    Args:
        dsl_code: The DSL code string

    Returns:
        Duration in seconds
    """
    max_end_time = 0.0

    # Find all note() commands: note("pitch", start, duration, velocity)
    note_pattern = r'note\s*\(\s*"[^"]+"\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)'
    for match in re.finditer(note_pattern, dsl_code):
        start = float(match.group(1))
        duration = float(match.group(2))
        end_time = start + duration
        max_end_time = max(max_end_time, end_time)

    # Find all chord() commands: chord([...], start, duration, velocity)
    chord_pattern = r'chord\s*\(\s*\[[^\]]+\]\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)'
    for match in re.finditer(chord_pattern, dsl_code):
        start = float(match.group(1))
        duration = float(match.group(2))
        end_time = start + duration
        max_end_time = max(max_end_time, end_time)

    # Find all loop() commands and calculate their end times
    loop_pattern = r'loop\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)'
    for match in re.finditer(loop_pattern, dsl_code):
        end_time = float(match.group(2))
        max_end_time = max(max_end_time, end_time)

    return max_end_time


def note_to_pitch_class(note: str) -> int:
    """
    Convert note name to pitch class (0-11).

    Args:
        note: Note name (C, C#, Db, etc.)

    Returns:
        Pitch class (0=C, 1=C#/Db, 2=D, etc.)
    """
    # Normalize flats to sharps
    note = note.replace('Db', 'C#').replace('Eb', 'D#').replace('Gb', 'F#')
    note = note.replace('Ab', 'G#').replace('Bb', 'A#')

    pitch_map = {
        'C': 0, 'C#': 1,
        'D': 2, 'D#': 3,
        'E': 4,
        'F': 5, 'F#': 6,
        'G': 7, 'G#': 8,
        'A': 9, 'A#': 10,
        'B': 11
    }

    return pitch_map.get(note, 0)


def analyze_chord_quality(notes: List[str]) -> str:
    """
    Analyze chord quality based on intervals.

    Args:
        notes: List of note names (without octave)

    Returns:
        Chord quality string (maj, m, 7, maj7, m7, dim, aug, sus, or unknown)
    """
    if not notes or len(notes) < 2:
        return ""

    # Root is the FIRST note (not the lowest)
    root = note_to_pitch_class(notes[0])

    # Convert all notes to pitch classes and remove duplicates
    pitch_classes = list(set(note_to_pitch_class(note) for note in notes))

    # Get intervals relative to root
    intervals = sorted((pc - root) % 12 for pc in pitch_classes)

    # Analyze intervals to determine quality
    has_maj3 = 4 in intervals
    has_min3 = 3 in intervals
    has_5th = 7 in intervals
    has_dim5 = 6 in intervals
    has_aug5 = 8 in intervals
    has_min7 = 10 in intervals
    has_maj7 = 11 in intervals
    has_2nd = 2 in intervals
    has_4th = 5 in intervals

    # Determine chord quality
    if len(intervals) == 2:
        # Two-note chords (power chords or dyads)
        if has_5th:
            return "5"  # Power chord
        return ""

    # Triads and extended chords
    if has_maj3 and has_5th:
        # Major-based chords
        if has_maj7:
            return "maj7"
        elif has_min7:
            return "7"  # Dominant 7th
        else:
            return "maj"

    elif has_min3 and has_5th:
        # Minor-based chords
        if has_maj7:
            return "m(maj7)"
        elif has_min7:
            return "m7"
        else:
            return "m"

    elif has_min3 and has_dim5:
        # Diminished
        if has_min7:
            return "m7b5"  # Half-diminished
        return "dim"

    elif has_maj3 and has_aug5:
        # Augmented
        return "aug"

    elif has_4th and has_5th:
        # Suspended chords
        if has_min7:
            return "7sus4"
        return "sus4"

    elif has_2nd and has_5th:
        if has_min7:
            return "7sus2"
        return "sus2"

    # Unknown/complex chord
    return ""


def extract_chord_progression(dsl_code: str) -> List[str]:
    """
    Extract chord progression with quality analysis from DSL code.

    Args:
        dsl_code: The DSL code string

    Returns:
        List of chord names with quality (e.g., ["Cmaj", "Dm7", "G7"])
    """
    chords = []

    # Find all chord() commands
    chord_pattern = r'chord\s*\(\s*\[([^\]]+)\]\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)'

    for match in re.finditer(chord_pattern, dsl_code):
        notes_str = match.group(1)
        start_time = float(match.group(2))

        # Extract note names from the chord (without octave)
        note_matches = re.findall(r'"([A-G][#b]?)\d+"', notes_str)
        if note_matches:
            # Get the root note (first note)
            root = note_matches[0]

            # Analyze chord quality
            quality = analyze_chord_quality(note_matches)

            # Build chord name
            if quality:
                chord_name = f"{root}{quality}"
            else:
                chord_name = root

            chords.append((start_time, chord_name))

    # Sort by start time and return just the chord names
    chords.sort(key=lambda x: x[0])
    return [chord[1] for chord in chords]


def extract_notes_with_durations(dsl_code: str) -> Dict[str, float]:
    """
    Extract notes with their total durations (weighted by duration).

    Args:
        dsl_code: The DSL code string

    Returns:
        Dictionary mapping note names to total duration in seconds
    """
    note_durations = {}

    # Find all note() commands: note("pitch", start, duration, velocity)
    note_pattern = r'note\s*\(\s*"([A-G][#b]?)\d+"\s*,\s*[\d.]+\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)'
    for match in re.finditer(note_pattern, dsl_code):
        note = match.group(1)
        duration = float(match.group(2))
        note_durations[note] = note_durations.get(note, 0) + duration

    # Find all chord() commands and add durations for each note in the chord
    chord_pattern = r'chord\s*\(\s*\[([^\]]+)\]\s*,\s*[\d.]+\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)'
    for match in re.finditer(chord_pattern, dsl_code):
        notes_str = match.group(1)
        duration = float(match.group(2))

        # Extract note names from the chord
        note_matches = re.findall(r'"([A-G][#b]?)\d+"', notes_str)
        for note in note_matches:
            note_durations[note] = note_durations.get(note, 0) + duration

    return note_durations


def detect_key(dsl_code: str, chord_progression: Optional[List[str]] = None) -> Optional[str]:
    """
    Detect the musical key using weighted note analysis and chord progressions.

    Args:
        dsl_code: The DSL code string
        chord_progression: Optional chord progression for additional context

    Returns:
        Detected key (e.g., "C major", "A minor") or None
    """
    # Extract notes weighted by duration
    note_durations = extract_notes_with_durations(dsl_code)

    if not note_durations:
        return None

    # Find the most emphasized note (by duration)
    tonic_candidate = max(note_durations.items(), key=lambda x: x[1])[0]
    tonic_pitch = note_to_pitch_class(tonic_candidate)

    # Analyze scale degrees to determine major vs minor
    # Build a pitch class profile weighted by duration
    pitch_class_weights = {}
    for note, duration in note_durations.items():
        pc = note_to_pitch_class(note)
        pitch_class_weights[pc] = pitch_class_weights.get(pc, 0) + duration

    # Get intervals relative to tonic candidate
    intervals_present = set()
    for pc in pitch_class_weights.keys():
        interval = (pc - tonic_pitch) % 12
        intervals_present.add(interval)

    # Characteristic intervals for major vs minor
    has_maj3 = 4 in intervals_present  # Major 3rd
    has_min3 = 3 in intervals_present  # Minor 3rd
    has_maj6 = 9 in intervals_present  # Major 6th
    has_min6 = 8 in intervals_present  # Minor 6th
    has_maj7 = 11 in intervals_present  # Major 7th
    has_min7 = 10 in intervals_present  # Minor 7th

    # Use chord progression as additional evidence if available
    is_major = True  # Default assumption

    if chord_progression:
        # Analyze first and last chords (strong key indicators)
        first_chord = chord_progression[0] if chord_progression else ""
        last_chord = chord_progression[-1] if chord_progression else ""

        # Check if tonic chord is major or minor
        if tonic_candidate in first_chord or tonic_candidate in last_chord:
            if 'm' in first_chord.replace('maj', '') or 'm' in last_chord.replace('maj', ''):
                is_major = False

    # Analyze scale degrees
    if has_min3 and not has_maj3:
        # Clear minor 3rd, no major 3rd
        is_major = False
    elif has_maj3 and not has_min3:
        # Clear major 3rd, no minor 3rd
        is_major = True
    elif has_min3 and has_min6:
        # Both minor 3rd and minor 6th suggest minor
        is_major = False
    elif has_maj3 and has_maj6:
        # Both major 3rd and major 6th suggest major
        is_major = True
    elif has_min7 and has_min3:
        # Minor 7th with minor 3rd suggests minor
        is_major = False
    elif has_maj7 and has_maj3:
        # Major 7th with major 3rd suggests major
        is_major = True

    # Consider relative major/minor
    if is_major:
        key_name = f"{tonic_candidate} major"
    else:
        key_name = f"{tonic_candidate} minor"

    # Note: Could also check for relative keys (e.g., C major vs A minor)
    # but this requires more sophisticated analysis

    return key_name


def analyze_note_density(dsl_code: str) -> Dict:
    """
    Analyze rhythmic/note density of the music.

    Args:
        dsl_code: The DSL code string

    Returns:
        Dictionary with density metrics
    """
    # Count total notes and chords
    note_count = 0
    chord_count = 0

    # Find all note() commands
    note_pattern = r'note\s*\(\s*"[^"]+"\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)'
    note_count = len(re.findall(note_pattern, dsl_code))

    # Find all chord() commands
    chord_pattern = r'chord\s*\(\s*\[[^\]]+\]\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)'
    chord_count = len(re.findall(chord_pattern, dsl_code))

    # Calculate duration
    duration = calculate_duration(dsl_code)
    if duration == 0:
        return {'density': 'unknown', 'notes_per_second': 0}

    # Calculate notes per second
    total_events = note_count + chord_count
    notes_per_second = total_events / duration

    # Classify density
    if notes_per_second < 0.5:
        density = 'very_sparse'  # Less than 1 event per 2 seconds
    elif notes_per_second < 1.5:
        density = 'sparse'  # 1-2 events per second
    elif notes_per_second < 3.0:
        density = 'moderate'  # 2-4 events per second
    elif notes_per_second < 5.0:
        density = 'busy'  # 4-6 events per second
    else:
        density = 'very_busy'  # More than 6 events per second

    return {
        'density': density,
        'notes_per_second': round(notes_per_second, 2),
        'total_events': total_events
    }


def analyze_harmonic_rhythm(dsl_code: str, chord_progression: Optional[List[str]] = None) -> Dict:
    """
    Analyze how frequently chords change (harmonic rhythm).

    Args:
        dsl_code: The DSL code string
        chord_progression: Optional chord progression

    Returns:
        Dictionary with harmonic rhythm metrics
    """
    if not chord_progression:
        chord_progression = extract_chord_progression(dsl_code)

    if not chord_progression:
        return {'pace': 'none', 'changes_per_minute': 0}

    # Find chord change times
    chord_pattern = r'chord\s*\(\s*\[[^\]]+\]\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\)'
    chord_times = [float(m.group(1)) for m in re.finditer(chord_pattern, dsl_code)]

    if len(chord_times) < 2:
        return {'pace': 'static', 'changes_per_minute': 0}

    # Calculate average time between chord changes
    duration = calculate_duration(dsl_code)
    num_changes = len(chord_times)

    # Changes per minute
    changes_per_minute = (num_changes / duration) * 60 if duration > 0 else 0

    # Classify pace
    if changes_per_minute < 15:
        pace = 'very_slow'  # Less than 1 chord per 4 seconds
    elif changes_per_minute < 30:
        pace = 'slow'  # 1 chord per 2-4 seconds
    elif changes_per_minute < 60:
        pace = 'moderate'  # 1-2 chords per second
    elif changes_per_minute < 120:
        pace = 'fast'  # 2-4 chords per second
    else:
        pace = 'very_fast'  # More than 4 chords per second

    return {
        'pace': pace,
        'changes_per_minute': round(changes_per_minute, 1),
        'num_chords': num_changes
    }


def analyze_melodic_range(dsl_code: str) -> Dict:
    """
    Analyze the melodic range and register of notes.

    Args:
        dsl_code: The DSL code string

    Returns:
        Dictionary with range metrics
    """
    # Find all notes with octaves
    note_pattern = r'"([A-G][#b]?)(\d+)"'
    notes_with_octaves = re.findall(note_pattern, dsl_code)

    if not notes_with_octaves:
        return {'register': 'unknown', 'range_semitones': 0}

    # Convert to MIDI numbers
    midi_numbers = []
    for note_name, octave in notes_with_octaves:
        pitch_class = note_to_pitch_class(note_name)
        midi = pitch_class + (int(octave) + 1) * 12
        midi_numbers.append(midi)

    if not midi_numbers:
        return {'register': 'unknown', 'range_semitones': 0}

    min_midi = min(midi_numbers)
    max_midi = max(midi_numbers)
    range_semitones = max_midi - min_midi
    avg_midi = sum(midi_numbers) / len(midi_numbers)

    # Classify register (based on average MIDI note)
    if avg_midi < 48:  # Below C3
        register = 'very_low'
    elif avg_midi < 60:  # C3 to C4
        register = 'low'
    elif avg_midi < 72:  # C4 to C5
        register = 'mid'
    elif avg_midi < 84:  # C5 to C6
        register = 'high'
    else:  # Above C6
        register = 'very_high'

    return {
        'register': register,
        'range_semitones': range_semitones,
        'min_midi': min_midi,
        'max_midi': max_midi,
        'avg_midi': round(avg_midi, 1)
    }


def analyze_voice_leading(dsl_code: str, chord_progression: Optional[List[str]] = None) -> Dict:
    """
    Analyze voice leading opportunities in the chord progression.

    Args:
        dsl_code: The DSL code string
        chord_progression: Optional chord progression

    Returns:
        Dictionary with voice leading suggestions
    """
    if not chord_progression:
        chord_progression = extract_chord_progression(dsl_code)

    if len(chord_progression) < 2:
        return {'guidance': 'none', 'suggestion': 'Not enough chords to analyze'}

    # Analyze chord transitions
    transitions = []
    for i in range(len(chord_progression) - 1):
        curr = chord_progression[i]
        next_chord = chord_progression[i + 1]

        # Extract root notes (remove quality markers)
        curr_root = curr.replace('maj7', '').replace('m7', '').replace('maj', '').replace('m', '').replace('7', '').replace('5', '')
        next_root = next_chord.replace('maj7', '').replace('m7', '').replace('maj', '').replace('m', '').replace('7', '').replace('5', '')

        # Calculate interval between roots
        curr_pitch = note_to_pitch_class(curr_root)
        next_pitch = note_to_pitch_class(next_root)
        interval = (next_pitch - curr_pitch) % 12

        # Classify transition
        if interval == 0:
            transition_type = 'same'
        elif interval == 7 or interval == 5:  # Perfect 5th up or 4th up
            transition_type = 'strong'  # Common progressions (V-I, IV-I)
        elif interval == 2 or interval == 10:  # Whole step
            transition_type = 'smooth'
        elif interval == 1 or interval == 11:  # Half step
            transition_type = 'chromatic'
        else:
            transition_type = 'leap'

        transitions.append({
            'from': curr,
            'to': next_chord,
            'type': transition_type,
            'interval': interval
        })

    # Generate guidance based on transitions
    strong_count = sum(1 for t in transitions if t['type'] == 'strong')
    smooth_count = sum(1 for t in transitions if t['type'] == 'smooth')
    leap_count = sum(1 for t in transitions if t['type'] == 'leap')
    total = len(transitions)

    if strong_count >= total * 0.6:
        guidance = 'functional'
        suggestion = 'Use functional harmony (V-I, IV-I). Keep common tones between chords for smooth voice leading.'
    elif smooth_count >= total * 0.5:
        guidance = 'stepwise'
        suggestion = 'Use smooth voice leading with stepwise motion. Move voices by step (2nd) when possible.'
    elif leap_count >= total * 0.5:
        guidance = 'independent'
        suggestion = 'Independent voice motion detected. Use contrary or oblique motion for smoother sound.'
    else:
        guidance = 'mixed'
        suggestion = 'Mixed progression. Balance common tones with melodic motion in inner voices.'

    return {
        'guidance': guidance,
        'suggestion': suggestion
    }


def build_music_context(dsl_code: str, include_key: bool = True, include_chords: bool = True, include_analysis: bool = True) -> Dict:
    """
    Build a complete music context dictionary from DSL code.

    Args:
        dsl_code: The DSL code string
        include_key: Whether to attempt key detection
        include_chords: Whether to extract chord progression
        include_analysis: Whether to include rhythm/density analysis

    Returns:
        Dictionary with musical context
    """
    context = {
        'duration': calculate_duration(dsl_code)
    }

    # Extract chords first (useful for key detection and harmonic rhythm)
    chords = None
    if include_chords:
        chords = extract_chord_progression(dsl_code)
        if chords:
            context['chords'] = chords

    # Detect key using chord progression as additional evidence
    if include_key:
        detected_key = detect_key(dsl_code, chord_progression=chords)
        if detected_key:
            context['key'] = detected_key

    # Add rhythm/density analysis
    if include_analysis:
        # Note density
        density = analyze_note_density(dsl_code)
        context['density'] = density['density']
        context['notes_per_second'] = density['notes_per_second']

        # Harmonic rhythm
        harmonic_rhythm = analyze_harmonic_rhythm(dsl_code, chord_progression=chords)
        context['harmonic_pace'] = harmonic_rhythm['pace']

        # Melodic range
        melodic_range = analyze_melodic_range(dsl_code)
        context['register'] = melodic_range['register']

        # Voice leading analysis (for chord tracks)
        if chords and len(chords) >= 2:
            voice_leading = analyze_voice_leading(dsl_code, chord_progression=chords)
            context['voice_leading'] = voice_leading['guidance']
            context['voice_leading_suggestion'] = voice_leading['suggestion']

    return context
