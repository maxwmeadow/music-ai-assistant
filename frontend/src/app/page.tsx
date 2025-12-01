"use client";

import { useState, useEffect } from "react";
import { CodeEditor } from "@/components/CodeEditor";
import { MixerPanel } from "@/components/MixerPanel";
import { api } from "@/lib/api";
import { RecorderControls } from "@/components/RecorderControls";
import { parseTracksFromDSL, ParsedTrack } from "@/lib/dslParser";
import { Mic, Music, Drum, Play, Square, Sparkles, Sliders, Piano, ChevronDown, Undo, Redo, Radio, Repeat } from "lucide-react";
import { Timeline } from "@/components/Timeline/Timeline";
import { PianoRoll } from "@/components/PianoRoll/PianoRoll";
import DetectionTuner from "@/components/DetectionTuner";
import { compileIR, VisualizationData } from "@/lib/hum2melody-api";
import { useHistory } from "@/hooks/useHistory";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";
import { AudioService } from "@/services/audioService";
import { DSLService } from "@/services/dslService";
import { FileMenu } from "@/components/FileMenu";
import { ProjectFile } from "@/lib/export";
import { TrackNameModal } from "@/components/TrackNameModal";
import ArrangerModal, { ArrangementConfig } from "@/components/ArrangerModal";
import { toast, Toaster } from "sonner";
import { Tutorial } from "@/components/Tutorial";
import { TUTORIAL_STEPS } from "@/config/tutorialSteps";

export default function Home() {
  const { pushHistory, undo, redo, canUndo, canRedo, currentState: code } = useHistory("// Your generated music code will appear here...");

  // Helper to update code and push to history
  const setCode = (newCode: string) => {
    pushHistory(newCode);
  };

  // Keyboard shortcuts modal state
  const [showKeyboardShortcuts, setShowKeyboardShortcuts] = useState(false);

  const [loadingTest, setLoadingTest] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [loadingPlay, setLoadingPlay] = useState(false);
  const [executableCode, setExecutableCode] = useState<string>("");
  const [tracks, setTracks] = useState<ParsedTrack[]>([]);
  const [trackVolumes, setTrackVolumes] = useState<Record<string, number>>({});
  const [trackPans, setTrackPans] = useState<Record<string, number>>({});
  const [trackMutes, setTrackMutes] = useState<Record<string, boolean>>({});
  const [trackSolos, setTrackSolos] = useState<Record<string, boolean>>({});
  const [masterVolume, setMasterVolume] = useState<number>(0);
  const [masterPan, setMasterPan] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [selectedTrackForPianoRoll, setSelectedTrackForPianoRoll] = useState<string | null>(null);
  const [soloedTrack, setSoloedTrack] = useState<string | null>(null);
  const [presoloVolumes, setPresoloVolumes] = useState<Record<string, number>>({});
  const [metronomeEnabled, setMetronomeEnabled] = useState(false);

  // Loop region state
  const [loopEnabled, setLoopEnabled] = useState(false);
  const [loopStart, setLoopStart] = useState(0);
  const [loopEnd, setLoopEnd] = useState(4);

  // Detection Tuning State
  const [tuningMode, setTuningMode] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [visualizationData, setVisualizationData] = useState<VisualizationData | null>(null);
  const [currentIR, setCurrentIR] = useState<any>(null);

  // Panel visibility
  const [showMixer, setShowMixer] = useState(false);
  const [showRecorder, setShowRecorder] = useState(false);
  const [recordingMode, setRecordingMode] = useState<'melody' | 'drums' | null>(null);

  // Track name modal state
  const [showTrackNameModal, setShowTrackNameModal] = useState(false);
  const [pendingIR, setPendingIR] = useState<any>(null);

  // Arranger modal state
  const [showArrangerModal, setShowArrangerModal] = useState(false);

  // Resizable panels
  const [leftPanelWidth, setLeftPanelWidth] = useState(50); // percentage
  const [isResizing, setIsResizing] = useState(false);

  const showToast = (message: string) => {
    toast(message);
  };

  const calculateMaxDuration = () => DSLService.calculateMaxDuration(code);

  const fetchTest = async () => {
    setLoadingTest(true);
    try {
      const response = await api("/test");
      const dslCode = await response.text();
      setCode(dslCode);

      const parsedTracks = parseTracksFromDSL(dslCode);
      setTracks(parsedTracks);

      showToast("Sample code loaded");
    } catch (error) {
      console.error(error);
      showToast("Failed to load sample");
    } finally {
      setLoadingTest(false);
    }
  };

  const sendToRunner = async () => {
    setLoadingRun(true);
    try {
      const response = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
      });

      const data = await response.json();
      const execCode = data.meta?.executable_code || "";
      setExecutableCode(execCode);

      const parsedTracks = parseTracksFromDSL(code);
      setTracks(parsedTracks);

      // Preload audio samples
      if (execCode) {
        setCurrentTime(0);
        setIsPlaying(false);
        await AudioService.preloadSamples(execCode, trackVolumes, setIsLoadingAudio);
      }

      showToast("Code compiled successfully");
    } catch (error) {
      console.error(error);
      showToast("Compilation failed");
      setIsLoadingAudio(false);
    } finally {
      setLoadingRun(false);
    }
  };

  const handleVolumeChange = (trackId: string, volume: number) => {
    setTrackVolumes(prev => ({ ...prev, [trackId]: volume }));

    if ((window as any).__musicControls?.pools) {
      const pool = (window as any).__musicControls.pools.get(trackId);
      if (pool?.voices) {
        pool.voices.forEach((voice: any) => {
          if (voice.volume) {
            // Respect the preGain that was calculated during instrument loading
            const preGain = voice.__preGain || 0;
            voice.volume.value = volume === -Infinity ? -Infinity : (preGain + volume);
          }
        });
      }
    }
  };

  const playAudio = async () => {
    if (!executableCode) {
      showToast("Please compile the code first");
      return;
    }

    if (isLoadingAudio) {
      showToast("Audio is still loading...");
      return;
    }

    AudioService.clearAutoStop();

    setLoadingPlay(true);
    setIsPlaying(true);

    try {
      await AudioService.initializeTone(trackVolumes);

      // Check if we're at or past the end - if so, reset to beginning
      const transportState = AudioService.getTransportState();
      const maxDuration = calculateMaxDuration();
      const currentPosition = transportState?.seconds || 0;

      if (currentPosition >= maxDuration - 0.1) {
        // At the end, reset to beginning
        AudioService.seek(0);
        setCurrentTime(0);
      }

      const updatedTransportState = AudioService.getTransportState();
      const isResuming = updatedTransportState && updatedTransportState.seconds > 0 && updatedTransportState.state === 'paused';

      // Load samples if not already preloaded
      if (!isResuming && !AudioService.areSamplesPreloaded()) {
        setIsLoadingAudio(true);
        eval(executableCode);
        AudioService.markSamplesPreloaded();
      }

      AudioService.startTransport();

      // Apply volume settings
      setTimeout(() => {
        Object.entries(trackVolumes).forEach(([trackId, volume]) => {
          handleVolumeChange(trackId, volume);
        });
      }, 100);

      // Calculate current position for auto-stop (use updated state after potential reset)
      const currentPositionForAutoStop = updatedTransportState?.seconds || 0;

      // Set auto-stop timeout (only if not looping)
      if (!loopEnabled) {
        const remainingTime = Math.max(0, maxDuration - currentPositionForAutoStop) + 1.5;
        AudioService.setAutoStop(remainingTime, stopAudio);
      }

      showToast(isResuming ? 'Resumed' : `Playing (${maxDuration.toFixed(1)}s)...`);
    } catch (error) {
      console.error(error);
      showToast("Playback error");
      setIsPlaying(false);
    } finally {
      setLoadingPlay(false);
    }
  };

  const stopAudio = () => {
    AudioService.clearAutoStop();
    const wasPlaying = AudioService.pauseTransport();
    setIsPlaying(false);
    if (wasPlaying) {
      showToast("Paused");
    }
  };

  const resetAudio = () => {
    AudioService.clearAutoStop();
    AudioService.stopTransport();
    setCurrentTime(0);
    setIsPlaying(false);
    showToast("Stopped");
  };

  const handleSeek = (time: number) => {
    if (isLoadingAudio) return;

    AudioService.seek(time);
    setCurrentTime(time);

    // Recalculate auto-stop timeout if playing (only if not looping)
    const transportState = AudioService.getTransportState();
    if (isPlaying && transportState?.state === 'started' && !loopEnabled) {
      const maxDuration = calculateMaxDuration();
      const remainingTime = Math.max(0, maxDuration - time) + 1.5;
      AudioService.setAutoStop(remainingTime, stopAudio);
    }
  };

  // Piano Roll handlers
  const handlePianoRollCompile = async () => {
    await sendToRunner();
    if (!isPlaying) {
      await playAudio();
    }
  };

  const handlePianoRollPlay = async () => {
    if (!executableCode) {
      showToast("Please compile first");
      return;
    }
    await playAudio();
  };

  const handlePianoRollSolo = (trackId: string) => {
    if (soloedTrack === trackId) {
      // Unsolo - restore all volumes from saved state
      setSoloedTrack(null);

      // Apply all volume changes atomically
      const pools = (window as any).__musicControls?.pools;
      if (pools) {
        tracks.forEach(track => {
          if (presoloVolumes.hasOwnProperty(track.id)) {
            const volume = presoloVolumes[track.id];
            const pool = pools.get(track.id);

            // Update state
            setTrackVolumes(prev => ({ ...prev, [track.id]: volume }));

            // Apply immediately to all voices
            if (pool?.voices) {
              pool.voices.forEach((voice: any) => {
                if (voice.volume) {
                  const preGain = voice.__preGain || 0;
                  voice.volume.value = volume === -Infinity ? -Infinity : (preGain + volume);
                }
              });
            }
          }
        });
      }

      setPresoloVolumes({});
    } else {
      // Save current volumes BEFORE making any changes
      const currentVolumes: Record<string, number> = {};
      tracks.forEach(track => {
        currentVolumes[track.id] = trackVolumes.hasOwnProperty(track.id) ? trackVolumes[track.id] : 0;
      });
      setPresoloVolumes(currentVolumes);
      setSoloedTrack(trackId);

      // Apply all volume changes atomically
      const pools = (window as any).__musicControls?.pools;
      if (pools) {
        tracks.forEach(track => {
          const volume = track.id === trackId ? (trackVolumes[track.id] || 0) : -Infinity;
          const pool = pools.get(track.id);

          // Update state
          setTrackVolumes(prev => ({ ...prev, [track.id]: volume }));

          // Apply immediately to all voices
          if (pool?.voices) {
            pool.voices.forEach((voice: any) => {
              if (voice.volume) {
                const preGain = voice.__preGain || 0;
                voice.volume.value = volume === -Infinity ? -Infinity : (preGain + volume);
              }
            });
          }
        });
      }
    }
  };

  // Metronome implementation using Tone.js
  useEffect(() => {
    if (typeof window === 'undefined') return;

    let metronome: any = null;
    let synth: any = null;

    const initMetronome = async () => {
      if (!metronomeEnabled || !isPlaying) {
        // Clean up existing metronome
        if ((window as any).__metronome) {
          (window as any).__metronome.stop();
          (window as any).__metronome.dispose();
          (window as any).__metronome = null;
        }
        if ((window as any).__metronomeSynth) {
          (window as any).__metronomeSynth.dispose();
          (window as any).__metronomeSynth = null;
        }
        if ((window as any).__metronomeChannel) {
          (window as any).__metronomeChannel.dispose();
          (window as any).__metronomeChannel = null;
        }
        return;
      }

      // Wait for Tone to be available
      if (!(window as any).Tone) {
        return;
      }

      const Tone = (window as any).Tone;
      await Tone.start();

      // Create a dedicated channel that bypasses Tone.Destination (so master volume doesn't affect it)
      const metronomeChannel = new Tone.Channel({
        volume: -6
      }).connect(Tone.context.rawContext.destination);

      // Create synth for metronome click - using FMSynth for punchy, clicky sound
      synth = new Tone.FMSynth({
        harmonicity: 3,
        modulationIndex: 10,
        oscillator: {
          type: 'sine'
        },
        envelope: {
          attack: 0.001,
          decay: 0.01,
          sustain: 0,
          release: 0.01
        },
        modulation: {
          type: 'square'
        },
        modulationEnvelope: {
          attack: 0.0002,
          decay: 0.02,
          sustain: 0,
          release: 0.01
        }
      }).connect(metronomeChannel);

      (window as any).__metronomeChannel = metronomeChannel;

      // Create loop that triggers on each beat
      metronome = new Tone.Loop((time: number) => {
        // Play different pitches for downbeat vs other beats
        const measure = Math.floor(Tone.Transport.position.split(':')[0]);
        const beat = Math.floor(Tone.Transport.position.split(':')[1]);

        // Higher pitch on downbeat (first beat of measure)
        const note = beat === 0 ? 'C5' : 'C4';
        synth.triggerAttackRelease(note, '16n', time);
      }, '4n'); // Trigger every quarter note

      metronome.start(0);

      // Store references for cleanup
      (window as any).__metronome = metronome;
      (window as any).__metronomeSynth = synth;
    };

    initMetronome();

    return () => {
      // Cleanup on unmount or when dependencies change
      if ((window as any).__metronome) {
        (window as any).__metronome.stop();
        (window as any).__metronome.dispose();
        (window as any).__metronome = null;
      }
      if ((window as any).__metronomeSynth) {
        (window as any).__metronomeSynth.dispose();
        (window as any).__metronomeSynth = null;
      }
      if ((window as any).__metronomeChannel) {
        (window as any).__metronomeChannel.dispose();
        (window as any).__metronomeChannel = null;
      }
    };
  }, [metronomeEnabled, isPlaying]);

  // Loop region implementation using Tone.Transport
  useEffect(() => {
    if (typeof window === 'undefined' || !(window as any).Tone) return;

    const Tone = (window as any).Tone;

    if (loopEnabled && isPlaying) {
      // Enable looping and set loop points
      Tone.Transport.loop = true;
      Tone.Transport.loopStart = loopStart;
      Tone.Transport.loopEnd = loopEnd;
    } else {
      // Disable looping
      Tone.Transport.loop = false;
    }
  }, [loopEnabled, loopStart, loopEnd, isPlaying]);

  // Auto-stop management: handle loop toggle during playback
  useEffect(() => {
    // Only react to loop toggle changes, not initial playback start
    if (!isPlaying) return;

    const transportState = AudioService.getTransportState();
    if (!transportState || transportState.state !== 'started') return;

    // Clear any existing auto-stop
    AudioService.clearAutoStop();

    if (!loopEnabled) {
      // When loop is disabled during playback, set auto-stop for remaining time
      const maxDuration = calculateMaxDuration();
      const currentPosition = transportState.seconds || 0;
      const remainingTime = Math.max(0, maxDuration - currentPosition) + 1.5;

      if (remainingTime > 0) {
        AudioService.setAutoStop(remainingTime, stopAudio);
      } else {
        // Already past the end, stop immediately
        stopAudio();
      }
    }
    // When loop is enabled, no auto-stop (plays indefinitely)
  }, [loopEnabled]);

  // Update transport time using requestAnimationFrame (smoother, less CPU)
  useEffect(() => {
    if (!isPlaying) return;

    let animationFrameId: number;
    let lastUpdateTime = 0;
    const UPDATE_THRESHOLD = 0.05; // Only update state if time changed by 50ms

    const updateTime = () => {
      const state = AudioService.getTransportState();
      if (state) {
        // Only update state if time changed significantly to reduce re-renders
        if (Math.abs(state.seconds - lastUpdateTime) >= UPDATE_THRESHOLD) {
          setCurrentTime(state.seconds);
          lastUpdateTime = state.seconds;
        }
      }
      animationFrameId = requestAnimationFrame(updateTime);
    };

    animationFrameId = requestAnimationFrame(updateTime);

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [isPlaying]);

  // Intercept console.log to detect when samples are loaded
  useEffect(() => {
    const originalLog = console.log;

    console.log = function (...args: any[]) {
      originalLog.apply(console, args);

      const message = args.join(' ');

      if (message.includes('[Music] Playing...')) {
        if (AudioService.isPreloading()) {
          AudioService.handlePreloadComplete();
        }
        setIsLoadingAudio(false);
      }
    };

    return () => {
      console.log = originalLog;
    };
  }, []);

  const handleMelodyGenerated = async (result: any) => {
    console.log("[DEBUG] handleMelodyGenerated called with:", result);
    console.log("[DEBUG] Has visualization?", !!result.visualization);
    console.log("[DEBUG] Has session_id?", !!result.session_id);

    try {
      // Check if user has existing code
      const hasExistingCode = code.trim() && code.trim() !== "// Your generated music code will appear here...";

      if (result.visualization && result.session_id) {
        console.log("[DEBUG] Opening tuning modal with session:", result.session_id);
        setSessionId(result.session_id);
        setVisualizationData(result.visualization);
        setCurrentIR(result.ir);
        setTuningMode(true);
        showToast("Tuning interface ready - adjust parameters to improve detection");
      } else if (recordingMode === 'drums' && result.visualization) {
        // Drums with visualization - store IR for after visualization closes
        console.log("[DEBUG] Drums visualization - storing IR for later");
        setPendingIR(result.ir);
        // Visualization modal will be shown by RecorderControls
        // Track name modal will be shown when visualization closes (via onVisualizationClose)
      } else {
        console.log("[DEBUG] No visualization data, using IR directly");

        if (hasExistingCode) {
          // Store IR and show track name modal
          setPendingIR(result.ir);
          setShowTrackNameModal(true);
          setShowRecorder(false);
          // Keep recordingMode for modal defaults - will be cleared when modal closes
        } else {
          // No existing code, just set it directly
          await applyIRAndCompile(result.ir);
          setShowRecorder(false);
          setRecordingMode(null);
        }
      }
    } catch (error) {
      console.error("Failed to process melody:", error);
      showToast("Failed to process melody");
    }

    // Close recorder after generation, UNLESS it's drums with visualization
    // (drums visualization modal is rendered inside RecorderControls)
    const isDrumsWithVisualization = recordingMode === 'drums' && result.visualization;
    if (!isDrumsWithVisualization) {
      setShowRecorder(false);
      // Don't clear recordingMode here if we're showing track name modal
      if (!result.visualization || result.session_id) {
        // Only clear if not waiting for track name modal
        const hasExistingCode = code.trim() && code.trim() !== "// Your generated music code will appear here...";
        if (!hasExistingCode) {
          setRecordingMode(null);
        }
      }
    }
  };

  const applyIRAndCompile = async (ir: any, append: boolean = false, trackName?: string, instrument?: string) => {
    showToast("Converting to DSL...");

    try {
      const response = await api("/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ir }),
      });

      const data = await response.json();

      if (data.dsl) {
        let finalDSL = data.dsl;

        // If appending, merge with existing DSL
        if (append && trackName) {
          try {
            finalDSL = DSLService.appendTrack(code, data.dsl, trackName, instrument);
          } catch (error) {
            console.error("Failed to append track:", error);
            showToast("Failed to append track");
            return;
          }
        }

        setCode(finalDSL);
        setExecutableCode(data.meta?.executable_code || "");

        const parsedTracks = parseTracksFromDSL(finalDSL);
        setTracks(parsedTracks);

        showToast(append ? "Track added!" : "Melody loaded! Click compile & play");
      } else {
        showToast("Failed to convert to DSL");
      }
    } catch (error) {
      console.error("Failed to convert IR to DSL:", error);
      showToast("Conversion failed");
    }
  };

  const handleApplyTuning = async (finalIR: any) => {
    setTuningMode(false);

    // Check if user has existing code - if so, show track name modal
    const hasExistingCode = code.trim() && code.trim() !== "// Your generated music code will appear here...";

    if (hasExistingCode) {
      // Store IR and show track name modal
      setPendingIR(finalIR);
      setShowTrackNameModal(true);
    } else {
      // No existing code, just set it directly
      await applyIRAndCompile(finalIR);
    }
  };

  const openRecorder = (mode: 'melody' | 'drums') => {
    setRecordingMode(mode);
    setShowRecorder(true);
  };

  const handleTrackNameConfirm = async (trackName: string, instrument: string) => {
    setShowTrackNameModal(false);
    if (pendingIR) {
      await applyIRAndCompile(pendingIR, true, trackName, instrument);
      setPendingIR(null);
    }
    setRecordingMode(null); // Clear recording mode after modal closes
  };

  const handleTrackNameCancel = () => {
    setShowTrackNameModal(false);
    setPendingIR(null);
    setRecordingMode(null); // Clear recording mode on cancel
    showToast("Track import cancelled");
  };

  const handleVisualizationClose = () => {
    // Called when drums visualization modal closes
    // Show track name modal if user has existing code and we have pending IR
    const hasExistingCode = code.trim() && code.trim() !== "// Your generated music code will appear here...";

    setShowRecorder(false);

    if (hasExistingCode && pendingIR) {
      // Show track name modal with the pending IR
      setShowTrackNameModal(true);
    } else if (pendingIR) {
      // No existing code, just apply directly
      applyIRAndCompile(pendingIR);
      setPendingIR(null);
      setRecordingMode(null);
    }
  };

  // File operations handlers
  const handleProjectImport = async (project: ProjectFile) => {
    try {
      setCode(project.dsl);

      // Parse tracks
      const parsedTracks = parseTracksFromDSL(project.dsl);
      setTracks(parsedTracks);

      // Restore IR if available
      if (project.ir) {
        setCurrentIR(project.ir);
      }

      // Restore all mixer settings if available
      if (project.settings?.trackVolumes) {
        setTrackVolumes(project.settings.trackVolumes);
      }
      if (project.settings?.trackPans) {
        setTrackPans(project.settings.trackPans);
      }
      if (project.settings?.trackMutes) {
        setTrackMutes(project.settings.trackMutes);
      }
      if (project.settings?.trackSolos) {
        setTrackSolos(project.settings.trackSolos);
      }
      if (project.settings?.masterVolume !== undefined) {
        setMasterVolume(project.settings.masterVolume);
      }
      if (project.settings?.masterPan !== undefined) {
        setMasterPan(project.settings.masterPan);
      }

      // Compile the imported DSL
      await sendToRunner();

      showToast(`Project "${project.metadata.title || 'Untitled'}" loaded`);
    } catch (error) {
      console.error('Failed to import project:', error);
      showToast('Failed to import project');
    }
  };

  const handleMIDIImport = async (ir: any) => {
    try {
      setCurrentIR(ir);
      await applyIRAndCompile(ir);
    } catch (error) {
      console.error('Failed to import MIDI:', error);
      showToast('Failed to import MIDI');
    }
  };

  // Handle arranger generation
  const handleArrangerGenerate = async (config: ArrangementConfig) => {
    try {
      showToast(`Generating ${config.trackType} track...`);

      // Call backend LLM arranger
      const response = await api("/arrange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dsl_code: code,
          track_type: config.trackType,
          genre: config.genre,
          custom_request: config.customRequest || null,
          creativity: config.creativity ?? 0.7,
          complexity: config.complexity || "medium"
        }),
      });

      const data = await response.json();

      if (data.status === "success" && data.generated_dsl) {
        // Insert the generated track into the editor
        // Add a newline before the generated track for formatting
        const newCode = code + "\n\n" + data.generated_dsl;
        setCode(newCode);
        showToast(`${config.trackType} track added!`);
      } else {
        showToast('Generation failed: No DSL returned');
      }
    } catch (error) {
      console.error('Failed to generate arrangement:', error);
      showToast(`Generation failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };

  // Handle resize drag
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  // Add/remove global mouse event listeners for resizing
  useEffect(() => {
    if (!isResizing) return;

    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = (e.clientX / window.innerWidth) * 100;
      // Constrain between 20% and 80%
      if (newWidth >= 20 && newWidth <= 80) {
        setLeftPanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);

  // Keyboard shortcuts for undo/redo, play/pause, file operations, and help
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if we're in an input/textarea (don't trigger shortcuts while typing)
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
        return;
      }

      // Check if Monaco editor is focused - if so, skip ALL our shortcuts to allow Monaco's native behavior
      const isMonacoFocused = target.classList.contains('monaco-editor') ||
        target.closest('.monaco-editor') ||
        target.classList.contains('view-line') ||
        target.classList.contains('inputarea') ||
        target.getAttribute('data-mode-id') !== null;

      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const ctrlOrCmd = isMac ? e.metaKey : e.ctrlKey;

      // If Monaco is focused, only allow our non-conflicting shortcuts (not Ctrl+Z, Ctrl+C, Ctrl+V, etc.)
      if (isMonacoFocused) {
        // Allow space to work normally in Monaco
        if (e.key === ' ') {
          return;
        }
        // Block undo/redo shortcuts - let Monaco handle them
        if (ctrlOrCmd && (e.key === 'z' || e.key === 'y')) {
          return;
        }
        // Allow all other keys (including Ctrl+C, Ctrl+V, Ctrl+X for copy/paste/cut)
        if (ctrlOrCmd && (e.key === 'c' || e.key === 'v' || e.key === 'x' || e.key === 'a')) {
          return;
        }
      }

      // File operations
      // Save Project: Ctrl+S
      if (ctrlOrCmd && e.key === 's' && !e.shiftKey) {
        e.preventDefault();
        // Trigger file menu export
        const exportButton = document.querySelector('[title*="Export project"]') as HTMLButtonElement;
        if (exportButton) exportButton.click();
      }

      // Open Project: Ctrl+O
      if (ctrlOrCmd && e.key === 'o') {
        e.preventDefault();
        const openButton = document.querySelector('[title*="Open project"]') as HTMLButtonElement;
        if (openButton) openButton.click();
      }

      // Export MIDI: Ctrl+E (without Shift)
      if (ctrlOrCmd && e.key === 'e' && !e.shiftKey) {
        e.preventDefault();
        const midiButton = document.querySelector('[title*="Export as MIDI"]') as HTMLButtonElement;
        if (midiButton) midiButton.click();
      }

      // Export Audio: Ctrl+Shift+E
      if (ctrlOrCmd && e.shiftKey && e.key === 'E') {
        e.preventDefault();
        const audioButton = document.querySelector('[title*="Export as WAV"]') as HTMLButtonElement;
        if (audioButton) audioButton.click();
      }

      // Play/Pause: Space
      if (e.key === ' ' && !ctrlOrCmd) {
        e.preventDefault();
        if (isPlaying) {
          stopAudio();
        } else {
          playAudio();
        }
      }

      // Show keyboard shortcuts: ?
      if (e.key === '?' && !ctrlOrCmd) {
        e.preventDefault();
        setShowKeyboardShortcuts(true);
      }

      // Undo: Ctrl+Z / Cmd+Z
      if (ctrlOrCmd && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        undo();
        showToast('Undo');
      }

      // Redo: Ctrl+Y / Cmd+Y or Ctrl+Shift+Z / Cmd+Shift+Z
      if ((ctrlOrCmd && e.key === 'y') || (ctrlOrCmd && e.shiftKey && e.key === 'z')) {
        e.preventDefault();
        redo();
        showToast('Redo');
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [undo, redo, isPlaying, isLoadingAudio, executableCode, playAudio, stopAudio]);

  return (
    <div className="h-screen flex flex-col bg-[#1a1a1a] overflow-hidden">
      {/* Tutorial for first-time users */}
      <Tutorial steps={TUTORIAL_STEPS} />

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcuts
        isOpen={showKeyboardShortcuts}
        onClose={() => setShowKeyboardShortcuts(false)}
      />

      {/* Detection Tuner Modal */}
      {tuningMode && visualizationData && sessionId && currentIR && (
        <DetectionTuner
          sessionId={sessionId}
          initialVisualization={visualizationData}
          initialIR={currentIR}
          onApply={handleApplyTuning}
          onCancel={() => {
            setTuningMode(false);
            showToast("Detection cancelled");
            // Don't apply anything when user cancels!
          }}
        />
      )}

      {/* Track Name Modal */}
      <TrackNameModal
        isOpen={showTrackNameModal}
        defaultTrackName={recordingMode === 'drums' ? 'drums' : 'melody'}
        defaultInstrument={recordingMode === 'drums' ? 'drums/bedroom_drums' : 'piano/steinway_grand'}
        onConfirm={handleTrackNameConfirm}
        onCancel={handleTrackNameCancel}
      />

      {/* Arranger Modal */}
      <ArrangerModal
        isOpen={showArrangerModal}
        onClose={() => setShowArrangerModal(false)}
        onGenerate={handleArrangerGenerate}
      />

      {/* Toast */}
      <Toaster
        position="top-right"
        theme="dark"
        richColors
        expand={false}
        toastOptions={{
          style: {
            background: '#2a2a2a',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            color: '#fff',
            padding: '12px 16px',
            borderRadius: '8px',
            fontSize: '14px',
            maxWidth: 'fit-content',
            minWidth: '200px',
          },
        }}
      />

      {/* Recorder Modal */}
      {showRecorder && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[#252525] border border-gray-700 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2">
                {recordingMode === 'melody' ? (
                  <>
                    <Mic className="w-5 h-5 text-blue-400" />
                    Record Melody
                  </>
                ) : (
                  <>
                    <Drum className="w-5 h-5 text-orange-400" />
                    Record Drums
                  </>
                )}
              </h2>
              <button
                onClick={() => {
                  setShowRecorder(false);
                  setRecordingMode(null);
                }}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <RecorderControls
              mode={recordingMode || 'melody'}
              onMelodyGenerated={handleMelodyGenerated}
              onVisualizationClose={handleVisualizationClose}
            />
          </div>
        </div>
      )}

      {/* Top Toolbar */}
      <div className="flex-none h-16 bg-[#252525] border-b border-gray-800 flex items-center px-4 gap-4 overflow-visible">
        {/* Logo */}
        <div className="flex items-center gap-2 mr-4 flex-shrink-0 select-none">
          <img
            src="/phonautoicon.png"
            alt="Phonauto"
            className="w-8 h-8 rounded-lg"
          />
          <span className="text-white font-bold text-lg">Phonauto</span>
        </div>

        {/* File Menu */}
        <div className="border-r border-gray-700 pr-4 flex-shrink-0">
          <FileMenu
            dslCode={code}
            tracks={tracks}
            currentIR={currentIR}
            executableCode={executableCode}
            metadata={{
              title: "Untitled Project",
              tempo: 120,
            }}
            settings={{
              trackVolumes: trackVolumes,
              trackPans: trackPans,
              trackMutes: trackMutes,
              trackSolos: trackSolos,
              masterVolume: masterVolume,
              masterPan: masterPan,
            }}
            onProjectImport={handleProjectImport}
            onMIDIImport={handleMIDIImport}
            onToast={showToast}
          />
        </div>

        {/* Model Buttons */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4 flex-shrink-0">
          <button
            id="hum2melody-button"
            onClick={() => openRecorder('melody')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors text-sm font-medium"
          >
            <Mic className="w-4 h-4" />
            Hum2Melody
          </button>
          <button
            id="beatbox2drums-button"
            onClick={() => openRecorder('drums')}
            className="flex items-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-500 text-white rounded-lg transition-colors text-sm font-medium"
          >
            <Drum className="w-4 h-4" />
            Beatbox2Drums
          </button>
          <button
            id="arranger-button"
            onClick={() => setShowArrangerModal(true)}
            disabled={!code || code.trim() === "// Your generated music code will appear here..."}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 disabled:text-gray-400 text-white rounded-lg text-sm font-medium transition-colors disabled:cursor-not-allowed"
            title={!code || code.trim() === "// Your generated music code will appear here..." ? "Add some music first" : "AI Arranger"}
          >
            <Sparkles className="w-4 h-4" />
            Arranger
          </button>
        </div>

        {/* Transport Controls */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4 flex-shrink-0">
          {!isPlaying ? (
            <button
              id="play-button"
              disabled={loadingPlay || !executableCode || isLoadingAudio}
              onClick={playAudio}
              className="flex items-center justify-center w-10 h-10 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              title={isLoadingAudio ? "Loading audio..." : "Play (Space)"}
            >
              <Play className="w-5 h-5" />
            </button>
          ) : (
            <button
              onClick={stopAudio}
              disabled={isLoadingAudio}
              className="flex items-center justify-center w-10 h-10 bg-red-600 hover:bg-red-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
              title="Stop (Space)"
            >
              <Square className="w-5 h-5" />
            </button>
          )}

          <button
            onClick={() => setMetronomeEnabled(!metronomeEnabled)}
            className={`flex items-center justify-center w-10 h-10 rounded-lg transition-colors ${metronomeEnabled
              ? 'bg-blue-600 hover:bg-blue-500 text-white'
              : 'bg-[#2a2a2a] hover:bg-[#333] text-gray-400 border border-gray-700'
              }`}
            title={metronomeEnabled ? "Metronome On (M)" : "Metronome Off (M)"}
          >
            <Radio className="w-5 h-5" />
          </button>

          {/* Loop Controls */}
          <button
            onClick={() => setLoopEnabled(!loopEnabled)}
            className={`flex items-center justify-center w-10 h-10 rounded-lg transition-colors ${loopEnabled
              ? 'bg-purple-600 hover:bg-purple-500 text-white'
              : 'bg-[#2a2a2a] hover:bg-[#333] text-gray-400 border border-gray-700'
              }`}
            title={loopEnabled ? `Loop ${loopStart.toFixed(1)}s - ${loopEnd.toFixed(1)}s` : "Loop Off (L)"}
          >
            <Repeat className="w-5 h-5" />
          </button>

          <button
            id="compile-button"
            disabled={loadingRun}
            onClick={sendToRunner}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 text-white rounded-lg text-sm font-medium transition-colors border border-gray-700"
          >
            {loadingRun ? "Compiling..." : "Compile"}
          </button>
        </div>

        {/* Undo/Redo Controls */}
        <div className="flex items-center gap-2 border-r border-gray-700 pr-4 flex-shrink-0">
          <button
            disabled={!canUndo}
            onClick={undo}
            className="flex items-center justify-center w-9 h-9 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-lg transition-colors border border-gray-700"
            title="Undo (Ctrl+Z)"
          >
            <Undo className="w-4 h-4" />
          </button>
          <button
            disabled={!canRedo}
            onClick={redo}
            className="flex items-center justify-center w-9 h-9 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-30 disabled:cursor-not-allowed text-white rounded-lg transition-colors border border-gray-700"
            title="Redo (Ctrl+Y)"
          >
            <Redo className="w-4 h-4" />
          </button>
        </div>

        {/* Panel Toggles */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <button
            id="mixer-button"
            onClick={() => setShowMixer(!showMixer)}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${showMixer
              ? 'bg-blue-600 text-white'
              : 'bg-[#2a2a2a] text-gray-300 hover:bg-[#333] border border-gray-700'
              }`}
          >
            <Sliders className="w-4 h-4" />
            Mixer
          </button>

          {selectedTrackForPianoRoll ? (
            <button
              onClick={() => setSelectedTrackForPianoRoll(null)}
              className="flex items-center gap-2 px-3 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium transition-colors"
            >
              <Piano className="w-4 h-4" />
              Close Piano Roll
            </button>
          ) : tracks.length > 0 && (
            <div className="relative group">
              <button
                className="flex items-center gap-2 px-3 py-2 bg-[#2a2a2a] text-gray-300 hover:bg-[#333] border border-gray-700 rounded-lg text-sm font-medium transition-colors"
              >
                <Piano className="w-4 h-4" />
                Piano Roll
                <ChevronDown className="w-4 h-4" />
              </button>
              <div className="absolute top-full left-0 pt-1 hidden group-hover:block z-[9999]">
                <div className="bg-[#2a2a2a] border border-gray-700 rounded-lg shadow-xl min-w-[150px]">
                  {tracks.map(track => (
                    <button
                      key={track.id}
                      onClick={() => setSelectedTrackForPianoRoll(track.id)}
                      className="block w-full text-left px-4 py-2 text-gray-300 hover:bg-[#333] first:rounded-t-lg last:rounded-b-lg transition-colors"
                    >
                      {track.id}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Right side - test button */}
        <div className="ml-auto flex-shrink-0">
          <button
            id="load-sample-button"
            disabled={loadingTest}
            onClick={fetchTest}
            className="px-4 py-2 bg-[#2a2a2a] hover:bg-[#333] disabled:opacity-50 text-gray-300 border border-gray-700 rounded-lg text-sm transition-colors"
          >
            {loadingTest ? "Loading..." : "Load Sample"}
          </button>
        </div>
      </div>

      {/* Main Content Area - 2 Panel Layout */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* LEFT PANEL - Timeline */}
        <div
          className="flex flex-col bg-[#1e1e1e] min-w-0"
          style={{ width: `${leftPanelWidth}%` }}
        >
          <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800">
            <h2 className="text-sm font-semibold text-gray-300 select-none">ARRANGEMENT</h2>
          </div>
          <div className="flex-1 overflow-hidden p-4">
            {tracks.length > 0 ? (
              <Timeline
                tracks={tracks}
                dslCode={code}
                onCodeChange={setCode}
                isPlaying={isPlaying}
                currentTime={currentTime}
                onSeek={handleSeek}
                isLoading={isLoadingAudio && !selectedTrackForPianoRoll}
                loopEnabled={loopEnabled}
                loopStart={loopStart}
                loopEnd={loopEnd}
                onLoopChange={(start, end) => {
                  setLoopStart(start);
                  setLoopEnd(end);
                }}
                onPlaybackStart={playAudio}
                onPlaybackStop={stopAudio}
                onMelodyGenerated={handleMelodyGenerated}
                onCompile={sendToRunner}
                executableCode={executableCode}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Music className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm select-none">No tracks yet</p>
                  <p className="text-xs mt-1 select-none">Record audio or load sample to get started</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* RESIZE BAR */}
        <div
          className={`w-1 bg-gray-800 hover:bg-blue-500 cursor-col-resize flex-shrink-0 relative group transition-colors ${isResizing ? 'bg-blue-500' : ''
            }`}
          onMouseDown={handleMouseDown}
        >
          {/* Visual indicator on hover */}
          <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-blue-500/20" />
        </div>

        {/* RIGHT PANEL - Code Editor */}
        <div
          className="flex flex-col bg-[#1e1e1e] min-w-0"
          style={{ width: `${100 - leftPanelWidth}%` }}
        >
          <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800">
            <h2 className="text-sm font-semibold text-gray-300 select-none">CODE EDITOR</h2>
          </div>
          <div className="flex-1 overflow-hidden">
            <CodeEditor value={code} onChange={setCode} />
          </div>
        </div>
      </div>

      {/* BOTTOM PANEL - Mixer (Toggleable) */}
      {showMixer && tracks.length > 0 && (
        <div className="flex-none h-56 bg-[#252525] border-t border-gray-800 overflow-x-auto overflow-y-hidden p-3">
          <MixerPanel
            tracks={tracks}
            trackVolumes={trackVolumes}
            trackPans={trackPans}
            trackMutes={trackMutes}
            trackSolos={trackSolos}
            masterVolume={masterVolume}
            masterPan={masterPan}
            onVolumeChange={handleVolumeChange}
            onPanChange={(trackId, pan) => setTrackPans(prev => ({ ...prev, [trackId]: pan }))}
            onMuteChange={(trackId, muted) => setTrackMutes(prev => ({ ...prev, [trackId]: muted }))}
            onSoloChange={(trackId, soloed) => setTrackSolos(prev => ({ ...prev, [trackId]: soloed }))}
            onMasterVolumeChange={setMasterVolume}
            onMasterPanChange={setMasterPan}
          />
        </div>
      )}

      {/* Piano Roll Overlay */}
      {selectedTrackForPianoRoll && (
        <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-40">
          <div className="bg-[#1e1e1e] border border-gray-700 rounded-xl w-[95vw] h-[90vh] flex flex-col shadow-2xl">
            <div className="flex-none px-4 py-3 bg-[#252525] border-b border-gray-800 flex items-center justify-between rounded-t-xl">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <Piano className="w-5 h-5 text-purple-400" />
                Piano Roll - {selectedTrackForPianoRoll}
              </h2>
              <button
                onClick={() => setSelectedTrackForPianoRoll(null)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 flex flex-col min-h-0">
              <PianoRoll
                track={tracks.find(t => t.id === selectedTrackForPianoRoll)!}
                dslCode={code}
                onCodeChange={setCode}
                isPlaying={isPlaying}
                currentTime={currentTime}
                onCompile={handlePianoRollCompile}
                onPlay={handlePianoRollPlay}
                onStop={stopAudio}
                onSolo={() => handlePianoRollSolo(selectedTrackForPianoRoll)}
                isSoloed={soloedTrack === selectedTrackForPianoRoll}
                isLoading={isLoadingAudio}
                onSeek={handleSeek}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}