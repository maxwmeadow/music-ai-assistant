import { TutorialStep } from "@/components/Tutorial";

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    title: "Welcome to Phonauto!",
    description: "This is an AI-powered music creation tool. You can hum melodies, beatbox drums, and create complete songs. Let's take a quick tour of the main features!",
  },
  {
    title: "File Menu",
    description: "Use the File menu to save your projects, export to MIDI/WAV, or import existing files. Your work is automatically saved in your browser's local storage.",
    targetId: "file-menu",
    position: "bottom",
  },
  {
    title: "Hum2Melody",
    description: "Click here to record yourself humming. Our AI will convert your humming into musical notes that you can edit and play back!",
    targetId: "hum2melody-button",
    position: "bottom",
  },
  {
    title: "Beatbox2Drums",
    description: "Record beatbox sounds here. The AI will convert them into professional drum patterns automatically.",
    targetId: "beatbox2drums-button",
    position: "bottom",
  },
  {
    title: "AI Arranger",
    description: "Once you have some music, the AI Arranger can automatically add bass, chords, and other accompaniment to create a full arrangement.",
    targetId: "arranger-button",
    position: "bottom",
  },
  {
    title: "Mixer Controls",
    description: "Click here to adjust volume, panning, mute, and solo for each track. Perfect for balancing your mix!",
    targetId: "mixer-button",
    position: "bottom",
  },
  {
    title: "Load Sample",
    description: "Not sure where to start? Click 'Load Sample' to load an example project and see how everything works.",
    targetId: "load-sample-button",
    position: "bottom",
  },
  {
    title: "Compile & Play",
    description: "After making changes, click 'Compile' to prepare your music, then hit the Play button to hear your creation!",
    targetId: "compile-button",
    position: "bottom",
  },
  {
    title: "You're Ready!",
    description: "That's it! Start creating music by loading a sample or recording your own ideas. Have fun experimenting!",
  },
];
