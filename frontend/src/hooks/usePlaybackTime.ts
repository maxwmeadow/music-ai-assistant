import { useState, useEffect } from 'react';

export function usePlaybackTime(isPlaying: boolean) {
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!isPlaying) {
      // Don't reset to 0 immediately - let the Transport position persist
      // This allows seeking while paused
      if ((window as any).Tone?.Transport) {
        const transportTime = (window as any).Tone.Transport.seconds;
        setCurrentTime(transportTime);
      }
      return;
    }

    const interval = setInterval(() => {
      if ((window as any).Tone?.Transport) {
        const transportTime = (window as any).Tone.Transport.seconds;
        setCurrentTime(transportTime);
      }
    }, 50);

    return () => clearInterval(interval);
  }, [isPlaying]);

  return currentTime;
}