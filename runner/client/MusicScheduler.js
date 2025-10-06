/**
 * MusicScheduler.js - Advanced Note Scheduling
 * Handles timing and scheduling of musical events
 */

class MusicScheduler {
    constructor() {
        this.scheduledEvents = [];
        this.isPlaying = false;
        this.startTime = null;
    }

    scheduleNote(instrument, note, duration, time, velocity = 1) {
        const event = {
            type: 'note',
            instrument,
            note,
            duration,
            time,
            velocity
        };

        this.scheduledEvents.push(event);

        if (this.isPlaying) {
            this._executeNoteEvent(event);
        }
    }

    scheduleChord(instrument, notes, duration, time, velocity = 1) {
        const event = {
            type: 'chord',
            instrument,
            notes: Array.isArray(notes) ? notes : [notes],
            duration,
            time,
            velocity
        };

        this.scheduledEvents.push(event);

        if (this.isPlaying) {
            this._executeChordEvent(event);
        }
    }

    scheduleTempo(bpm, time) {
        const event = {
            type: 'tempo',
            bpm,
            time
        };

        this.scheduledEvents.push(event);

        if (this.isPlaying) {
            Tone.Transport.schedule(() => {
                Tone.Transport.bpm.value = bpm;
                console.log(`[MusicScheduler] Tempo changed to ${bpm} BPM`);
            }, time);
        }
    }

    _executeNoteEvent(event) {
        Tone.Transport.schedule((time) => {
            try {
                if (event.instrument.triggerChord) {
                    event.instrument.triggerAttackRelease(
                        event.note,
                        event.duration,
                        time,
                        event.velocity
                    );
                } else {
                    event.instrument.triggerAttackRelease(
                        event.note,
                        event.duration,
                        time,
                        event.velocity
                    );
                }
            } catch (error) {
                console.error('[MusicScheduler] Error playing note:', error);
            }
        }, event.time);
    }

    _executeChordEvent(event) {
        Tone.Transport.schedule((time) => {
            try {
                if (event.instrument.triggerChord) {
                    event.instrument.triggerChord(
                        event.notes,
                        event.duration,
                        time,
                        event.velocity
                    );
                } else if (event.instrument.triggerAttackRelease) {
                    event.notes.forEach(note => {
                        event.instrument.triggerAttackRelease(
                            note,
                            event.duration,
                            time,
                            event.velocity
                        );
                    });
                }
            } catch (error) {
                console.error('[MusicScheduler] Error playing chord:', error);
            }
        }, event.time);
    }

    async start(tempo = 120) {
        if (this.isPlaying) {
            console.warn('[MusicScheduler] Already playing');
            return;
        }

        if (Tone.context.state !== 'running') {
            await Tone.start();
        }

        Tone.Transport.cancel();
        Tone.Transport.bpm.value = tempo;

        this.isPlaying = true;
        this.startTime = Tone.now();

        console.log(`[MusicScheduler] Starting playback at ${tempo} BPM with ${this.scheduledEvents.length} events`);

        this.scheduledEvents.forEach(event => {
            switch (event.type) {
                case 'note':
                    this._executeNoteEvent(event);
                    break;
                case 'chord':
                    this._executeChordEvent(event);
                    break;
                case 'tempo':
                    Tone.Transport.schedule(() => {
                        Tone.Transport.bpm.value = event.bpm;
                    }, event.time);
                    break;
            }
        });

        Tone.Transport.start();
    }

    stop() {
        if (!this.isPlaying) return;

        Tone.Transport.stop();
        Tone.Transport.cancel();
        this.isPlaying = false;
        console.log('[MusicScheduler] Stopped playback');
    }

    pause() {
        if (!this.isPlaying) return;

        Tone.Transport.pause();
        console.log('[MusicScheduler] Paused playback');
    }

    resume() {
        if (!this.isPlaying) return;

        Tone.Transport.start();
        console.log('[MusicScheduler] Resumed playback');
    }

    clear() {
        this.stop();
        this.scheduledEvents = [];
        console.log('[MusicScheduler] Cleared all events');
    }

    getTotalDuration() {
        if (this.scheduledEvents.length === 0) return 0;

        let maxTime = 0;
        this.scheduledEvents.forEach(event => {
            const eventEnd = Tone.Time(event.time).toSeconds() +
                            Tone.Time(event.duration || 0).toSeconds();
            maxTime = Math.max(maxTime, eventEnd);
        });

        return maxTime;
    }

    getStats() {
        return {
            isPlaying: this.isPlaying,
            eventCount: this.scheduledEvents.length,
            totalDuration: this.getTotalDuration(),
            currentTime: this.isPlaying ? Tone.Transport.seconds : 0,
            bpm: Tone.Transport.bpm.value
        };
    }

    async renderToBuffer(duration = null) {
        const renderDuration = duration || this.getTotalDuration() + 2;

        console.log(`[MusicScheduler] Rendering ${renderDuration}s to buffer...`);

        const buffer = await Tone.Offline(async ({ transport }) => {
            this.scheduledEvents.forEach(event => {
                switch (event.type) {
                    case 'note':
                        transport.schedule((time) => {
                            event.instrument.triggerAttackRelease(
                                event.note,
                                event.duration,
                                time,
                                event.velocity
                            );
                        }, event.time);
                        break;
                    case 'chord':
                        transport.schedule((time) => {
                            event.notes.forEach(note => {
                                event.instrument.triggerAttackRelease(
                                    note,
                                    event.duration,
                                    time,
                                    event.velocity
                                );
                            });
                        }, event.time);
                        break;
                    case 'tempo':
                        transport.schedule(() => {
                            transport.bpm.value = event.bpm;
                        }, event.time);
                        break;
                }
            });

            transport.start();
        }, renderDuration);

        console.log('[MusicScheduler] Rendering complete');
        return buffer;
    }
}

export default MusicScheduler;