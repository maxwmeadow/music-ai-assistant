class DSLGenerator {
    generate(parsedData) {
        let dsl = `tempo(${parsedData.metadata.tempo})\n\n`;
        
        parsedData.tracks.forEach(track => {
            dsl += this.generateTrack(track);
        });
        
        return dsl;
    }

    generateTrack(track) {
        let trackCode = `track("${track.id}") {\n`;

        if (track.instrument) {
            trackCode += `  instrument("${track.instrument}")\n`;
        }

        if (track.notes) {
            track.notes.forEach(note => {
                const noteName = this.midiToNote(note.pitch);
                trackCode += `  note("${noteName}", ${note.duration}, ${note.velocity})\n`;
            });
        }

        if (track.samples) {
            track.samples.forEach(sample => {
                trackCode += `  ${sample.sample}(${sample.start})\n`;
            });
        }

        trackCode += `}\n\n`;
        return trackCode;
    }

    midiToNote(midi) {
        const notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        const octave = Math.floor(midi / 12) - 1;
        const note = notes[midi % 12];
        return note + octave;
    }

    compileDSLToToneJS(dslCode) {
        let toneCode = `
// Auto-generated Tone.js from DSL
async function playMusic() {
    console.log("Initializing music playback...");
    
    if (Tone.context.state !== 'running') {
        await Tone.start();
        console.log("Tone.js audio context started");
    }
    
    // Clear existing events
    Tone.Transport.cancel();
    console.log("Cleared existing transport events");
    
    const tracks = {};
`;

        const tempoMatch = dslCode.match(/tempo\((\d+)\)/);
        if (tempoMatch) {
            toneCode += `    Tone.Transport.bpm.value = ${tempoMatch[1]};\n`;
            toneCode += `    console.log("Set tempo to ${tempoMatch[1]} BPM");\n`;
        }

        const trackMatches = dslCode.match(/track\("([^"]+)"\)\s*{([^}]+)}/g);
        if (trackMatches) {
            trackMatches.forEach(trackMatch => {
                const trackIdMatch = trackMatch.match(/track\("([^"]+)"\)/);
                const trackId = trackIdMatch[1];

                if (trackMatch.includes('instrument(')) {
                    toneCode += `    tracks.${trackId} = new Tone.Synth().toDestination();\n`;
                    toneCode += `    console.log("Created synth for track: ${trackId}");\n`;
                }

                if (trackMatch.includes('kick(') || trackMatch.includes('snare(')) {
                    toneCode += `    tracks.${trackId} = {\n`;
                    toneCode += `        kick: new Tone.MembraneSynth().toDestination(),\n`;
                    toneCode += `        snare: new Tone.NoiseSynth().toDestination()\n`;
                    toneCode += `    };\n`;
                    toneCode += `    console.log("Created drum kit for track: ${trackId}");\n`;
                }

                const noteMatches = trackMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/g);
                if (noteMatches) {
                    noteMatches.forEach((noteMatch, index) => {
                        const [, note, duration, velocity] = noteMatch.match(/note\("([^"]+)",\s*([\d.]+),\s*([\d.]+)\)/);
                        const startTime = index * parseFloat(duration);
                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        tracks.${trackId}.triggerAttackRelease("${note}", ${duration}, time, ${velocity});\n`;
                        toneCode += `        console.log("Playing ${note} on ${trackId}");\n`;
                        toneCode += `    }, "${startTime}");\n`;
                    });
                }

                const kickMatches = trackMatch.match(/kick\(([\d.]+)\)/g);
                if (kickMatches) {
                    kickMatches.forEach(kickMatch => {
                        const [, startTime] = kickMatch.match(/kick\(([\d.]+)\)/);
                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        tracks.${trackId}.kick.triggerAttackRelease("C2", "8n", time);\n`;
                        toneCode += `        console.log("Playing kick at ${startTime}");\n`;
                        toneCode += `    }, "${startTime}");\n`;
                    });
                }
                
                const snareMatches = trackMatch.match(/snare\(([\d.]+)\)/g);
                if (snareMatches) {
                    snareMatches.forEach(snareMatch => {
                        const [, startTime] = snareMatch.match(/snare\(([\d.]+)\)/);
                        toneCode += `    Tone.Transport.schedule((time) => {\n`;
                        toneCode += `        tracks.${trackId}.snare.triggerAttackRelease("4n", time);\n`;
                        toneCode += `        console.log("Playing snare at ${startTime}");\n`;
                        toneCode += `    }, "${startTime}");\n`;
                    });
                }
            });
        }

        toneCode += `
    // Start transport
    Tone.Transport.start();
    console.log("Music playback started");
    
    // Stop after 8 seconds
    setTimeout(() => {
        Tone.Transport.stop();
        console.log("Music playback finished");
    }, 8000);
}

// Execute immediately
playMusic().catch(console.error);
`;
        
        return toneCode;
    }
}

module.exports = DSLGenerator;