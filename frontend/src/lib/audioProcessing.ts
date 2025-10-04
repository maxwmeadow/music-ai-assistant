
export async function blobToAudioBuffer(blob: Blob): Promise<AudioBuffer> {
    const arrayBuffer = await blob.arrayBuffer();
    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    try {
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        // close context if needed
        audioCtx.close();
        return audioBuffer;
    } catch (err) {
        audioCtx.close();
        throw err;
    }
}

/** Resample to targetSampleRate (e.g., 16000) and force mono. */
export async function resampleAudioBuffer(
    input: AudioBuffer,
    targetSampleRate = 16000
): Promise<AudioBuffer> {
    const channels = 1; // force mono
    const lengthInSec = input.duration;
    const offlineCtx = new OfflineAudioContext(channels, Math.ceil(lengthInSec * targetSampleRate), targetSampleRate);

    // create buffer source
    const bufferCopy = offlineCtx.createBuffer(input.numberOfChannels, input.length, input.sampleRate);

    // mixdown to mono by averaging channels
    for (let ch = 0; ch < input.numberOfChannels; ch++) {
        const inData = input.getChannelData(ch);
        for (let i = 0; i < inData.length; i++) {
            bufferCopy.getChannelData(0)[i] = (bufferCopy.getChannelData(0)[i] || 0) + inData[i] / input.numberOfChannels;
        }
    }

    const source = offlineCtx.createBufferSource();
    source.buffer = bufferCopy;
    source.connect(offlineCtx.destination);
    source.start(0);
    const rendered = await offlineCtx.startRendering();
    return rendered;
}

/** Normalize to peak 1.0 (avoid clipping) */
export function normalizeFloat32Array(arr: Float32Array): Float32Array {
    let max = 0;
    for (let i = 0; i < arr.length; i++) {
        const v = Math.abs(arr[i]);
        if (v > max) max = v;
    }
    if (max === 0 || max === 1) return arr;
    const normFactor = 1 / max;
    const out = new Float32Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] * normFactor;
    return out;
}

/** Convert AudioBuffer -> Float32Array (mono) */
export function audioBufferToFloat32(audioBuffer: AudioBuffer): Float32Array {
    // we force reading the first channel (resampleAudioBuffer already made it mono)
    const ch = audioBuffer.getChannelData(0);
    return new Float32Array(ch);
}

/** Encode Float32Array to 16-bit PCM WAV Blob (mono) */
export function floatTo16BitPCM(float32Array: Float32Array): ArrayBuffer {
    const buffer = new ArrayBuffer(float32Array.length * 2);
    const view = new DataView(buffer);
    let offset = 0;
    for (let i = 0; i < float32Array.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, float32Array[i]));
        s = s < 0 ? s * 0x8000 : s * 0x7fff;
        view.setInt16(offset, s, true);
    }
    return buffer;
}

export function encodeWAV(float32Array: Float32Array, sampleRate = 16000): Blob {
    const pcmBuffer = floatTo16BitPCM(float32Array);
    const wavBuffer = new ArrayBuffer(44 + pcmBuffer.byteLength);
    const view = new DataView(wavBuffer);

    /* RIFF identifier */
    writeString(view, 0, "RIFF");
    /* file length */
    view.setUint32(4, 36 + pcmBuffer.byteLength, true);
    /* RIFF type */
    writeString(view, 8, "WAVE");
    /* format chunk identifier */
    writeString(view, 12, "fmt ");
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sampleRate * blockAlign) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, "data");
    /* data chunk length */
    view.setUint32(40, pcmBuffer.byteLength, true);

    // write PCM samples
    const pcmView = new Uint8Array(wavBuffer, 44);
    pcmView.set(new Uint8Array(pcmBuffer));

    return new Blob([wavBuffer], { type: "audio/wav" });
}

function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

/** Full pipeline: blob -> Float32Array (16k mono normalized) + WAV blob */
export async function processAudioBlob(
    rawBlob: Blob,
    targetSampleRate = 16000
): Promise<{ float32: Float32Array; wav: Blob }> {
    // decode to AudioBuffer
    const decoded = await blobToAudioBuffer(rawBlob);

    // resample to target sample rate & mono
    const resampled = await resampleAudioBuffer(decoded, targetSampleRate);

    // convert to float array
    const floatArr = audioBufferToFloat32(resampled);

    // normalize amplitude
    const normalized = normalizeFloat32Array(floatArr);

    // create WAV Blob (16-bit PCM)
    const wav = encodeWAV(normalized, targetSampleRate);

    return { float32: normalized, wav };
}
