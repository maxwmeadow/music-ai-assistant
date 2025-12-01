
"use client";

import { useEffect, useRef } from "react";

export function WaveformCanvas({ analyser, height = 80, width = 400 }: { analyser: AnalyserNode | null; height?: number; width?: number }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rafRef = useRef<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext("2d")!;
        const displayWidth = width;
        const displayHeight = height;

        canvas.width = displayWidth * devicePixelRatio;
        canvas.height = displayHeight * devicePixelRatio;
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;
        ctx.scale(devicePixelRatio, devicePixelRatio);

        // If no analyser, just draw a flat line
        if (!analyser) {
            ctx.fillStyle = "transparent";
            ctx.clearRect(0, 0, displayWidth, displayHeight);
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#4B5563"; // Tailwind gray-600
            ctx.beginPath();
            ctx.moveTo(0, displayHeight / 2);
            ctx.lineTo(displayWidth, displayHeight / 2);
            ctx.stroke();
            return;
        }

        analyser.fftSize = 2048;
        const bufferLen = analyser.fftSize;
        const dataArray = new Uint8Array(bufferLen);

        function draw() {
            rafRef.current = requestAnimationFrame(draw);
            analyser?.getByteTimeDomainData(dataArray);

            ctx.fillStyle = "transparent";
            ctx.clearRect(0, 0, displayWidth, displayHeight);

            ctx.lineWidth = 2;
            ctx.strokeStyle = "#8B5CF6"; // Tailwind purple-500
            ctx.beginPath();

            const sliceWidth = (displayWidth * 1.0) / bufferLen;
            let x = 0;
            for (let i = 0; i < bufferLen; i++) {
                const v = dataArray[i] / 128.0; // 0..2
                const y = (v * displayHeight) / 2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.lineTo(displayWidth, displayHeight / 2);
            ctx.stroke();
        }

        draw();

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, [analyser, height, width]);

    return <canvas ref={canvasRef} className="w-full" />;
}
