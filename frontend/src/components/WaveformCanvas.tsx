
"use client";

import { useEffect, useRef } from "react";

export function WaveformCanvas({ analyser, height = 80, width = 400 }: { analyser: AnalyserNode | null; height?: number; width?: number }) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const rafRef = useRef<number | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !analyser) return;

        const ctx = canvas.getContext("2d")!;
        canvas.width = width * devicePixelRatio;
        canvas.height = height * devicePixelRatio;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.scale(devicePixelRatio, devicePixelRatio);

        analyser.fftSize = 2048;
        const bufferLen = analyser.fftSize;
        const dataArray = new Uint8Array(bufferLen);

        function draw() {
            rafRef.current = requestAnimationFrame(draw);
            analyser?.getByteTimeDomainData(dataArray);

            ctx.fillStyle = "transparent";
            ctx.clearRect(0, 0, width, height);

            ctx.lineWidth = 2;
            ctx.strokeStyle = "#111827"; // Tailwind gray-900; color can be adjusted
            ctx.beginPath();

            const sliceWidth = (width * 1.0) / bufferLen;
            let x = 0;
            for (let i = 0; i < bufferLen; i++) {
                const v = dataArray[i] / 128.0; // 0..2
                const y = (v * height) / 2;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.lineTo(width, height / 2);
            ctx.stroke();
        }

        draw();

        return () => {
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, [analyser, height, width]);

    return <canvas ref={canvasRef} />;
}
