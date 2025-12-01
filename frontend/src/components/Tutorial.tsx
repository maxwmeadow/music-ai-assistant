"use client";

import { useState, useEffect } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

export interface TutorialStep {
  title: string;
  description: string;
  targetId?: string; // Optional: ID of element to highlight
  position?: "top" | "bottom" | "left" | "right"; // Position of tooltip relative to target
}

interface TutorialProps {
  steps: TutorialStep[];
  onComplete?: () => void;
  localStorageKey?: string;
}

export function Tutorial({
  steps,
  onComplete,
  localStorageKey = "music-ai-tutorial-completed"
}: TutorialProps) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);
  const [isTransitioning, setIsTransitioning] = useState(false);

  useEffect(() => {
    // Check if user has already seen the tutorial
    const hasSeenTutorial = localStorage.getItem(localStorageKey);
    if (!hasSeenTutorial) {
      setIsVisible(true);
    }
  }, [localStorageKey]);

  useEffect(() => {
    if (!isVisible) return;

    const step = steps[currentStep];
    if (step.targetId) {
      // Start transition (hide tooltip and highlight)
      setIsTransitioning(true);
      setTargetRect(null);

      const element = document.getElementById(step.targetId);
      if (element) {
        // Scroll element into view first
        element.scrollIntoView({ behavior: "smooth", block: "center" });

        // Wait for scroll to complete, then show both highlight and tooltip together
        const timeout = setTimeout(() => {
          const rect = element.getBoundingClientRect();
          setTargetRect(rect);
          setIsTransitioning(false);
        }, 300); // 300ms should be enough for smooth scroll

        return () => clearTimeout(timeout);
      }
    } else {
      setTargetRect(null);
      setIsTransitioning(false);
    }
  }, [currentStep, isVisible, steps]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    handleComplete();
  };

  const handleComplete = () => {
    localStorage.setItem(localStorageKey, "true");
    setIsVisible(false);
    onComplete?.();
  };

  if (!isVisible) return null;

  const step = steps[currentStep];
  const progress = ((currentStep + 1) / steps.length) * 100;

  // Calculate tooltip position
  const getTooltipStyle = (): React.CSSProperties => {
    if (!targetRect) {
      // Center of screen if no target
      return {
        position: "fixed",
        top: "50%",
        left: "50%",
        transform: "translate(-50%, -50%)",
        zIndex: 10003,
      };
    }

    const position = step.position || "bottom";
    const padding = 20;
    const tooltipWidth = 448; // max-w-md = 28rem = 448px
    const tooltipHeight = 250; // Approximate height

    const style: React.CSSProperties = {
      position: "fixed",
      zIndex: 10003,
    };

    switch (position) {
      case "top":
        style.bottom = `${window.innerHeight - targetRect.top + padding}px`;
        style.left = `${Math.min(Math.max(targetRect.left + targetRect.width / 2, tooltipWidth / 2 + 20), window.innerWidth - tooltipWidth / 2 - 20)}px`;
        style.transform = "translateX(-50%)";
        break;
      case "bottom":
        style.top = `${Math.min(targetRect.bottom + padding, window.innerHeight - tooltipHeight - 20)}px`;
        style.left = `${Math.min(Math.max(targetRect.left + targetRect.width / 2, tooltipWidth / 2 + 20), window.innerWidth - tooltipWidth / 2 - 20)}px`;
        style.transform = "translateX(-50%)";
        break;
      case "left":
        style.top = `${Math.min(Math.max(targetRect.top + targetRect.height / 2, tooltipHeight / 2 + 20), window.innerHeight - tooltipHeight / 2 - 20)}px`;
        style.right = `${window.innerWidth - targetRect.left + padding}px`;
        style.transform = "translateY(-50%)";
        break;
      case "right":
        style.top = `${Math.min(Math.max(targetRect.top + targetRect.height / 2, tooltipHeight / 2 + 20), window.innerHeight - tooltipHeight / 2 - 20)}px`;
        style.left = `${Math.min(targetRect.right + padding, window.innerWidth - tooltipWidth - 20)}px`;
        style.transform = "translateY(-50%)";
        break;
    }

    return style;
  };

  return (
    <>
      {/* Overlay backdrop - no blur here, just for click handling */}
      <div className="fixed inset-0 bg-transparent z-[10000]" onClick={handleSkip} />

      {/* Highlight for target element */}
      {targetRect && (
        <>
          {/* Cut out a hole for the highlighted element by creating 4 overlays around it */}
          {/* Top overlay */}
          <div
            className="fixed bg-black/40 backdrop-blur-[2px] pointer-events-none z-[10001]"
            style={{
              top: 0,
              left: 0,
              right: 0,
              height: `${targetRect.top - 4}px`,
            }}
          />
          {/* Bottom overlay */}
          <div
            className="fixed bg-black/40 backdrop-blur-[2px] pointer-events-none z-[10001]"
            style={{
              top: `${targetRect.bottom + 4}px`,
              left: 0,
              right: 0,
              bottom: 0,
            }}
          />
          {/* Left overlay */}
          <div
            className="fixed bg-black/40 backdrop-blur-[2px] pointer-events-none z-[10001]"
            style={{
              top: `${targetRect.top - 4}px`,
              left: 0,
              width: `${targetRect.left - 4}px`,
              height: `${targetRect.height + 8}px`,
            }}
          />
          {/* Right overlay */}
          <div
            className="fixed bg-black/40 backdrop-blur-[2px] pointer-events-none z-[10001]"
            style={{
              top: `${targetRect.top - 4}px`,
              left: `${targetRect.right + 4}px`,
              right: 0,
              height: `${targetRect.height + 8}px`,
            }}
          />
          {/* Border highlight */}
          <div
            className="fixed border-4 border-purple-500 rounded-lg pointer-events-none z-[10002] shadow-2xl"
            style={{
              top: `${targetRect.top - 4}px`,
              left: `${targetRect.left - 4}px`,
              width: `${targetRect.width + 8}px`,
              height: `${targetRect.height + 8}px`,
            }}
          />
        </>
      )}

      {/* Tutorial tooltip - hidden during transitions */}
      {!isTransitioning && (
        <div
          className="bg-gray-900 border-2 border-purple-500 rounded-xl shadow-2xl max-w-md"
          style={getTooltipStyle()}
        >
        {/* Progress bar */}
        <div className="h-1 bg-gray-800 rounded-t-xl overflow-hidden">
          <div
            className="h-full bg-purple-600 transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1">
              <div className="text-xs text-purple-400 font-semibold mb-1">
                STEP {currentStep + 1} OF {steps.length}
              </div>
              <h3 className="text-xl font-bold text-white">{step.title}</h3>
            </div>
            <button
              onClick={handleSkip}
              className="text-gray-400 hover:text-white transition-colors ml-4"
              aria-label="Skip tutorial"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Description */}
          <p className="text-gray-300 text-sm leading-relaxed mb-6">
            {step.description}
          </p>

          {/* Navigation buttons */}
          <div className="flex items-center justify-between">
            <button
              onClick={handlePrevious}
              disabled={currentStep === 0}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-400 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
              Previous
            </button>

            <button
              onClick={handleSkip}
              className="px-4 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
            >
              Skip Tutorial
            </button>

            <button
              onClick={handleNext}
              className="flex items-center gap-2 px-6 py-2 text-sm font-semibold bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors"
            >
              {currentStep === steps.length - 1 ? "Finish" : "Next"}
              {currentStep < steps.length - 1 && <ChevronRight className="w-4 h-4" />}
            </button>
          </div>
        </div>
      </div>
      )}
    </>
  );
}
