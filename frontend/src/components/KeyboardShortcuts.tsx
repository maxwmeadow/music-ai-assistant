"use client";

import { useEffect } from "react";
import { X } from "lucide-react";

interface KeyboardShortcutsProps {
  isOpen: boolean;
  onClose: () => void;
}

export function KeyboardShortcuts({ isOpen, onClose }: KeyboardShortcutsProps) {
  useEffect(() => {
    if (!isOpen) return;

    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const isMac = typeof window !== 'undefined' && navigator.platform.toUpperCase().indexOf('MAC') >= 0;
  const modKey = isMac ? '⌘' : 'Ctrl';

  const shortcuts = [
    { category: "Editing", items: [
      { keys: [modKey, 'Z'], description: "Undo" },
      { keys: [modKey, 'Y'], description: "Redo" },
      { keys: [modKey, 'Shift', 'Z'], description: "Redo (alternative)" },
    ]},
    { category: "Selection", items: [
      { keys: ['Click + Drag'], description: "Box select notes" },
      { keys: ['Shift', 'Click'], description: "Add/remove from selection" },
      { keys: [modKey, 'A'], description: "Select all" },
    ]},
    { category: "Clipboard", items: [
      { keys: [modKey, 'C'], description: "Copy selected notes" },
      { keys: [modKey, 'X'], description: "Cut selected notes" },
      { keys: [modKey, 'V'], description: "Paste notes" },
      { keys: [modKey, 'D'], description: "Duplicate selected notes" },
    ]},
    { category: "Notes", items: [
      { keys: ['Delete'], description: "Delete selected notes" },
      { keys: ['Backspace'], description: "Delete selected notes" },
    ]},
    { category: "Playback", items: [
      { keys: ['Space'], description: "Play/Pause" },
    ]},
    { category: "Help", items: [
      { keys: ['?'], description: "Show keyboard shortcuts" },
      { keys: ['Esc'], description: "Close modal" },
    ]},
  ];

  return (
      <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={onClose}
      >
        <div
            className="bg-[#1e1e1e] border border-gray-700 rounded-xl w-[600px] max-h-[80vh] overflow-auto shadow-2xl"
            onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 bg-[#252525] border-b border-gray-700 rounded-t-xl">
            <div>
              <h2 className="text-xl font-bold text-white">Keyboard Shortcuts</h2>
              <p className="text-sm text-gray-400 mt-1">Quick reference for all available shortcuts</p>
            </div>
            <button
                onClick={onClose}
                className="text-gray-400 hover:text-white transition-colors p-2"
                title="Close (Esc)"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {shortcuts.map((section, idx) => (
                <div key={idx}>
                  <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-3">
                    {section.category}
                  </h3>
                  <div className="space-y-2">
                    {section.items.map((shortcut, itemIdx) => (
                        <div
                            key={itemIdx}
                            className="flex items-center justify-between py-2 px-3 bg-[#252525] rounded-lg hover:bg-[#2a2a2a] transition-colors"
                        >
                          <span className="text-gray-300">{shortcut.description}</span>
                          <div className="flex items-center gap-1">
                            {shortcut.keys.map((key, keyIdx) => (
                                <span key={keyIdx} className="flex items-center gap-1">
                                  <kbd className="px-2 py-1 text-xs font-bold bg-gray-800 text-gray-200 border border-gray-600 rounded min-w-[2rem] text-center">
                                    {key}
                                  </kbd>
                                  {keyIdx < shortcut.keys.length - 1 && (
                                      <span className="text-gray-500 text-xs">+</span>
                                  )}
                                </span>
                            ))}
                          </div>
                        </div>
                    ))}
                  </div>
                </div>
            ))}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 bg-[#252525] border-t border-gray-700 rounded-b-xl text-center">
            <p className="text-sm text-gray-400">
              Press <kbd className="px-2 py-1 text-xs font-bold bg-gray-800 text-gray-200 border border-gray-600 rounded">?</kbd> anytime to view this reference
            </p>
          </div>
        </div>
      </div>
  );
}
