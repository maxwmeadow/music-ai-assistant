"use client";

import { useRef, useState, useEffect } from "react";
import Editor from "@monaco-editor/react";
import type * as Monaco from 'monaco-editor';
import { getInstrumentsByCategory, type Instrument } from "@/lib/instrumentCatalog";

export function CodeEditor({
                               value,
                               onChange,
                           }: {
    value: string;
    onChange: (v: string) => void;
}) {
    const editorRef = useRef<Monaco.editor.IStandaloneCodeEditor | null>(null);
    const [showInstrumentPicker, setShowInstrumentPicker] = useState(false);
    const [pickerPosition, setPickerPosition] = useState({ top: 0, left: 0 });
    const [currentLineNumber, setCurrentLineNumber] = useState<number | null>(null);
    const [currentInstrumentMatch, setCurrentInstrumentMatch] = useState<RegExpMatchArray | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const pickerRef = useRef<HTMLDivElement>(null);

    const instrumentsByCategory = getInstrumentsByCategory();

    // Filter instruments based on search
    const filteredCategories = Object.entries(instrumentsByCategory).reduce((acc, [category, instruments]) => {
        if (searchQuery.trim() === "") {
            acc[category] = instruments;
        } else {
            const filtered = instruments.filter(inst =>
                inst.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                inst.path.toLowerCase().includes(searchQuery.toLowerCase())
            );
            if (filtered.length > 0) {
                acc[category] = filtered;
            }
        }
        return acc;
    }, {} as Record<string, Instrument[]>);

    const handleEditorMount = (editor: Monaco.editor.IStandaloneCodeEditor, monaco: typeof Monaco) => {
        editorRef.current = editor;

        // Add decorations for instrument lines
        editor.onDidChangeCursorPosition((e) => {
            const model = editor.getModel();
            if (!model) return;

            const lineNumber = e.position.lineNumber;
            const lineContent = model.getLineContent(lineNumber);

            // Check if line contains instrument("...")
            const instrumentMatch = lineContent.match(/instrument\("([^"]*)"\)/);

            if (instrumentMatch) {
                // Store the match for later use
                setCurrentInstrumentMatch(instrumentMatch);
                setCurrentLineNumber(lineNumber);
            }
        });
    };

    const handleSelectInstrument = (instrument: Instrument) => {
        if (!editorRef.current || currentLineNumber === null || !currentInstrumentMatch) return;

        const model = editorRef.current.getModel();
        if (!model) return;

        const lineContent = model.getLineContent(currentLineNumber);
        const startIndex = lineContent.indexOf('instrument("') + 'instrument("'.length;
        const endIndex = lineContent.indexOf('")', startIndex);

        // Replace the instrument path
        const range = {
            startLineNumber: currentLineNumber,
            startColumn: startIndex + 1,
            endLineNumber: currentLineNumber,
            endColumn: endIndex + 1,
        };

        editorRef.current.executeEdits('instrument-picker', [{
            range,
            text: instrument.path,
        }]);

        setShowInstrumentPicker(false);
        setSearchQuery("");
    };

    const handleKeyDown = (e: React.KeyboardEvent) => {
        // Close picker on Escape
        if (e.key === 'Escape') {
            setShowInstrumentPicker(false);
            setSearchQuery("");
        }
    };

    // Close picker when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (pickerRef.current && !pickerRef.current.contains(event.target as Node)) {
                setShowInstrumentPicker(false);
                setSearchQuery("");
            }
        };

        if (showInstrumentPicker) {
            document.addEventListener('mousedown', handleClickOutside);
            return () => document.removeEventListener('mousedown', handleClickOutside);
        }
    }, [showInstrumentPicker]);

    // Add keyboard shortcut Ctrl/Cmd+I to open instrument picker when on instrument line
    useEffect(() => {
        if (!editorRef.current) return;

        const disposable = editorRef.current.addCommand(
            2087, // Monaco.KeyMod.CtrlCmd | Monaco.KeyCode.KeyI
            () => {
                const position = editorRef.current?.getPosition();
                if (!position || !editorRef.current) return;

                const model = editorRef.current.getModel();
                if (!model) return;

                const lineContent = model.getLineContent(position.lineNumber);
                const instrumentMatch = lineContent.match(/instrument\("([^"]*)"\)/);

                if (instrumentMatch) {
                    setCurrentLineNumber(position.lineNumber);
                    setCurrentInstrumentMatch(instrumentMatch);

                    // Calculate position for picker
                    const pos = editorRef.current.getScrolledVisiblePosition(position);
                    if (pos) {
                        setPickerPosition({
                            top: pos.top + pos.height,
                            left: pos.left,
                        });
                    }

                    setShowInstrumentPicker(true);
                }
            }
        );

        return () => {
            if (disposable) {
                disposable.dispose();
            }
        };
    }, []);

    return (
        <div className="relative h-full">
            <Editor
                height="100%"
                defaultLanguage="ruby"
                value={value}
                onChange={(v) => onChange(v || "")}
                theme="vs-dark"
                onMount={handleEditorMount}
                options={{
                    minimap: { enabled: false },
                    fontSize: 14,
                    scrollBeyondLastLine: false,
                    wordWrap: "on",
                }}
            />

            {/* Floating button to open instrument picker */}
            {currentLineNumber !== null && currentInstrumentMatch && editorRef.current && (
                <button
                    onClick={() => {
                        if (!editorRef.current) return;
                        const position = editorRef.current.getPosition();
                        if (!position) return;

                        const pos = editorRef.current.getScrolledVisiblePosition(position);
                        if (pos) {
                            setPickerPosition({
                                top: pos.top + pos.height,
                                left: pos.left,
                            });
                        }
                        setShowInstrumentPicker(true);
                    }}
                    className="absolute top-2 right-2 bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-medium transition-colors z-10"
                    title="Change Instrument (Ctrl+I)"
                >
                    🎹 Instruments
                </button>
            )}

            {/* Instrument Picker Dropdown */}
            {showInstrumentPicker && (
                <div
                    ref={pickerRef}
                    className="fixed bg-[#1e1e1e] border border-gray-700 rounded-lg shadow-2xl z-50 w-96 max-h-96 overflow-hidden flex flex-col"
                    style={{
                        top: `${Math.min(pickerPosition.top, window.innerHeight - 420)}px`,
                        left: `${Math.min(pickerPosition.left, window.innerWidth - 400)}px`,
                    }}
                    onKeyDown={handleKeyDown}
                >
                    {/* Search bar */}
                    <div className="p-3 border-b border-gray-700">
                        <input
                            type="text"
                            placeholder="Search instruments..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="w-full bg-[#2a2a2a] text-white px-3 py-2 rounded text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
                            autoFocus
                        />
                    </div>

                    {/* Instrument list */}
                    <div className="overflow-y-auto flex-1">
                        {Object.keys(filteredCategories).length === 0 ? (
                            <div className="p-4 text-gray-500 text-sm text-center">
                                No instruments found
                            </div>
                        ) : (
                            Object.entries(filteredCategories).map(([category, instruments]) => (
                                <div key={category}>
                                    {/* Category header */}
                                    <div className="sticky top-0 bg-[#2a2a2a] px-3 py-2 border-b border-gray-700">
                                        <h3 className="text-xs font-bold text-gray-400 uppercase">{category}</h3>
                                    </div>
                                    {/* Instruments in category */}
                                    {instruments.map((instrument) => (
                                        <button
                                            key={instrument.path}
                                            onClick={() => handleSelectInstrument(instrument)}
                                            className="w-full text-left px-4 py-2 hover:bg-[#2a2a2a] text-gray-300 text-sm transition-colors flex items-center justify-between group"
                                        >
                                            <div className="flex-1 min-w-0">
                                                <div className="font-medium text-white truncate">{instrument.name}</div>
                                                <div className="text-xs text-gray-500 truncate">{instrument.path}</div>
                                            </div>
                                            <div className="text-xs text-gray-600 ml-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                                {instrument.samples} samples
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            ))
                        )}
                    </div>

                    {/* Footer */}
                    <div className="p-2 border-t border-gray-700 bg-[#252525] text-xs text-gray-500 text-center">
                        Press <kbd className="px-1 bg-gray-700 rounded">Esc</kbd> to close
                    </div>
                </div>
            )}
        </div>
    );
}
