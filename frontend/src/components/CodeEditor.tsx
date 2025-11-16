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
    const [currentLineNumber, setCurrentLineNumber] = useState<number | null>(null);
    const [currentInstrumentMatch, setCurrentInstrumentMatch] = useState<RegExpMatchArray | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());
    const pickerRef = useRef<HTMLDivElement>(null);
    const buttonRef = useRef<HTMLButtonElement>(null);

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
            } else {
                // Clear when moving away from an instrument line
                setCurrentInstrumentMatch(null);
                setCurrentLineNumber(null);
            }
        });
    };

    const toggleCategory = (category: string) => {
        const newExpanded = new Set(expandedCategories);
        if (newExpanded.has(category)) {
            newExpanded.delete(category);
        } else {
            newExpanded.add(category);
        }
        setExpandedCategories(newExpanded);
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
        setExpandedCategories(new Set()); // Reset expanded categories
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

        const commandId = editorRef.current.addCommand(
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
                    setShowInstrumentPicker(true);
                }
            }
        );

        return () => {
            // Commands are automatically cleaned up when editor is disposed
            // No manual cleanup needed for commandId
        };
    }, []);

    // Auto-expand categories when searching
    useEffect(() => {
        if (searchQuery.trim()) {
            // Expand all categories that have matching instruments
            const categoriesToExpand = new Set<string>();
            Object.entries(filteredCategories).forEach(([category, instruments]) => {
                if (instruments.length > 0) {
                    categoriesToExpand.add(category);
                }
            });
            setExpandedCategories(categoriesToExpand);
        }
    }, [searchQuery, filteredCategories]);

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
                    ref={buttonRef}
                    onClick={() => setShowInstrumentPicker(true)}
                    className="absolute top-2 right-2 bg-blue-600 hover:bg-blue-500 text-white px-3 py-1 rounded text-xs font-medium transition-colors z-10"
                    title="Change Instrument (Ctrl+I)"
                >
                    🎹 Instruments
                </button>
            )}

            {/* Instrument Picker Dropdown */}
            {showInstrumentPicker && buttonRef.current && (
                <div
                    ref={pickerRef}
                    className="absolute bg-[#1e1e1e] border border-gray-700 rounded-lg shadow-2xl z-50 w-96 max-h-[500px] overflow-hidden flex flex-col"
                    style={{
                        top: `${buttonRef.current.offsetTop + buttonRef.current.offsetHeight + 4}px`,
                        right: '8px',
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

                    {/* Instrument list with custom scrollbar */}
                    <div className="overflow-y-auto flex-1 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-[#1e1e1e] [&::-webkit-scrollbar-thumb]:bg-gray-600 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-gray-500">
                        {Object.keys(filteredCategories).length === 0 ? (
                            <div className="p-4 text-gray-500 text-sm text-center">
                                No instruments found
                            </div>
                        ) : (
                            Object.entries(filteredCategories).map(([category, instruments]) => (
                                <div key={category}>
                                    {/* Category folder button */}
                                    <button
                                        onClick={() => toggleCategory(category)}
                                        className="w-full text-left px-3 py-2 bg-[#252525] hover:bg-[#2a2a2a] border-b border-gray-700 flex items-center gap-2 transition-colors"
                                    >
                                        <span className="text-gray-400 text-xs">
                                            {expandedCategories.has(category) ? '▼' : '▶'}
                                        </span>
                                        <h3 className="text-xs font-bold text-gray-300 uppercase flex-1">{category}</h3>
                                        <span className="text-xs text-gray-500">
                                            {instruments.length}
                                        </span>
                                    </button>

                                    {/* Instruments in category (collapsible) */}
                                    {expandedCategories.has(category) && instruments.map((instrument) => (
                                        <button
                                            key={instrument.path}
                                            onClick={() => handleSelectInstrument(instrument)}
                                            className="w-full text-left px-4 py-2 pl-8 hover:bg-[#2a2a2a] text-gray-300 text-sm transition-colors flex items-center justify-between group border-b border-gray-800"
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
