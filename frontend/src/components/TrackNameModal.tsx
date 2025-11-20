"use client";

import { useState, useEffect } from "react";
import { getInstrumentsByCategory, type Instrument } from "@/lib/instrumentCatalog";

interface TrackNameModalProps {
  isOpen: boolean;
  defaultTrackName: string;
  defaultInstrument: string;
  onConfirm: (trackName: string, instrument: string) => void;
  onCancel: () => void;
}

export function TrackNameModal({
  isOpen,
  defaultTrackName,
  defaultInstrument,
  onConfirm,
  onCancel,
}: TrackNameModalProps) {
  const [trackName, setTrackName] = useState(defaultTrackName);
  const [selectedInstrument, setSelectedInstrument] = useState(defaultInstrument);
  const [searchQuery, setSearchQuery] = useState("");
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

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

  // Auto-expand categories when searching
  useEffect(() => {
    if (searchQuery.trim()) {
      const categoriesToExpand = new Set<string>();
      Object.entries(filteredCategories).forEach(([category, instruments]) => {
        if (instruments.length > 0) {
          categoriesToExpand.add(category);
        }
      });
      setExpandedCategories(categoriesToExpand);
    }
  }, [searchQuery, filteredCategories]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setTrackName(defaultTrackName);
      setSelectedInstrument(defaultInstrument);
      setSearchQuery("");
      setExpandedCategories(new Set());
    }
  }, [isOpen, defaultTrackName, defaultInstrument]);

  const toggleCategory = (category: string) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const handleSelectInstrument = (instrument: Instrument) => {
    setSelectedInstrument(instrument.path);
  };

  const handleConfirm = () => {
    if (trackName.trim()) {
      onConfirm(trackName.trim(), selectedInstrument);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && trackName.trim()) {
      handleConfirm();
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onCancel}>
      <div
        className="bg-[#1e1e1e] border border-gray-700 rounded-lg shadow-2xl w-[500px] max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Header */}
        <div className="p-4 border-b border-gray-700">
          <h2 className="text-xl font-bold text-white">Add Track</h2>
          <p className="text-sm text-gray-400 mt-1">Name your track and select an instrument</p>
        </div>

        {/* Track Name Input */}
        <div className="p-4 border-b border-gray-700">
          <label className="block text-sm font-medium text-gray-300 mb-2">Track Name</label>
          <input
            type="text"
            value={trackName}
            onChange={(e) => setTrackName(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            placeholder="Enter track name..."
            className="w-full bg-[#2a2a2a] text-white px-3 py-2 rounded text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
            autoFocus
          />
        </div>

        {/* Instrument Selection */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-4 pb-2">
            <label className="block text-sm font-medium text-gray-300 mb-2">Instrument</label>
            {/* Selected instrument display */}
            <div className="bg-[#2a2a2a] px-3 py-2 rounded text-sm border border-gray-600 mb-3">
              <div className="text-white font-medium truncate">{selectedInstrument}</div>
            </div>
            {/* Search bar */}
            <input
              type="text"
              placeholder="Search instruments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onClick={(e) => e.stopPropagation()}
              className="w-full bg-[#2a2a2a] text-white px-3 py-2 rounded text-sm border border-gray-600 focus:outline-none focus:border-blue-500"
            />
          </div>

          {/* Instrument list */}
          <div className="flex-1 overflow-y-auto px-4 pb-4 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-[#1e1e1e] [&::-webkit-scrollbar-thumb]:bg-gray-600 [&::-webkit-scrollbar-thumb]:rounded [&::-webkit-scrollbar-thumb:hover]:bg-gray-500">
            {Object.keys(filteredCategories).length === 0 ? (
              <div className="p-4 text-gray-500 text-sm text-center">
                No instruments found
              </div>
            ) : (
              Object.entries(filteredCategories).map(([category, instruments]) => (
                <div key={category} className="mb-2">
                  {/* Category folder button */}
                  <button
                    onClick={() => toggleCategory(category)}
                    className="w-full text-left px-3 py-2 bg-[#252525] hover:bg-[#2a2a2a] rounded flex items-center gap-2 transition-colors"
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
                      className={`w-full text-left px-4 py-2 pl-8 mt-1 rounded hover:bg-[#2a2a2a] text-gray-300 text-sm transition-colors flex items-center justify-between group ${
                        selectedInstrument === instrument.path ? 'bg-blue-600/30 border border-blue-500' : ''
                      }`}
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
        </div>

        {/* Footer with buttons */}
        <div className="p-4 border-t border-gray-700 bg-[#252525] flex gap-3 justify-end">
          <button
            onClick={onCancel}
            className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600 text-white text-sm font-medium transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleConfirm}
            disabled={!trackName.trim()}
            className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add Track
          </button>
        </div>
      </div>
    </div>
  );
}
