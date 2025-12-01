"use client";

import { useState, useRef } from 'react';
import { Download, Upload, FileText, Music, Save, FolderOpen, FileMusic } from 'lucide-react';
import { exportProject, createProjectFile, ProjectFile } from '@/lib/export';
import { importProject, isValidProjectFile } from '@/lib/import';
import { downloadMIDI } from '@/lib/midi-export';
import { importFromMIDI, isValidMIDIFile, getMIDIFileInfo } from '@/lib/midi-import';
import { exportAndDownloadAudio } from '@/lib/audio-export';
import { DSLService } from '@/services/dslService';

interface FileMenuProps {
  // Current app state
  dslCode: string;
  tracks: any[];
  currentIR: any;
  executableCode: string;
  metadata?: {
    title?: string;
    tempo?: number;
    key?: string;
    timeSignature?: string;
  };
  settings?: {
    loopEnabled?: boolean;
    loopStart?: number;
    loopEnd?: number;
    trackVolumes?: Record<string, number>;
    trackPans?: Record<string, number>;
    trackMutes?: Record<string, boolean>;
    trackSolos?: Record<string, boolean>;
    masterVolume?: number;
    masterPan?: number;
  };

  // State setters
  onProjectImport: (project: ProjectFile) => void;
  onMIDIImport: (ir: any) => void;
  onToast: (message: string) => void;
}

export function FileMenu({
  dslCode,
  tracks,
  currentIR,
  executableCode,
  metadata,
  settings,
  onProjectImport,
  onMIDIImport,
  onToast,
}: FileMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState(0);
  const projectInputRef = useRef<HTMLInputElement>(null);
  const midiInputRef = useRef<HTMLInputElement>(null);

  // Export Project (JSON)
  const handleExportProject = () => {
    try {
      const projectFile = createProjectFile(
        dslCode,
        tracks,
        metadata || {},
        settings || {},
        currentIR
      );

      exportProject(projectFile);
      onToast('✓ Project exported successfully');
      setIsOpen(false);
    } catch (error) {
      console.error('Export project failed:', error);
      onToast('✗ Failed to export project');
    }
  };

  // Import Project (JSON)
  const handleImportProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidProjectFile(file)) {
      onToast('✗ Invalid file type. Please select a .maa or .json file');
      return;
    }

    try {
      const project = await importProject(file);
      onProjectImport(project);
      onToast(`✓ Project "${project.metadata.title}" imported`);
      setIsOpen(false);
    } catch (error: any) {
      console.error('Import project failed:', error);
      onToast(`✗ Import failed: ${error.message}`);
    }

    // Reset input
    if (projectInputRef.current) {
      projectInputRef.current.value = '';
    }
  };

  // Export MIDI
  const handleExportMIDI = () => {
    if (!currentIR) {
      onToast('✗ No music to export. Generate or write some music first.');
      return;
    }

    try {
      downloadMIDI(currentIR, metadata?.title);
      onToast('✓ MIDI file exported');
      setIsOpen(false);
    } catch (error) {
      console.error('MIDI export failed:', error);
      onToast('✗ Failed to export MIDI');
    }
  };

  // Import MIDI
  const handleImportMIDI = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!isValidMIDIFile(file)) {
      onToast('✗ Invalid file type. Please select a .mid or .midi file');
      return;
    }

    try {
      // Get file info first
      const info = await getMIDIFileInfo(file);
      console.log('[FileMenu] MIDI file info:', info);

      // Import MIDI
      const ir = await importFromMIDI(file);
      onMIDIImport(ir);
      onToast(`✓ Imported ${info.trackCount} tracks from MIDI`);
      setIsOpen(false);
    } catch (error: any) {
      console.error('MIDI import failed:', error);
      onToast(`✗ Import failed: ${error.message}`);
    }

    // Reset input
    if (midiInputRef.current) {
      midiInputRef.current.value = '';
    }
  };

  // Export Audio (WAV)
  const handleExportAudio = async () => {
    if (!executableCode) {
      onToast('✗ No music to export. Compile your code first.');
      return;
    }

    try {
      setIsExporting(true);
      setExportProgress(0);

      // Calculate duration (with loop expansion)
      const duration = DSLService.calculateMaxDuration(dslCode) + 2; // Add 2s buffer for reverb/release
      console.log(`[FileMenu] Exporting audio (${duration}s)...`);

      await exportAndDownloadAudio(
        executableCode,
        duration,
        metadata?.title || 'project',
        {
          onProgress: (progress) => {
            setExportProgress(progress);
          },
          onComplete: () => {
            onToast('✓ Audio exported successfully');
            setIsExporting(false);
            setIsOpen(false);
          },
          onError: (error) => {
            console.error('Audio export error:', error);
            onToast(`✗ Export failed: ${error.message}`);
            setIsExporting(false);
          }
        }
      );
    } catch (error: any) {
      console.error('Audio export failed:', error);
      onToast(`✗ Export failed: ${error.message}`);
      setIsExporting(false);
    }
  };

  return (
    <div className="relative">
      {/* File Menu Button */}
      <button
        id="file-menu"
        onClick={() => setIsOpen(!isOpen)}
        className="px-4 py-2 rounded-lg border border-white/20 bg-gray-800 hover:bg-gray-700 text-white font-medium transition-colors flex items-center gap-2"
        title="File operations (Ctrl+Shift+F)"
      >
        <FolderOpen size={18} />
        File
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-12 left-0 bg-[#1e1e1e] border border-gray-700 rounded-lg shadow-2xl z-50 w-72 overflow-hidden">
          {/* Export Section */}
          <div className="border-b border-gray-700">
            <div className="px-3 py-2 bg-[#252525] text-xs font-bold text-gray-400 uppercase">
              Export
            </div>

            <button
              onClick={handleExportProject}
              className="w-full px-4 py-3 hover:bg-[#2a2a2a] text-left flex items-center gap-3 transition-colors"
              title="Export project as .maa file (Ctrl+S)"
            >
              <Save size={18} className="text-blue-400" />
              <div className="flex-1">
                <div className="text-white font-medium text-sm">Save Project</div>
                <div className="text-gray-500 text-xs">Export as .maa file</div>
              </div>
              <kbd className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">Ctrl+S</kbd>
            </button>

            <button
              onClick={handleExportMIDI}
              disabled={!currentIR}
              className="w-full px-4 py-3 hover:bg-[#2a2a2a] text-left flex items-center gap-3 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export as MIDI file (Ctrl+E)"
            >
              <FileMusic size={18} className="text-purple-400" />
              <div className="flex-1">
                <div className="text-white font-medium text-sm">Export MIDI</div>
                <div className="text-gray-500 text-xs">Standard MIDI format</div>
              </div>
              <kbd className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">Ctrl+E</kbd>
            </button>

            <button
              onClick={handleExportAudio}
              disabled={!executableCode || isExporting}
              className="w-full px-4 py-3 hover:bg-[#2a2a2a] text-left flex items-center gap-3 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Export as WAV audio (Ctrl+Shift+E)"
            >
              <Music size={18} className="text-green-400" />
              <div className="flex-1">
                <div className="text-white font-medium text-sm">
                  {isExporting ? `Exporting... ${exportProgress}%` : 'Export Audio'}
                </div>
                <div className="text-gray-500 text-xs">WAV format</div>
              </div>
              {!isExporting && (
                <kbd className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">Ctrl+Shift+E</kbd>
              )}
            </button>
          </div>

          {/* Import Section */}
          <div>
            <div className="px-3 py-2 bg-[#252525] text-xs font-bold text-gray-400 uppercase">
              Import
            </div>

            <button
              onClick={() => projectInputRef.current?.click()}
              className="w-full px-4 py-3 hover:bg-[#2a2a2a] text-left flex items-center gap-3 transition-colors"
              title="Open project file (Ctrl+O)"
            >
              <FolderOpen size={18} className="text-blue-400" />
              <div className="flex-1">
                <div className="text-white font-medium text-sm">Open Project</div>
                <div className="text-gray-500 text-xs">Load .maa or .json file</div>
              </div>
              <kbd className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">Ctrl+O</kbd>
            </button>

            <button
              onClick={() => midiInputRef.current?.click()}
              className="w-full px-4 py-3 hover:bg-[#2a2a2a] text-left flex items-center gap-3 transition-colors"
              title="Import MIDI file"
            >
              <Upload size={18} className="text-purple-400" />
              <div className="flex-1">
                <div className="text-white font-medium text-sm">Import MIDI</div>
                <div className="text-gray-500 text-xs">Load .mid or .midi file</div>
              </div>
            </button>
          </div>

          {/* Footer */}
          <div className="px-3 py-2 border-t border-gray-700 bg-[#252525] text-xs text-gray-500 text-center">
            Press <kbd className="px-1 bg-gray-700 rounded">Esc</kbd> to close
          </div>
        </div>
      )}

      {/* Hidden file inputs */}
      <input
        ref={projectInputRef}
        type="file"
        accept=".maa,.json"
        onChange={handleImportProject}
        className="hidden"
      />
      <input
        ref={midiInputRef}
        type="file"
        accept=".mid,.midi"
        onChange={handleImportMIDI}
        className="hidden"
      />

      {/* Click outside to close */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
}
