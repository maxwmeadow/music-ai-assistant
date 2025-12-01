/**
 * Export functionality for Music AI Assistant projects
 * Exports complete project state to JSON file (.maa format)
 */

export interface ProjectFile {
  version: string;           // "1.0.0"
  created: string;           // ISO timestamp
  modified: string;          // ISO timestamp
  metadata: {
    title: string;
    tempo: number;
    key: string;
    timeSignature: string;
  };
  tracks: any[];            // Track array from IR
  settings: {
    loopEnabled: boolean;
    loopStart: number;
    loopEnd: number;
    trackVolumes: Record<string, number>;
    trackPans: Record<string, number>;
    trackMutes: Record<string, boolean>;
    trackSolos: Record<string, boolean>;
    masterVolume: number;
    masterPan: number;
  };
  dsl: string;              // Raw DSL code
  ir?: any;                 // Optional IR representation
}

/**
 * Export current project to downloadable JSON file
 * @param data - Project data to export
 * @param filename - Optional custom filename (without extension)
 */
export function exportProject(data: ProjectFile, filename?: string): void {
  try {
    // Generate JSON with proper formatting
    const jsonString = JSON.stringify(data, null, 2);

    // Create Blob
    const blob = new Blob([jsonString], { type: 'application/json' });

    // Generate filename
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const finalFilename = filename
      ? `${filename}.maa`
      : `${data.metadata.title || 'project'}_${timestamp}.maa`;

    // Create download link and trigger
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = finalFilename;
    document.body.appendChild(link);
    link.click();

    // Cleanup
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    console.log(`[Export] Successfully exported project: ${finalFilename}`);
  } catch (error) {
    console.error('[Export] Failed to export project:', error);
    throw new Error('Failed to export project');
  }
}

/**
 * Create a ProjectFile object from current app state
 */
export function createProjectFile(
  dsl: string,
  tracks: any[],
  metadata: {
    title?: string;
    tempo?: number;
    key?: string;
    timeSignature?: string;
  },
  settings: {
    loopEnabled?: boolean;
    loopStart?: number;
    loopEnd?: number;
    trackVolumes?: Record<string, number>;
    trackPans?: Record<string, number>;
    trackMutes?: Record<string, boolean>;
    trackSolos?: Record<string, boolean>;
    masterVolume?: number;
    masterPan?: number;
  },
  ir?: any
): ProjectFile {
  const now = new Date().toISOString();

  return {
    version: '1.0.0',
    created: now,
    modified: now,
    metadata: {
      title: metadata.title || 'Untitled Project',
      tempo: metadata.tempo || 120,
      key: metadata.key || 'C',
      timeSignature: metadata.timeSignature || '4/4',
    },
    tracks,
    settings: {
      loopEnabled: settings.loopEnabled || false,
      loopStart: settings.loopStart || 0,
      loopEnd: settings.loopEnd || 0,
      trackVolumes: settings.trackVolumes || {},
      trackPans: settings.trackPans || {},
      trackMutes: settings.trackMutes || {},
      trackSolos: settings.trackSolos || {},
      masterVolume: settings.masterVolume ?? 0,
      masterPan: settings.masterPan ?? 0,
    },
    dsl,
    ir,
  };
}
