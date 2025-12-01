/**
 * Import functionality for Music AI Assistant projects
 * Loads saved project files and validates schema
 */

import { ProjectFile } from './export';

/**
 * Validation error class
 */
export class ProjectValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ProjectValidationError';
  }
}

/**
 * Import project from file
 * @param file - File object from input element
 * @returns Parsed and validated ProjectFile
 */
export async function importProject(file: File): Promise<ProjectFile> {
  try {
    // Read file as text
    const text = await file.text();

    // Parse JSON
    let data: any;
    try {
      data = JSON.parse(text);
    } catch (error) {
      throw new ProjectValidationError('Invalid JSON format - file is corrupted or not a valid project file');
    }

    // Validate schema
    validateProjectFile(data);

    console.log(`[Import] Successfully imported project: ${data.metadata.title || 'Untitled'}`);
    return data as ProjectFile;

  } catch (error) {
    if (error instanceof ProjectValidationError) {
      console.error('[Import] Validation error:', error.message);
      throw error;
    }
    console.error('[Import] Failed to import project:', error);
    throw new Error('Failed to import project file');
  }
}

/**
 * Validate project file structure
 * @param data - Parsed JSON data
 * @throws ProjectValidationError if validation fails
 */
function validateProjectFile(data: any): void {
  // Check for required top-level fields
  if (!data.version) {
    throw new ProjectValidationError('Missing required field: version');
  }

  if (!data.metadata) {
    throw new ProjectValidationError('Missing required field: metadata');
  }

  if (!data.dsl && !data.ir) {
    throw new ProjectValidationError('Project must contain either DSL code or IR data');
  }

  // Validate metadata
  const metadata = data.metadata;
  if (typeof metadata.tempo !== 'number' || metadata.tempo <= 0) {
    throw new ProjectValidationError('Invalid tempo value');
  }

  // Validate version compatibility
  const [major] = data.version.split('.');
  if (major !== '1') {
    throw new ProjectValidationError(
      `Incompatible project version: ${data.version}. This app supports version 1.x.x`
    );
  }

  // Validate tracks array
  if (data.tracks && !Array.isArray(data.tracks)) {
    throw new ProjectValidationError('Invalid tracks format - must be an array');
  }

  // Validate settings if present
  if (data.settings) {
    if (data.settings.loopEnabled !== undefined && typeof data.settings.loopEnabled !== 'boolean') {
      throw new ProjectValidationError('Invalid loopEnabled value - must be boolean');
    }

    if (data.settings.trackVolumes !== undefined && typeof data.settings.trackVolumes !== 'object') {
      throw new ProjectValidationError('Invalid trackVolumes format - must be an object');
    }
  }
}

/**
 * Check if file is valid project file based on extension
 * @param file - File object to check
 * @returns true if file has .maa or .json extension
 */
export function isValidProjectFile(file: File): boolean {
  const validExtensions = ['.maa', '.json'];
  return validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
}

/**
 * Extract title from project file for preview
 * @param file - File object
 * @returns Project title or filename
 */
export async function getProjectTitle(file: File): Promise<string> {
  try {
    const text = await file.text();
    const data = JSON.parse(text);
    return data.metadata?.title || file.name.replace(/\.(maa|json)$/, '');
  } catch {
    return file.name.replace(/\.(maa|json)$/, '');
  }
}
