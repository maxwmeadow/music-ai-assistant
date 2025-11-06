// Instrument catalog utilities
import catalogData from '../data/catalog.json';

export interface Instrument {
  name: string;
  category: string;
  path: string;
  samples: number;
  size_mb: number;
  preset_type?: string;
}

export interface CategorizedInstruments {
  [category: string]: Instrument[];
}

/**
 * Get all instruments grouped by category
 */
export function getInstrumentsByCategory(): CategorizedInstruments {
  const instruments = catalogData.instruments as Instrument[];
  const categorized: CategorizedInstruments = {};

  // Remove duplicates based on path
  const uniqueInstruments = instruments.filter(
    (instrument, index, self) =>
      index === self.findIndex((t) => t.path === instrument.path)
  );

  // Group by category
  uniqueInstruments.forEach((instrument) => {
    if (!categorized[instrument.category]) {
      categorized[instrument.category] = [];
    }
    categorized[instrument.category].push(instrument);
  });

  // Sort each category alphabetically by name
  Object.keys(categorized).forEach((category) => {
    categorized[category].sort((a, b) => a.name.localeCompare(b.name));
  });

  return categorized;
}

/**
 * Get all category names sorted
 */
export function getCategories(): string[] {
  const categorized = getInstrumentsByCategory();
  return Object.keys(categorized).sort();
}

/**
 * Search instruments by name or path
 */
export function searchInstruments(query: string): Instrument[] {
  const instruments = catalogData.instruments as Instrument[];
  const lowerQuery = query.toLowerCase();

  // Remove duplicates based on path
  const uniqueInstruments = instruments.filter(
    (instrument, index, self) =>
      index === self.findIndex((t) => t.path === instrument.path)
  );

  return uniqueInstruments.filter(
    (instrument) =>
      instrument.name.toLowerCase().includes(lowerQuery) ||
      instrument.path.toLowerCase().includes(lowerQuery)
  );
}
