/**
 * SampleCache - Manages sample metadata and loading
 * Implements caching and batching to minimize requests
 */

class SampleCache {
    constructor(cdnBaseUrl = 'https://pub-e7b8ae5d5dcb4e23b0bf02e7b966c2f7.r2.dev', runnerUrl = 'http://localhost:5001') {
        this.cdnBaseUrl = cdnBaseUrl;
        this.runnerUrl = runnerUrl;
        this.mappingCache = new Map(); // instrumentPath -> mapping
        this.sampleDataCache = new Map(); // full URL -> ArrayBuffer
        this.pendingMappings = new Map(); // instrumentPath -> Promise
        this.catalog = null;
    }

    /**
     * Load and cache the instrument catalog
     */
    async loadCatalog() {
        if (this.catalog) return this.catalog;

        try {
            // Catalog is served by the runner server, not CDN
            const response = await fetch(`${this.runnerUrl}/catalog.json`);
            this.catalog = await response.json();
            console.log('[SampleCache] Loaded catalog with', this.catalog.instruments.length, 'instruments');
            return this.catalog;
        } catch (error) {
            console.error('[SampleCache] Failed to load catalog:', error);
            throw error;
        }
    }

    /**
     * Get instrument info from catalog
     */
    getInstrumentInfo(instrumentPath) {
        if (!this.catalog) {
            throw new Error('Catalog not loaded. Call loadCatalog() first.');
        }
        return this.catalog.instruments.find(inst => inst.path === instrumentPath);
    }

    /**
     * Load and cache instrument mapping
     * Deduplicates concurrent requests for the same mapping
     */
    async getMapping(instrumentPath) {
        // Return cached mapping if available
        if (this.mappingCache.has(instrumentPath)) {
            return this.mappingCache.get(instrumentPath);
        }

        // Return pending promise if already loading
        if (this.pendingMappings.has(instrumentPath)) {
            return this.pendingMappings.get(instrumentPath);
        }

        // Create new loading promise
        const loadPromise = this._loadMapping(instrumentPath);
        this.pendingMappings.set(instrumentPath, loadPromise);

        try {
            const mapping = await loadPromise;
            this.mappingCache.set(instrumentPath, mapping);
            return mapping;
        } finally {
            this.pendingMappings.delete(instrumentPath);
        }
    }

    async _loadMapping(instrumentPath) {
        const url = `${this.cdnBaseUrl}/samples/${instrumentPath}/mapping.json`;
        console.log(`[SampleCache] Loading mapping: ${instrumentPath}`);

        try {
            const response = await fetch(url, { cache: 'force-cache' });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const mapping = await response.json();

            const noteCount = mapping.velocity_layers
                ? Object.keys(mapping.velocity_layers).length
                : (mapping.samples ? Object.keys(mapping.samples).length : 0);

            const format = mapping.velocity_layers ? 'velocity layers' : 'simple samples'
            console.log(`[SampleCache] Loaded mapping for ${instrumentPath}:`, noteCount, 'notes -', format);

            return mapping;
        } catch (error) {
            console.error(`[SampleCache] Failed to load mapping for ${instrumentPath}:`, error);
            throw error;
        }
    }

    /**
     * Build optimized URL map for Tone.Sampler
     * Selects appropriate velocity layer based on instrument type
     */
    buildSamplerUrls(mapping, instrumentPath) {
        const urls = {};
        const baseUrl = `${this.cdnBaseUrl}/samples/${instrumentPath}/`;

        // Handle velocity_layers format (complex multi-layer samples)
        if (mapping.velocity_layers) {
            for (const [note, layers] of Object.entries(mapping.velocity_layers)) {
                let sampleKey = note;

                // Map drum sounds to MIDI notes
                if (mapping.type === 'drums') {
                    const drumNoteMap = {
                        "kick": "C2",
                        "snare": "D2",
                        "snare_rimshot": "E2",
                        "snare_buzz": "D#2",
                        "hihat_closed": "F#2",
                        "tom": "G2",
                        "crash": "C#3",
                        "ride": "D#3",
                        "hihat_pedal": "G#2",
                        "hihat_open": "A#2"
                    };
                    sampleKey = drumNoteMap[note] || note;
                }

                // Select best sample based on instrument type
                let selectedLayer = null;

                // Prefer sustain samples for melodic instruments
                if (mapping.type !== 'drums') {
                    selectedLayer = layers.find(l =>
                        l.file.includes('Sustains') ||
                        l.file.includes('sus') ||
                        l.file.includes('sustain')
                    );
                }

                // Fall back to mid-velocity or first available
                if (!selectedLayer && layers.length > 0) {
                    selectedLayer = layers.find(l => l.file.includes('vel4') || l.file.includes('v3'))
                        || layers[Math.floor(layers.length / 2)];
                }

                if (selectedLayer) {
                    urls[sampleKey] = selectedLayer.file.split('/').map(encodeURIComponent).join('/');
                }
            }
        }
        // Handle simple samples format (direct note -> file mapping)
        else if (mapping.samples) {
            for (const [note, file] of Object.entries(mapping.samples)) {
                urls[note] = file.split('/').map(encodeURIComponent).join('/');
            }
        }

        console.log(`[SampleCache] Built ${Object.keys(urls).length} sample URLs for ${instrumentPath}`);
        return { urls, baseUrl };
    }

    /**
     * Preload specific samples into cache
     * Useful for critical instruments
     */
    async preloadSamples(urls, baseUrl) {
        const loadPromises = Object.entries(urls).map(async ([note, file]) => {
            const fullUrl = baseUrl + file;
            if (!this.sampleDataCache.has(fullUrl)) {
                try {
                    const response = await fetch(fullUrl);
                    const buffer = await response.arrayBuffer();
                    this.sampleDataCache.set(fullUrl, buffer);
                } catch (error) {
                    console.warn(`[SampleCache] Failed to preload ${note}:`, error.message);
                }
            }
        });

        await Promise.allSettled(loadPromises);
    }

    /**
     * Clear cache for memory management
     */
    clearCache(instrumentPath = null) {
        if (instrumentPath) {
            this.mappingCache.delete(instrumentPath);
            // Could also clear related sample data
        } else {
            this.mappingCache.clear();
            this.sampleDataCache.clear();
            console.log('[SampleCache] Cleared all caches');
        }
    }

    /**
     * Get cache statistics
     */
    getStats() {
        return {
            mappingsCached: this.mappingCache.size,
            samplesCached: this.sampleDataCache.size,
            pendingLoads: this.pendingMappings.size,
            catalogLoaded: !!this.catalog
        };
    }
}

// Export singleton instance
const sampleCache = new SampleCache();
export default sampleCache;