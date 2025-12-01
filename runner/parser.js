class MusicJSONParser {
    parse(jsonString) {
        const data = JSON.parse(jsonString);
        console.log(`[PARSER] Processing ${data.tracks.length} tracks`);
        return {
            metadata: data.metadata,
            tracks: data.tracks
        };
    }
}

module.exports = MusicJSONParser;