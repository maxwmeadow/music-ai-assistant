'use client';

import { useState } from 'react';
import { X, Sparkles, Info, Lightbulb, Music, Piano, Waves, Music2, Music3, Activity, Circle } from 'lucide-react';

export interface ArrangementConfig {
  trackType: string;
  genre: string;
  customGenre?: string;
  customRequest?: string;
  temperature?: number;
  creativity?: number;
  complexity?: string;
}

interface ArrangerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGenerate: (config: ArrangementConfig) => Promise<void>;
}

// Track type definitions with icon components
const TRACK_TYPES = [
  {
    value: "bass",
    label: "Bass",
    description: "Low-end foundation",
    icon: Music
  },
  {
    value: "chords",
    label: "Chords",
    description: "Harmonic support",
    icon: Piano
  },
  {
    value: "pad",
    label: "Pad",
    description: "Atmospheric texture",
    icon: Waves
  },
  {
    value: "melody",
    label: "Melody",
    description: "Lead line",
    icon: Music2
  },
  {
    value: "counterMelody",
    label: "Counter",
    description: "Secondary melody",
    icon: Music3
  },
  {
    value: "arpeggio",
    label: "Arp",
    description: "Broken chords",
    icon: Activity
  },
  {
    value: "drums",
    label: "Drums",
    description: "Percussion",
    icon: Circle
  }
];

// Genre definitions
const GENRES = [
  { value: "pop", label: "Pop" },
  { value: "jazz", label: "Jazz" },
  { value: "electronic", label: "Electronic" },
  { value: "rock", label: "Rock" },
  { value: "classical", label: "Classical" },
  { value: "lofi", label: "Lo-fi" },
  { value: "ambient", label: "Ambient" },
  { value: "funk", label: "Funk" },
  { value: "custom", label: "Custom" }
];

export default function ArrangerModal({
  isOpen,
  onClose,
  onGenerate
}: ArrangerModalProps) {
  const [trackType, setTrackType] = useState('bass');
  const [genre, setGenre] = useState('pop');
  const [customGenre, setCustomGenre] = useState('');
  const [customRequest, setCustomRequest] = useState('');
  const [creativity, setCreativity] = useState(0.7);
  const [complexity, setComplexity] = useState('medium');
  const [isGenerating, setIsGenerating] = useState(false);

  const handleGenerate = async () => {
    setIsGenerating(true);

    try {
      const config: ArrangementConfig = {
        trackType,
        genre: genre === 'custom' ? customGenre : genre,
        customRequest: customRequest || undefined,
        creativity,
        complexity
      };

      await onGenerate(config);

      // Reset form and close on success
      setCustomRequest('');
      onClose();
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClose = () => {
    if (!isGenerating) {
      onClose();
    }
  };

  if (!isOpen) return null;

  const selectedTrackType = TRACK_TYPES.find(t => t.value === trackType);
  const selectedGenre = GENRES.find(g => g.value === genre);
  const isCustomGenre = genre === 'custom';
  const canGenerate = isCustomGenre ? customGenre.trim() !== '' : true;

  // Generate order summary
  const creativityLabel = creativity < 0.3 ? 'very safe' : creativity < 0.5 ? 'predictable' : creativity < 0.7 ? 'balanced' : creativity < 0.9 ? 'creative' : 'experimental';
  const genreLabel = isCustomGenre ? (customGenre || 'custom style') : selectedGenre?.label.toLowerCase() || 'pop';

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 rounded-lg border border-gray-700 w-full max-w-4xl shadow-2xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-700 bg-gray-900/95">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 bg-blue-600/20 rounded-lg flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-blue-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">AI Arranger</h2>
              <p className="text-xs text-gray-400">Generate complementary tracks</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            disabled={isGenerating}
            className="text-gray-400 hover:text-white transition-colors disabled:opacity-50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Body - Horizontal Layout */}
        <div className="flex flex-1 overflow-hidden">
          {/* Left Panel - Controls */}
          <div className="flex-1 overflow-y-auto p-5 space-y-4">
            {/* Track Type Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-2">
                TRACK TYPE
              </label>
              <div className="grid grid-cols-4 gap-1.5">
                {TRACK_TYPES.map(type => {
                  const IconComponent = type.icon;
                  return (
                    <button
                      key={type.value}
                      onClick={() => setTrackType(type.value)}
                      disabled={isGenerating}
                      className={`
                        p-2 rounded-md border text-center transition-all group
                        ${trackType === type.value
                          ? 'bg-blue-600/20 border-blue-500 text-white'
                          : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-blue-500/50'
                        }
                        disabled:opacity-50 disabled:cursor-not-allowed
                      `}
                      title={type.description}
                    >
                      <div className="flex items-center justify-center mb-0.5">
                        <IconComponent className={`w-5 h-5 ${trackType === type.value ? 'text-blue-400' : 'text-gray-400'}`} />
                      </div>
                      <div className="text-xs font-medium">{type.label}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Genre Selection */}
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-2">
                GENRE / STYLE
              </label>
              <div className="grid grid-cols-5 gap-1.5">
                {GENRES.map(g => (
                  <button
                    key={g.value}
                    onClick={() => setGenre(g.value)}
                    disabled={isGenerating}
                    className={`
                      px-2 py-1.5 rounded-md border text-xs font-medium transition-all
                      ${genre === g.value
                        ? 'bg-blue-600/20 border-blue-500 text-white'
                        : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-blue-500/50'
                      }
                      disabled:opacity-50 disabled:cursor-not-allowed
                    `}
                  >
                    {g.label}
                  </button>
                ))}
              </div>

              {/* Custom genre input */}
              {isCustomGenre && (
                <input
                  type="text"
                  value={customGenre}
                  onChange={(e) => setCustomGenre(e.target.value)}
                  placeholder="e.g., 'synthwave' or 'reggae'"
                  disabled={isGenerating}
                  className="w-full mt-2 bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-white
                             placeholder:text-gray-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                             outline-none disabled:opacity-50"
                />
              )}
            </div>

            {/* Creativity and Complexity - Side by Side */}
            <div className="grid grid-cols-2 gap-3">
              {/* Creativity Slider */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-medium text-gray-400">
                    CREATIVITY
                  </label>
                  <span className="text-xs font-semibold text-blue-400">
                    {creativityLabel.toUpperCase()}
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={creativity}
                  onChange={(e) => setCreativity(parseFloat(e.target.value))}
                  disabled={isGenerating}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                             [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3.5
                             [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:rounded-full
                             [&::-webkit-slider-thumb]:bg-blue-500 [&::-webkit-slider-thumb]:cursor-pointer
                             [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:h-3.5
                             [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-blue-500
                             [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer
                             disabled:opacity-50 disabled:cursor-not-allowed"
                />
                <p className="text-xs text-gray-500 mt-1.5 leading-tight">
                  {creativity < 0.4
                    ? 'Sticks to proven patterns and common progressions'
                    : creativity < 0.7
                    ? 'Mixes familiar structures with some variation'
                    : 'Explores unusual harmonies and rhythmic ideas'}
                </p>
              </div>

              {/* Complexity Selector */}
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-2">
                  COMPLEXITY
                </label>
                <div className="grid grid-cols-3 gap-1.5">
                  {(['simple', 'medium', 'complex'] as const).map((level) => (
                    <button
                      key={level}
                      onClick={() => setComplexity(level)}
                      disabled={isGenerating}
                      className={`
                        px-2 py-1.5 rounded-md border text-xs font-medium transition-all
                        ${complexity === level
                          ? 'bg-blue-600/20 border-blue-500 text-white'
                          : 'bg-gray-800 border-gray-700 text-gray-300 hover:border-blue-500/50'
                        }
                        disabled:opacity-50 disabled:cursor-not-allowed
                      `}
                    >
                      {level === 'simple' ? 'Simple' : level === 'medium' ? 'Medium' : 'Complex'}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-gray-500 mt-1.5 leading-tight">
                  {complexity === 'simple' && 'Fewer notes, basic rhythms, straightforward patterns'}
                  {complexity === 'medium' && 'Standard note density with moderate variation'}
                  {complexity === 'complex' && 'Dense textures, intricate rhythms, rich harmonies'}
                </p>
              </div>
            </div>

            {/* Custom Request */}
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-2">
                ADDITIONAL INSTRUCTIONS <span className="text-gray-500">(Optional)</span>
              </label>
              <textarea
                value={customRequest}
                onChange={(e) => setCustomRequest(e.target.value)}
                placeholder="e.g., 'make it groovy' or 'use walking bass'"
                rows={2}
                disabled={isGenerating}
                className="w-full bg-gray-800 border border-gray-700 rounded-md px-3 py-2 text-sm text-white
                           placeholder:text-gray-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500
                           outline-none resize-none disabled:opacity-50"
              />
            </div>

            {/* Info Box */}
            <div className="bg-gray-800/50 border border-gray-700/50 rounded-md p-3">
              <div className="flex items-start gap-2">
                <Lightbulb className="w-3.5 h-3.5 text-yellow-400 mt-0.5 flex-shrink-0" />
                <p className="text-xs text-gray-400 leading-relaxed">
                  <span className="font-medium text-gray-300">Creativity</span> affects how adventurous the AI is with harmonies and rhythms.
                  <span className="font-medium text-gray-300 ml-1">Complexity</span> controls note density and pattern intricacy.
                </p>
              </div>
            </div>
          </div>

          {/* Right Panel - Order Summary */}
          <div className="w-72 bg-gray-800/30 border-l border-gray-700 p-5 flex flex-col">
            <div className="flex items-center gap-2 mb-4">
              <Info className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-semibold text-white">Generation Order</h3>
            </div>

            <div className="flex-1 space-y-3">
              {/* Main Order */}
              <div className="bg-gray-900/50 border border-gray-700/50 rounded-lg p-4">
                <p className="text-sm text-gray-200 leading-relaxed">
                  Generate a <span className="font-semibold text-blue-400">{creativityLabel}</span>{' '}
                  <span className="font-semibold text-white">{selectedTrackType?.label.toLowerCase()}</span>{' '}
                  with <span className="font-semibold text-blue-400">{complexity}</span> complexity in{' '}
                  <span className="font-semibold text-white">{genreLabel}</span> style
                  {customRequest && <span>.</span>}
                </p>

                {customRequest && (
                  <div className="mt-3 pt-3 border-t border-gray-700/50">
                    <p className="text-xs font-medium text-gray-400 mb-1">Additional:</p>
                    <p className="text-sm text-gray-300 italic">"{customRequest}"</p>
                  </div>
                )}
              </div>

              {/* Parameter Details */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Track Type</span>
                  <span className="text-gray-300 font-medium flex items-center gap-1.5">
                    {selectedTrackType?.icon && (() => {
                      const IconComponent = selectedTrackType.icon;
                      return <IconComponent className="w-3.5 h-3.5 text-blue-400" />;
                    })()}
                    {selectedTrackType?.label}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Genre</span>
                  <span className="text-gray-300 font-medium">
                    {isCustomGenre ? (customGenre || 'Custom') : selectedGenre?.label}
                  </span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Creativity Level</span>
                  <span className="text-blue-400 font-medium">{(creativity * 100).toFixed(0)}%</span>
                </div>
                <div className="flex justify-between items-center text-xs">
                  <span className="text-gray-500">Complexity</span>
                  <span className="text-blue-400 font-medium capitalize">{complexity}</span>
                </div>
              </div>

              {/* What to Expect */}
              <div className="bg-gray-900/30 border border-gray-700/30 rounded-lg p-3">
                <p className="text-xs font-medium text-gray-400 mb-2">What to expect:</p>
                <ul className="text-xs text-gray-500 space-y-1">
                  <li>• Analyzes your existing tracks</li>
                  <li>• Detects key, chords, and rhythm</li>
                  <li>• Generates complementary part</li>
                  <li>• Uses appropriate instruments</li>
                </ul>
              </div>
            </div>

            {/* Footer Buttons */}
            <div className="flex gap-2 mt-4 pt-4 border-t border-gray-700">
              <button
                onClick={handleClose}
                disabled={isGenerating}
                className="flex-1 px-3 py-2 bg-gray-800 hover:bg-gray-700 text-white text-sm rounded-md
                           font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Cancel
              </button>
              <button
                onClick={handleGenerate}
                disabled={isGenerating || !canGenerate}
                className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700
                           text-white text-sm rounded-md font-medium transition-colors flex items-center justify-center gap-2
                           disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <>
                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}