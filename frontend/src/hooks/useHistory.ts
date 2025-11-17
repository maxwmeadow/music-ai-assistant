import { useState, useCallback, useRef, useEffect } from 'react';

interface HistoryEntry<T> {
  state: T;
  timestamp: number;
}

interface HistoryState<T> {
  entries: HistoryEntry<T>[];
  currentIndex: number;
}

interface UseHistoryReturn<T> {
  pushHistory: (state: T) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  currentState: T;
}

/**
 * useHistory hook - manages undo/redo functionality with debouncing
 * @param initialState - The initial state value
 * @param maxHistorySize - Maximum number of history entries to keep (default: 50)
 * @param debounceMs - Debounce delay in milliseconds (default: 500ms)
 * @returns History management functions and state
 */
export function useHistory<T>(
  initialState: T,
  maxHistorySize: number = 50,
  debounceMs: number = 500
): UseHistoryReturn<T> {
  // Single state object to avoid race conditions
  const [historyState, setHistoryState] = useState<HistoryState<T>>({
    entries: [{ state: initialState, timestamp: Date.now() }],
    currentIndex: 0
  });

  // Track if we're currently applying undo/redo to avoid re-pushing
  const isApplyingHistory = useRef(false);

  // Debouncing state
  const pendingState = useRef<T | null>(null);
  const debounceTimer = useRef<NodeJS.Timeout | null>(null);

  // Get current state
  const currentState = historyState.entries[historyState.currentIndex]?.state ?? initialState;

  /**
   * Actually push state to history (internal, after debounce)
   */
  const pushHistoryImmediate = useCallback((state: T) => {
    setHistoryState(prev => {
      // If we're not at the end, discard everything after current position (branching)
      const newEntries = prev.entries.slice(0, prev.currentIndex + 1);

      // Add new entry
      newEntries.push({
        state,
        timestamp: Date.now()
      });

      // Limit history size
      const finalEntries = newEntries.length > maxHistorySize
        ? newEntries.slice(newEntries.length - maxHistorySize)
        : newEntries;

      return {
        entries: finalEntries,
        currentIndex: finalEntries.length - 1
      };
    });
  }, [maxHistorySize]);

  /**
   * Push a new state to history (debounced)
   * If we're not at the end of history, this creates a new branch (discards forward history)
   */
  const pushHistory = useCallback((state: T) => {
    // Don't push if we're applying undo/redo
    if (isApplyingHistory.current) {
      return;
    }

    // Store the pending state
    pendingState.current = state;

    // Clear existing timer
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current);
    }

    // Set new timer
    debounceTimer.current = setTimeout(() => {
      if (pendingState.current !== null) {
        pushHistoryImmediate(pendingState.current);
        pendingState.current = null;
      }
    }, debounceMs);
  }, [debounceMs, pushHistoryImmediate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current);
      }
    };
  }, []);

  /**
   * Undo - go back one step in history
   */
  const undo = useCallback(() => {
    setHistoryState(prev => {
      if (prev.currentIndex > 0) {
        isApplyingHistory.current = true;
        setTimeout(() => {
          isApplyingHistory.current = false;
        }, 0);

        return {
          ...prev,
          currentIndex: prev.currentIndex - 1
        };
      }
      return prev;
    });
  }, []);

  /**
   * Redo - go forward one step in history
   */
  const redo = useCallback(() => {
    setHistoryState(prev => {
      if (prev.currentIndex < prev.entries.length - 1) {
        isApplyingHistory.current = true;
        setTimeout(() => {
          isApplyingHistory.current = false;
        }, 0);

        return {
          ...prev,
          currentIndex: prev.currentIndex + 1
        };
      }
      return prev;
    });
  }, []);

  const canUndo = historyState.currentIndex > 0;
  const canRedo = historyState.currentIndex < historyState.entries.length - 1;

  return {
    pushHistory,
    undo,
    redo,
    canUndo,
    canRedo,
    currentState
  };
}
