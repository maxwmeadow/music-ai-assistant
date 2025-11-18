/**
 * Error handling utility for user-friendly error messages
 */

export interface ErrorContext {
  operation?: string;
  endpoint?: string;
  statusCode?: number;
  originalError?: any;
}

export class AppError extends Error {
  constructor(
    public userMessage: string,
    public technicalMessage: string,
    public context?: ErrorContext
  ) {
    super(userMessage);
    this.name = 'AppError';
  }
}

/**
 * Map error to user-friendly message based on error type and context
 */
export function handleError(error: any, context?: ErrorContext): AppError {
  // Log technical details to console
  console.error('[Error Handler]', {
    error,
    context,
    message: error?.message,
    status: error?.status || error?.statusCode,
    name: error?.name,
  });

  const statusCode = error?.status || error?.statusCode || context?.statusCode;
  const errorMessage = error?.message || '';
  const errorName = error?.name || '';

  // Network Errors
  if (
    errorName === 'TypeError' && 
    (errorMessage.includes('fetch') || errorMessage.includes('network') || errorMessage.includes('Failed to fetch'))
  ) {
    return new AppError(
      "Cannot reach backend server. Check that all services are running.",
      `Network error: ${errorMessage}`,
      { ...context, operation: 'network_request' }
    );
  }

  if (statusCode === 0 || errorMessage.includes('NetworkError') || errorMessage.includes('ERR_INTERNET_DISCONNECTED')) {
    return new AppError(
      "Cannot reach backend server. Check your internet connection and that all services are running.",
      `Network disconnected: ${errorMessage}`,
      { ...context, operation: 'network_request' }
    );
  }

  // HTTP Status Code Errors
  if (statusCode === 404) {
    return new AppError(
      "Resource not found. The requested endpoint may not exist.",
      `404 Not Found: ${errorMessage}`,
      { ...context, statusCode: 404 }
    );
  }

  if (statusCode === 500 || statusCode === 502 || statusCode === 503) {
    // Check if it's an audio processing error
    if (context?.endpoint?.includes('hum2melody') || context?.endpoint?.includes('beatbox2drums')) {
      return new AppError(
        "Audio processing failed. Try a shorter recording (max 30 seconds) or check your microphone input.",
        `Server error ${statusCode}: ${errorMessage}`,
        { ...context, statusCode }
      );
    }

    // Generic server error
    return new AppError(
      "Server error occurred. Please try again or contact support if the problem persists.",
      `Server error ${statusCode}: ${errorMessage}`,
      { ...context, statusCode }
    );
  }

  if (statusCode === 400) {
    return new AppError(
      "Invalid request. Please check your input and try again.",
      `Bad request: ${errorMessage}`,
      { ...context, statusCode: 400 }
    );
  }

  if (statusCode === 413) {
    return new AppError(
      "File too large. Please use a shorter recording (max 30 seconds).",
      `Payload too large: ${errorMessage}`,
      { ...context, statusCode: 413 }
    );
  }

  if (statusCode === 408 || errorMessage.includes('timeout')) {
    return new AppError(
      "Request timed out. The server took too long to respond. Please try again.",
      `Timeout: ${errorMessage}`,
      { ...context, statusCode: 408 }
    );
  }

  // Model Inference Errors
  if (
    errorMessage.includes('prediction') ||
    errorMessage.includes('model') ||
    errorMessage.includes('inference') ||
    context?.endpoint?.includes('hum2melody') ||
    context?.endpoint?.includes('beatbox2drums')
  ) {
    if (errorMessage.includes('no notes') || errorMessage.includes('empty')) {
      return new AppError(
        "Could not generate melody. Make sure you're humming clearly and there's enough audio input.",
        `Model inference failed: ${errorMessage}`,
        { ...context, operation: 'model_inference' }
      );
    }
    
    return new AppError(
      "Could not generate melody. Make sure you're humming clearly and try again.",
      `Model inference error: ${errorMessage}`,
      { ...context, operation: 'model_inference' }
    );
  }

  // Playback Errors
  if (
    errorMessage.includes('Transport') ||
    errorMessage.includes('playback') ||
    errorMessage.includes('Tone') ||
    context?.operation === 'playback'
  ) {
    if (errorMessage.includes('sample') || errorMessage.includes('load')) {
      return new AppError(
        "Playback failed. Check that samples are loaded and try again.",
        `Playback error: ${errorMessage}`,
        { ...context, operation: 'playback' }
      );
    }
    
    return new AppError(
      "Playback failed. Check that samples are loaded and the audio context is ready.",
      `Playback error: ${errorMessage}`,
      { ...context, operation: 'playback' }
    );
  }

  // Compilation Errors
  if (context?.operation === 'compilation' || context?.endpoint?.includes('/run')) {
    return new AppError(
      "Compilation failed. Check your code syntax and try again.",
      `Compilation error: ${errorMessage}`,
      { ...context, operation: 'compilation' }
    );
  }

  // Audio Processing Errors (generic)
  if (context?.operation === 'audio_processing' || errorMessage.includes('audio')) {
    return new AppError(
      "Audio processing failed. Try a shorter recording (max 30 seconds).",
      `Audio processing error: ${errorMessage}`,
      { ...context, operation: 'audio_processing' }
    );
  }

  // Generic error fallback
  return new AppError(
    errorMessage || "An unexpected error occurred. Please try again.",
    `Unexpected error: ${errorMessage || errorName || 'Unknown error'}`,
    context
  );
}

/**
 * Wrapper for async operations with error handling
 */
export async function handleAsync<T>(
  operation: () => Promise<T>,
  context?: ErrorContext
): Promise<T> {
  try {
    return await operation();
  } catch (error) {
    throw handleError(error, context);
  }
}

