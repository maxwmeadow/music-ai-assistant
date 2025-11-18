import { handleError, ErrorContext } from './errorHandler';

export const api = async (path: string, init?: RequestInit): Promise<Response> => {
  const context: ErrorContext = {
    endpoint: path,
    operation: init?.method || 'GET',
  };

  // Create abort controller for timeout
  const controller = new AbortController();
  let timeoutId: NodeJS.Timeout | null = setTimeout(() => controller.abort(), 30000); // 30 second timeout

  try {
    const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }

    // Handle non-OK responses
    if (!response.ok) {
      let errorMessage = `HTTP ${response.status}`;
      try {
        const errorText = await response.text();
        errorMessage = errorText || errorMessage;
      } catch (e) {
        // Ignore parsing errors
      }

      const error = new Error(errorMessage);
      (error as any).status = response.status;
      throw handleError(error, { ...context, statusCode: response.status });
    }

    return response;
  } catch (error: any) {
    if (timeoutId) {
      clearTimeout(timeoutId);
      timeoutId = null;
    }
    
    // Handle AbortError (timeout)
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      throw handleError(
        new Error('Request timeout'),
        { ...context, operation: 'timeout' }
      );
    }

    // Re-throw if already an AppError
    if (error instanceof Error && error.name === 'AppError') {
      throw error;
    }

    // Handle fetch errors
    throw handleError(error, context);
  }
};
