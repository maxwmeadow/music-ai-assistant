export const api = (path: string, init?: RequestInit) =>
    fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, init);
