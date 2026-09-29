export function extractSseMessages(buffer) {
  const frames = buffer.split(/\r?\n\r?\n/);
  const rest = frames.pop() ?? '';
  const messages = frames.flatMap((frame) => {
    const data = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith('data:'))
      .map((line) => line.slice(5).trimStart())
      .join('\n');
    return data ? [data] : [];
  });
  return { messages, rest };
}

export async function streamSse(url, { headers, signal, onMessage }) {
  const response = await fetch(url, { headers, signal });
  if (!response.ok || !response.body) {
    throw new Error(`SSE connection failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = extractSseMessages(buffer);
    buffer = parsed.rest;
    parsed.messages.forEach(onMessage);
  }
}
