import test from 'node:test';
import assert from 'node:assert/strict';
import { extractSseMessages } from '../src/sse.js';

test('extracts complete SSE frames and retains a partial frame', () => {
  const parsed = extractSseMessages('data: {"event":"one"}\n\ndata: partial');
  assert.deepEqual(parsed.messages, ['{"event":"one"}']);
  assert.equal(parsed.rest, 'data: partial');
});

test('joins multiline SSE data fields', () => {
  const parsed = extractSseMessages('data: first\ndata: second\n\n');
  assert.deepEqual(parsed.messages, ['first\nsecond']);
});
