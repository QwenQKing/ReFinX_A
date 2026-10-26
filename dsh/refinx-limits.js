import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { randomUUID } from 'node:crypto';

export const name = 'refinx-limits';
export const inject = ['tools'];

const MAX_STEPS = 10;
const MAX_TOOL_CALLS = 12;
const MAX_CALLS_PER_TOOL = 8;
const MAX_TOTAL_TOKENS = 250000;
const BUDGET_MS = 240000;
const RESERVE_MS = 45000;
const BUDGET_PROMPT =
  'Budget reached. Give your final answer now based on what you have, without calling tools.';

function userMessage(text) {
  const message = structuredClone({
    content: [{ type: 'text', text }],
    source: {
      kind: name,
      form: 'notice',
      summary: text.length <= 120 ? text : text.slice(0, 119) + '…',
    },
    role: 'user',
    id: randomUUID(),
  });
  Object.freeze(message.content[0]);
  Object.freeze(message.content);
  Object.freeze(message.source);
  return Object.freeze(message);
}

export function apply(ctx) {
  const states = new WeakMap();

  function stateFor(session) {
    let state = states.get(session);
    if (state === undefined) {
      state = {
        startedAt: null,
        step: 0,
        totalTokens: 0,
        toolAttempts: 0,
        callsPerTool: new Map(),
        forcedFinal: false,
      };
      states.set(session, state);
    }
    return state;
  }

  function write(record) {
    const log = process.env.REFINX_TRAJECTORY_LOG;
    if (log) fs.appendFileSync(log, JSON.stringify(record) + '\n', 'utf8');
  }

  ctx.on('session/event', (session, event) => {
    const state = stateFor(session);
    switch (event.type) {
      case 'step/start':
        state.step = event.data.step;
        break;
      case 'assistant/message': {
        const usage = event.data.usage ?? {};
        state.totalTokens += (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0)
          + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0);
        write({
          step: state.step, type: 'usage', input: usage.inputTokens ?? null, output: usage.outputTokens ?? null,
          reasoning: usage.reasoningTokens ?? null, cache_read: usage.cacheReadTokens ?? null,
        });
        for (const block of event.data.message.content) {
          if (block.type === 'reasoning' || block.type === 'text') {
            write({
              step: state.step,
              type: block.type === 'reasoning' ? 'thinking' : 'text',
              text: block.text,
            });
          }
        }
        break;
      }
      case 'tool/call': {
        let args = event.data.arguments;
        try {
          const parsed = JSON.parse(args);
          if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
            args = parsed;
          }
        } catch {}
        write({
          step: state.step, type: 'tool_call', call_id: event.data.callId,
          name: event.data.name.replace(/^mcp__refinx__/, ''), arguments: args,
        });
        break;
      }
      case 'tool/result': {
        if (event.surfaceOp !== 'append') break;
        const message = event.data.message;
        const result = message.content
          .filter((block) => block.type === 'text' && typeof block.text === 'string')
          .map((block) => block.text).join('');
        write({
          step: event.data.step, type: 'tool_result', call_id: message.toolCallId,
          result, is_error: message.isError === true || result.startsWith('Error:'),
        });
        break;
      }
    }
  });

  ctx.on('agent/request', async (_payload, next) => {
    const config = await next();
    return { ...config, temperature: 0 };
  });

  ctx.on('agent/pre-step', async ({ agent, step, signal }, next) => {
    const decision = await next();
    if (decision.kind === 'reject' || signal.aborted) return decision;

    const state = stateFor(agent.session);
    if (state.forcedFinal) return { kind: 'reject' };

    const now = performance.now();
    state.startedAt ??= now;
    const elapsed = now - state.startedAt;
    if (elapsed >= BUDGET_MS) return { kind: 'reject' };

    let reason;
    if (step >= MAX_STEPS) {
      reason = 'turns';
    } else if (state.toolAttempts >= MAX_TOOL_CALLS) {
      reason = 'tool calls';
    } else if (BUDGET_MS - elapsed <= RESERVE_MS) {
      reason = 'time';
    } else if (state.totalTokens >= MAX_TOTAL_TOKENS) {
      reason = 'tokens';
    }
    if (reason === undefined) return decision;

    const extra = userMessage(BUDGET_PROMPT);
    state.forcedFinal = true;
    process.stderr.write(`refinx-limits: final answer forced (${reason})\n`);
    return { ...decision, messages: [...decision.messages, extra] };
  });

  ctx.tools.guard((execution) => {
    if (execution.agent === undefined) return undefined;

    const state = stateFor(execution.agent.session);
    if (state.forcedFinal) return BUDGET_PROMPT;

    state.toolAttempts += 1;
    if (state.toolAttempts > MAX_TOOL_CALLS) {
      return 'Error: total tool call limit reached';
    }

    const count = state.callsPerTool.get(execution.name) ?? 0;
    if (count >= MAX_CALLS_PER_TOOL) {
      return 'Error: per-tool call limit reached';
    }

    state.callsPerTool.set(execution.name, count + 1);
    return undefined;
  });
}
