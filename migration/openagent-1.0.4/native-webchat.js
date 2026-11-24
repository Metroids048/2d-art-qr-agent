import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

const studio = "/oasn/art-qr-studio";
const generating = "正在制作二维码预览，做好后会发到这里。";
const received = "已收到图片和网址，正在读取制作要求。";
const harnessRuntime = "/opt/oasn/platform/provider-runtime/lib/node_modules/openclaw/dist/plugin-sdk/agent-harness-runtime.js";
// OpenClaw loads lifecycle hooks and tool factories through separate registries.
const turnStorage = Symbol.for("artqr-chat.native-webchat-turns.v4");
const { calls, replies, waiting, fallbackChecks, webchatDispatchers } = globalThis[turnStorage] ??= {
  calls: new Map(), replies: new Map(), waiting: new Map(), fallbackChecks: new Map(),
  webchatDispatchers: new Map()
};
const dashboardSession = /^agent:main:dashboard:[0-9a-f-]{36}$/;
const imageMime = /^image\/(?:png|jpeg|webp)$/;
const parameters = {
  type: "object",
  additionalProperties: false,
  properties: {
    title: { type: "string", maxLength: 60 },
    brief: { type: "string", maxLength: 450 },
    caption: { type: "string", maxLength: 85 },
    layout: { type: "string", enum: ["auto", "cup", "round", "sticker", "bar", "arch", "label", "shelf", "beads"] },
    width: { type: "number", minimum: 45, maximum: 260 },
    height: { type: "number", minimum: 45, maximum: 260 },
    fusion_style: { type: "string", enum: ["auto", "print", "glass", "weave"] },
    qr_percent: { type: "number", minimum: 52, maximum: 80 },
    art_px: { type: "integer", minimum: 120, maximum: 2400 }
  }
};

function rememberTurn(map, key, value) {
  map.set(key, value);
  setTimeout(() => {
    if (map.get(key) === value) map.delete(key);
  }, 300000).unref();
}

async function sendProgress(api, runId, signal, message, stage) {
  try {
    const { emitAgentEvent } = await import(harnessRuntime);
    if (signal?.aborted || waiting.has(runId)) return;
    rememberTurn(waiting, runId, true);
    emitAgentEvent({ runId, stream: "assistant", data: { text: message, delta: message } });
    api.logger.info(`artqr-native-reply: sent wait message (${stage})`);
  } catch {
    api.logger.warn("artqr-native-reply: wait message could not be sent");
  }
}

function runSession(context, runId, design, { signal, onPreviewStarted, onInputRetry, replyOnly = false } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(studio + "/.venv/bin/python", [studio + "/native_turn.py"], {
      cwd: studio, stdio: ["pipe", "pipe", "pipe"], signal, timeout: replyOnly ? 8000 : 240000
    });
    let output = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (data) => { output += data; });
    const stderr = createInterface({ input: child.stderr });
    stderr.on("line", (line) => {
      if (line === "ARTQR_PREVIEW_STARTED") onPreviewStarted?.();
      const retry = /^artqr_input_retry: (session_unbound|gateway_run_unbound|invocation_missing|developer_message_unbound|invocation_schema|invocation_run_mismatch|invocation_input_invalid)$/.exec(line);
      if (retry) onInputRetry?.(retry[1]);
    });
    child.stdin.on("error", reject);
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error("Art QR worker did not report a result"));
        return;
      }
      try {
        resolve(JSON.parse(output));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(JSON.stringify({
      context: { sessionKey: context.sessionKey, sessionId: context.sessionId, runId },
      design, replyOnly
    }));
  });
}

function toolResult(reply) {
  const lines = [reply.message, ...reply.media.map((path) => "MEDIA:" + path)];
  return { content: [{ type: "text", text: lines.join("\n") }], details: { status: reply.status } };
}

export default {
  id: "artqr-chat",
  register(api) {
    const nativeReplies = api.pluginConfig?.nativeWebchatReplies === true;
    api.on("before_tool_call", (event, context) => {
      if (event.toolName !== "artqr_webchat" || !context.sessionKey ||
          !context.sessionId || !event.runId || !event.toolCallId) return;
      rememberTurn(calls, event.toolCallId, {
        runId: event.runId, sessionKey: context.sessionKey, sessionId: context.sessionId
      });
      api.logger.info("artqr-native-reply: bound tool call");
    });

    if (nativeReplies) {
      api.on("gateway_start", () => {
        api.logger.info("artqr-native-reply: current-turn delivery loaded");
        void import(harnessRuntime).catch(() => api.logger.warn("artqr-native-reply: wait channel unavailable"));
      });

      api.on("reply_dispatch", (event, context) => {
        const { ctx, runId, sessionKey } = event;
        if (ctx.Provider !== "webchat" || ctx.Surface !== "webchat" ||
            ctx.ExplicitDeliverRoute || ctx.CommandSource || !dashboardSession.test(sessionKey) ||
            ctx.SessionKey !== sessionKey || !runId || ctx.MessageSid !== runId ||
            event.suppressUserDelivery || context.abortSignal?.aborted) return;
        rememberTurn(webchatDispatchers, runId, { sessionKey, dispatcher: context.dispatcher });
        const hasImage = ctx.MediaTypes?.some((type) => imageMime.test(type))
          || event.images?.some((image) => imageMime.test(image.mimeType));
        if (hasImage && /https?:\/\/[^\s<>"，。]+/i.test(ctx.RawBody ?? "")) {
          void sendProgress(api, runId, context.abortSignal, received, "intake");
        }
        // The host still runs its ordinary agent and records the original user turn.
      }, { eligibleDispatchKinds: ["agent"] });

      api.on("reply_payload_sending", async (event, context) => {
        if (event.kind !== "block" && event.kind !== "final") return;
        let reply = replies.get(event.runId);
        if (event.payload.isError) return;
        if (!reply && event.kind === "final" && event.runId && event.sessionKey) {
          // Verify current-turn questions, selections, and requests to retrieve
          // existing files. This path cannot start preview generation.
          let check = fallbackChecks.get(event.runId);
          if (!check) {
            check = runSession({ sessionKey: event.sessionKey, sessionId: context?.sessionId },
              event.runId, {}, { replyOnly: true }).catch(() => {
              api.logger.warn("artqr-native-reply: current question or selection could not be checked");
              return null;
            });
            rememberTurn(fallbackChecks, event.runId, check);
          }
          const outcome = await check;
          reply = replies.get(event.runId);
          if (!reply && outcome) {
            reply = { ...outcome, sessionKey: event.sessionKey };
            rememberTurn(replies, event.runId, reply);
          }
        }
        if (!reply || reply.sessionKey !== event.sessionKey) return;
        // WebChat projects media from a block reply as well as the final reply.
        // Keep one authoritative reply when the runtime sends both.
        if (reply.sent) return { cancel: true };
        reply.sent = true;
        api.logger.info(`artqr-native-reply: applied current result (${reply.status}, ${event.kind})`);
        return {
          payload: { ...event.payload, text: reply.message, mediaUrl: undefined, mediaUrls: reply.media }
        };
      });
    }

    api.registerTool((context) => ({
      name: "artqr_webchat",
      label: "制作艺术二维码",
      description: "读取当前聊天的真实图片、网址和确认消息，生成一张预览、按反馈调整或重取成品。只传用户明确要求的设计选项。",
      parameters,
      async execute(toolCallId, design, signal) {
        const call = calls.get(toolCallId);
        const runId = call && call.sessionKey === context.sessionKey && call.sessionId === context.sessionId
          ? call.runId : undefined;
        api.logger.info(`artqr-native-reply: tool call matched ${Boolean(runId)}`);
        if (runId) calls.delete(toolCallId);
        const existing = runId && replies.get(runId);
        if (existing && existing.sessionKey === context.sessionKey &&
            (existing.status === "preview" || existing.status === "delivered")) {
          api.logger.info(`artqr-native-reply: reused current result (${existing.status})`);
          return toolResult(existing);
        }
        const onPreviewStarted = nativeReplies && runId &&
          (() => { void sendProgress(api, runId, signal, generating, "worker"); });
        let result;
        try {
          result = await runSession(context, runId, design, {
            signal, onPreviewStarted, onInputRetry: (reason) => {
              const safeRun = /^(?:run_[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})$/.test(runId)
                ? runId : "unbound";
              api.logger.warn(`artqr-input: ${reason} (run ${safeRun})`);
            }
          });
        } catch (error) {
          if (signal?.aborted) throw error;
          result = { message: "这次没能发出完整结果。你可以回复「重试」，我会先找这单已有的预览。", status: "retry", media: [] };
        }
        if (nativeReplies && runId) {
          const reply = { ...result, sessionKey: context.sessionKey };
          rememberTurn(replies, runId, reply);
          api.logger.info(`artqr-native-reply: stored current result (${result.status})`);
          const route = webchatDispatchers.get(runId);
          if (route?.sessionKey === context.sessionKey && !signal?.aborted &&
              result.media.length > 0 && (result.status === "preview" || result.status === "delivered")) {
            webchatDispatchers.delete(runId);
            // Replies queued during an embedded agent run require gateway
            // notice delivery so that WebChat renders them on this turn.
            const queued = route.dispatcher.sendFinalReply({
              text: result.message, mediaUrls: result.media, isStatusNotice: true
            });
            if (queued) api.logger.info(`artqr-native-reply: queued worker result (${result.status})`);
          }
        }
        return toolResult(result);
      }
    }), { name: "artqr_webchat" });
  }
};
