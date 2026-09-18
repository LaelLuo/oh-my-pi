// Contract: an image block whose mimeType declares video/* ships as a
// `video_url` content part on the Chat Completions wire (multimodal extension
// used by OpenAI-compatible providers with native video input, e.g. GLM),
// inlined as a base64 data URL unless the block carries a url. Undecorated
// image blocks keep the byte-for-byte `image_url` forms.
import { describe, expect, it } from "bun:test";
import { streamOpenAICompletions } from "@oh-my-pi/pi-ai/providers/openai-completions";
import type { Context, FetchImpl, Message, Model, UserMessage } from "@oh-my-pi/pi-ai/types";
import { getBundledModel } from "@oh-my-pi/pi-catalog/models";

const MP4_B64 = Buffer.from("not-actually-an-mp4, but bytes are opaque here").toString("base64");
const VIDEO_URL = "https://blobs.example.com/0123456789abcdef0123456789abcdef.mp4";
const PNG_B64 = Buffer.from("not-actually-a-png, but bytes are opaque here").toString("base64");

function completionsModel(): Model<"openai-completions"> {
	return {
		...(getBundledModel("openai", "gpt-4o-mini") as Model<"openai-completions">),
		api: "openai-completions",
	} satisfies Model<"openai-completions">;
}

function captureFetch(model: Model<"openai-completions">, captured: { body?: unknown }): FetchImpl {
	return (async (_input: string | URL | Request, init?: RequestInit) => {
		captured.body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
		const sse =
			`data: ${JSON.stringify({ id: "c", object: "chat.completion.chunk", created: 0, model: model.id, choices: [{ index: 0, delta: { role: "assistant", content: "ok" } }] })}\n\n` +
			`data: ${JSON.stringify({ id: "c", object: "chat.completion.chunk", created: 0, model: model.id, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\n` +
			"data: [DONE]\n\n";
		return new Response(sse, { status: 200, headers: { "content-type": "text/event-stream" } });
	}) as FetchImpl;
}

function userMessage(content: UserMessage["content"]): UserMessage {
	return { role: "user", content, timestamp: 0 };
}

function toolResultMessages(mimeType: string): Message[] {
	return [
		{
			role: "assistant",
			content: [{ type: "toolCall", id: "call_1", name: "screen_recorder", arguments: {} }],
			api: "openai-completions",
			provider: "openai",
			model: "gpt-4o-mini",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "toolUse",
			timestamp: 0,
		},
		{
			role: "toolResult",
			toolCallId: "call_1",
			toolName: "screen_recorder",
			content: [{ type: "image", data: MP4_B64, mimeType }],
			isError: false,
			timestamp: 0,
		},
	];
}

type WirePart = { type: string; text?: string; image_url?: { url: string }; video_url?: { url: string } };

async function wireMessages(model: Model<"openai-completions">, messages: Message[]) {
	const captured: { body?: { messages?: Array<{ role: string; content: unknown }> } } = {};
	const context: Context = { messages };
	await streamOpenAICompletions(model, context, { apiKey: "test", fetch: captureFetch(model, captured) }).result();
	return captured.body?.messages ?? [];
}

describe("video url parts", () => {
	it("ships inline video blocks as video_url data URLs", async () => {
		const messages = [
			userMessage([
				{ type: "text", text: "what happens in this clip?" },
				{ type: "image", data: MP4_B64, mimeType: "video/mp4" },
			]),
		];

		const parts = ((await wireMessages(completionsModel(), messages))[0]?.content as WirePart[]) ?? [];

		expect(parts.filter(part => part.type === "video_url")).toEqual([
			{ type: "video_url", video_url: { url: `data:video/mp4;base64,${MP4_B64}` } },
		]);
		expect(parts.some(part => part.type === "image_url")).toBe(false);
	});

	it("sends the url for video blocks decorated with one", async () => {
		const messages = [userMessage([{ type: "image", data: MP4_B64, mimeType: "video/webm", url: VIDEO_URL }])];

		const parts = ((await wireMessages(completionsModel(), messages))[0]?.content as WirePart[]) ?? [];

		expect(parts.filter(part => part.type === "video_url")).toEqual([
			{ type: "video_url", video_url: { url: VIDEO_URL } },
		]);
	});

	it("keeps image blocks on the image_url wire form", async () => {
		const messages = [
			userMessage([
				{ type: "text", text: "what is in these?" },
				{ type: "image", data: MP4_B64, mimeType: "video/mp4" },
				{ type: "image", data: PNG_B64, mimeType: "image/png" },
			]),
		];

		const parts = ((await wireMessages(completionsModel(), messages))[0]?.content as WirePart[]) ?? [];

		expect(parts.filter(part => part.type === "video_url")).toHaveLength(1);
		expect(parts.filter(part => part.type === "image_url")).toEqual([
			{ type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
		]);
	});

	it("ships tool-result videos as attached video_url parts", async () => {
		const wire = await wireMessages(completionsModel(), toolResultMessages("video/mp4"));
		const attached = wire.find(
			message =>
				message.role === "user" &&
				Array.isArray(message.content) &&
				(message.content as WirePart[]).some(part => part.type === "video_url"),
		);

		const parts = (attached?.content as WirePart[]) ?? [];
		expect(parts).toEqual([
			{ type: "text", text: "Attached media from tool result:" },
			{ type: "video_url", video_url: { url: `data:video/mp4;base64,${MP4_B64}` } },
		]);
	});

	it("keeps image-only tool results byte-for-byte", async () => {
		const wire = await wireMessages(
			completionsModel(),
			toolResultMessages("image/png").map(message =>
				message.role === "toolResult"
					? { ...message, content: [{ type: "image" as const, data: PNG_B64, mimeType: "image/png" }] }
					: message,
			),
		);
		const attached = wire.find(
			message =>
				message.role === "user" &&
				Array.isArray(message.content) &&
				(message.content as WirePart[]).some(part => part.type === "image_url"),
		);

		const parts = (attached?.content as WirePart[]) ?? [];
		expect(parts).toEqual([
			{ type: "text", text: "Attached image(s) from tool result:" },
			{ type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
		]);
	});

	it("omits video blocks for text-only models", async () => {
		const model = { ...completionsModel(), input: ["text"] as Model["input"] };
		const messages = [
			userMessage([
				{ type: "text", text: "what happens in this clip?" },
				{ type: "image", data: MP4_B64, mimeType: "video/mp4" },
			]),
		];

		const wire = await wireMessages(model, messages);
		const parts = (wire[0]?.content as WirePart[]) ?? [];

		expect(parts.some(part => part.type === "video_url")).toBe(false);
		expect(parts.some(part => part.type === "image_url")).toBe(false);
	});
});
