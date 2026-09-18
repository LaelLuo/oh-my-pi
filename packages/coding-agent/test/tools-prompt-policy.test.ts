// Contract: `toolsPromptPolicy: "frozen"` keeps the tool-inventory section of
// the system prompt byte-stable after its first assembly — later
// tool-signature drift follows the frozen path (roster notice, no rebuild) —
// while the default (`"auto"`) and an explicit forcePromptRefresh keep today's
// rebuild behavior.
import { afterEach, describe, expect, it, type Mock, vi } from "bun:test";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import type { Model } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { buildModel } from "@oh-my-pi/pi-catalog/build";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { AgentSessionConfig } from "@oh-my-pi/pi-coding-agent/session/agent-session-types";

function createModel(): Model<"openai-responses"> {
	return buildModel({
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	});
}

function createTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} tool`,
		parameters: type({ value: "string" }),
		strict: true,
		async execute() {
			return { content: [{ type: "text", text: `${name} executed` }] };
		},
	};
}

interface TestSession {
	session: AgentSession;
	rebuild: Mock<(toolNames: string[]) => Promise<string>>;
}

function newSession(configExtra?: Partial<Pick<AgentSessionConfig, "toolsPromptPolicy">>): TestSession {
	const read = createTool("read");
	const bash = createTool("bash");
	const toolRegistry = new Map<string, AgentTool>([
		[read.name, read],
		[bash.name, bash],
	]);
	const mock = createMockModel({ responses: [{ content: ["ok"] }, { content: ["ok"] }] });
	const rebuilder = {
		async rebuildSystemPrompt(toolNames: string[]): Promise<string> {
			return `tools:${toolNames.join(",")}`;
		},
	};
	const rebuild = vi.spyOn(rebuilder, "rebuildSystemPrompt");
	const agent = new Agent({
		getApiKey: () => "test-key",
		initialState: { model: createModel(), systemPrompt: ["initial"], tools: [read], messages: [] },
		convertToLlm,
		streamFn: (requestModel, context, streamOptions) => {
			return mock.stream(requestModel, context, streamOptions);
		},
	});
	const session = new AgentSession({
		agent,
		sessionManager: SessionManager.inMemory(),
		settings: Settings.isolated({ "compaction.enabled": false }),
		modelRegistry: { getApiKey: async () => "test-key" } as never,
		toolRegistry,
		builtInToolNames: ["read", "bash"],
		rebuildSystemPrompt: async toolNames => ({
			systemPrompt: [await rebuilder.rebuildSystemPrompt(toolNames)],
		}),
		toolsPromptPolicy: configExtra?.toolsPromptPolicy,
	});
	return { session, rebuild };
}

describe("tools prompt policy", () => {
	const sessions: AgentSession[] = [];

	afterEach(async () => {
		for (const session of sessions.splice(0)) await session.dispose();
		vi.restoreAllMocks();
	});

	it("defaults to rebuilding the prompt on roster drift (auto)", async () => {
		const harness = newSession();
		sessions.push(harness.session);
		await harness.session.setActiveToolPresentation(["read"], []);
		await harness.session.prompt("first");
		const rebuildsBeforeRosterChange = harness.rebuild.mock.calls.length;

		await harness.session.setActiveToolPresentation(["read", "bash"], []);

		expect(harness.rebuild.mock.calls.length).toBeGreaterThan(rebuildsBeforeRosterChange);
		expect(harness.session.agent.state.systemPrompt).toEqual(["tools:read,bash"]);
	});

	it("frozen rebuilds on the first application", async () => {
		const harness = newSession({ toolsPromptPolicy: "frozen" });
		sessions.push(harness.session);

		await harness.session.setActiveToolPresentation(["read"], []);

		expect(harness.session.agent.state.systemPrompt).toEqual(["tools:read"]);
	});

	it("frozen keeps the prompt stable across roster drift after a turn", async () => {
		const harness = newSession({ toolsPromptPolicy: "frozen" });
		sessions.push(harness.session);
		await harness.session.setActiveToolPresentation(["read"], []);
		await harness.session.prompt("first");
		const rebuildsBeforeRosterChange = harness.rebuild.mock.calls.length;
		const promptBeforeRosterChange = [...harness.session.agent.state.systemPrompt];

		await harness.session.setActiveToolPresentation(["read", "bash"], []);

		expect(harness.rebuild).toHaveBeenCalledTimes(rebuildsBeforeRosterChange);
		expect(harness.session.agent.state.systemPrompt).toEqual(promptBeforeRosterChange);
	});

	it("frozen still rebuilds on explicit forcePromptRefresh", async () => {
		const harness = newSession({ toolsPromptPolicy: "frozen" });
		sessions.push(harness.session);
		await harness.session.setActiveToolPresentation(["read"], []);
		await harness.session.prompt("first");
		const rebuildsBeforeForce = harness.rebuild.mock.calls.length;

		await harness.session.setActiveToolPresentation(["read", "bash"], [], true);

		expect(harness.rebuild.mock.calls.length).toBeGreaterThan(rebuildsBeforeForce);
		expect(harness.session.agent.state.systemPrompt).toEqual(["tools:read,bash"]);
	});

	it("frozen delivers the roster notice with the next user prompt", async () => {
		const harness = newSession({ toolsPromptPolicy: "frozen" });
		sessions.push(harness.session);
		await harness.session.setActiveToolPresentation(["read"], []);
		await harness.session.prompt("first");
		await harness.session.setActiveToolPresentation(["read", "bash"], []);

		await harness.session.prompt("second");

		const notices = harness.session.agent.state.messages.filter(
			message => message.role === "custom" && message.customType === "tool-roster-notice",
		);
		expect(notices).toHaveLength(1);
		expect(notices[0]).toMatchObject({
			details: { added: ["bash"], removed: [] },
			display: false,
			attribution: "agent",
		});
	});

	it("frozen keeps the tool surface itself up to date", async () => {
		const harness = newSession({ toolsPromptPolicy: "frozen" });
		sessions.push(harness.session);
		await harness.session.setActiveToolPresentation(["read"], []);
		await harness.session.prompt("first");

		await harness.session.setActiveToolPresentation(["read", "bash"], []);

		const toolNames = harness.session.agent.state.tools.map(tool => tool.name).sort();
		expect(toolNames).toEqual(["bash", "read"]);
	});
});
