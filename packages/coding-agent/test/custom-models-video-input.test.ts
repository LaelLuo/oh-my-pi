// Contract: models.yml model definitions and overrides may declare the
// `video` input modality; the runtime validation schema retains it instead of
// silently stripping the whole input field (omptype drops unknown union
// members without erroring).
import { expect, test } from "bun:test";
import { ModelsConfigSchema } from "@oh-my-pi/pi-coding-agent/config/models-config-schema";

test("models.yml schema retains the video input modality", () => {
	const checked = ModelsConfigSchema({
		providers: {
			"my-gateway": {
				baseUrl: "https://example.invalid",
				api: "openai-completions",
				models: [{ id: "video-model", input: ["text", "image", "video"] }],
				modelOverrides: { "some-model": { input: ["text", "video"] } },
			},
		},
	});

	expect(checked).toMatchObject({
		providers: {
			"my-gateway": {
				models: [{ id: "video-model", input: ["text", "image", "video"] }],
				modelOverrides: { "some-model": { input: ["text", "video"] } },
			},
		},
	});
});
