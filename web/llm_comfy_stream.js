import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "llm_comfy_multimodal.stream",
	async setup() {
		app.api.addEventListener("llm_comfy_multimodal.stream", (event) => {
			const t = event.detail && event.detail.text;
			if (typeof t === "string" && t.length)
				console.info(
					"[llm_comfy_multimodal stream]",
					t.length > 300 ? t.slice(0, 300) + "…" : t
				);
		});
	},
});
