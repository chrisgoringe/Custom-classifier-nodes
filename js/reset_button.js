import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function resetButtonPressed() {
    const body = new FormData();
    api.fetchApi("/image_classify_reset", { method: "POST", body, });
}

app.registerExtension({
    name: "cg.image_classify.reset_button",
    async nodeCreated(node) {
        if (node?.comfyClass === "Running Average") {
            const reset_button_widget = node.addWidget("button", "reset", "reset", resetButtonPressed);
        }
    }
})