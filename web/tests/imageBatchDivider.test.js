import { readFileSync } from "node:fs";
import vm from "node:vm";
import { describe, expect, it } from "vitest";

const source = readFileSync(new URL("../image_batch_divider.js", import.meta.url), "utf8")
    .replace(/^import .*;\r?\n/gm, "");

const MAX_OUTPUTS = 16;

function setup(divideBy, comfyClass = "ImageBatchDivider") {
    let extension;
    vm.runInNewContext(source, {
        app: { registerExtension: (ext) => { extension = ext; } },
        // Run the delayed initial refresh synchronously so the test is stable.
        setTimeout: (fn) => fn(),
        parseInt,
        Math,
        Array,
        console,
        MutationObserver: class { observe() {} disconnect() {} },
    });
    const node = {
        comfyClass,
        widgets: [{ name: "divide_by", value: divideBy }],
        outputs: [],
        addOutput(name, type) { this.outputs.push({ name, type }); },
        removeOutput(index) { this.outputs.splice(index, 1); },
        setDirty() {},
    };
    return { extension, node };
}

describe("Batch divider sockets", () => {
    it("shows one socket per fixed split", async () => {
        const { extension, node } = setup(3);
        await extension.nodeCreated(node);
        expect(node.outputs.map((output) => output.name)).toEqual(["batch_1", "batch_2", "batch_3"]);
        expect(node.outputs.every((output) => output.type === "IMAGE")).toBe(true);
    });

    it("shows every socket in auto mode so any fleet size can be wired", async () => {
        const { extension, node } = setup(0);
        await extension.nodeCreated(node);
        expect(node.outputs).toHaveLength(MAX_OUTPUTS);
        expect(node.outputs[0].name).toBe("batch_1");
        expect(node.outputs.at(-1).name).toBe(`batch_${MAX_OUTPUTS}`);
    });

    it("caps a manual split at the socket limit", async () => {
        const { extension, node } = setup(24);
        await extension.nodeCreated(node);
        expect(node.outputs).toHaveLength(MAX_OUTPUTS);
    });

    it("uses the audio prefix on the audio divider", async () => {
        const { extension, node } = setup(0, "AudioBatchDivider");
        await extension.nodeCreated(node);
        expect(node.outputs[0]).toMatchObject({ name: "audio_1", type: "AUDIO" });
        expect(node.outputs).toHaveLength(MAX_OUTPUTS);
    });

    it("grows and shrinks sockets when divide_by changes", async () => {
        const { extension, node } = setup(0);
        await extension.nodeCreated(node);
        expect(node.outputs).toHaveLength(MAX_OUTPUTS);

        const widget = node.widgets.find((candidate) => candidate.name === "divide_by");
        widget.value = 2;
        widget.callback(2);
        expect(node.outputs.map((output) => output.name)).toEqual(["batch_1", "batch_2"]);

        widget.value = 0;
        widget.callback(0);
        expect(node.outputs).toHaveLength(MAX_OUTPUTS);
    });
});
