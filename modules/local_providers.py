"""Local-only providers reusing the original retrieval and perception wrappers.

Quality values are explicit structural/proxy features, not calibrated accuracy.
Use the same extraction convention when fitting the packet verifier. Missing
assets produce invalid packets through PacketProvider; no downloads are started.
"""

from pathlib import Path
from .evidence_contract import EvidenceChannel, EvidencePacket
from .factory import build_knowledge, _local_path
from .providers import PacketProvider


def build_providers(config):
    providers = {}
    knowledge = build_knowledge(config)
    know = config.get("knowledge", {})
    for mode in ("template", "rag"):
        if mode == "rag" and not Path(knowledge.embedding_model_name).is_dir():
            continue

        def acquire(question, image, mode=mode):
            content = knowledge(question, mode)
            return EvidencePacket(
                EvidenceChannel.KNOWLEDGE,
                mode,
                content,
                coverage=float(bool(content)),
                confidence=0.0,
                ambiguity=0.0,
                provenance=(f"local-{mode}:" + str(know.get("revision", "unversioned")),),
                metadata={"quality_convention": "structural-coverage; confidence-unavailable"},
            )

        providers["knowledge:" + mode] = acquire
    from tools.depth_tool import DepthAnythingTool
    from tools.grounding_tool import GroundingDinoTool

    for mode, cls in (("depth", DepthAnythingTool), ("ground", GroundingDinoTool)):
        item = config.get("tools", {}).get("grounding" if mode == "ground" else mode, {})
        local = _local_path(item.get("local_path"))
        if not local or not Path(local).is_dir():
            continue
        tool = cls(local_path=local, device=config.get("device", "cuda"))

        def acquire(question, image, mode=mode, tool=tool, item=item):
            if isinstance(image, list):
                # The tool's input is one explicitly recorded frame, never a hidden loop.
                observation = image[0] if image else None
            else:
                observation = image
            content = tool(observation, question)
            return EvidencePacket(
                EvidenceChannel.TOOL,
                mode,
                content,
                coverage=float(bool(content)),
                confidence=0.0,
                ambiguity=0.0,
                provenance=(f"{mode}:{item.get('revision', 'unversioned')}:frame-0",),
                cost=float(config.get("router", {}).get("tool_costs", {}).get(mode, 1)),
                metadata={
                    "frame_index": 0,
                    "quality_convention": "structural-coverage; confidence-unavailable",
                },
            )

        providers["tool:" + mode] = acquire
    if config.get("tools", {}).get("ocr", {}).get("enabled", False):

        def ocr(question, image):
            import pytesseract
            from PIL import Image

            observation = image[0] if isinstance(image, list) else image
            im = Image.open(observation) if isinstance(observation, (str, Path)) else observation
            data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT)
            rows = [
                (
                    text,
                    float(data["conf"][i]),
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i],
                )
                for i, text in enumerate(data["text"])
                if text.strip() and float(data["conf"][i]) >= 0
            ]
            content = "\n".join(
                f"text={t!r}; confidence={c / 100:.3f}; pixel_box=({x},{y},{w},{h}); frame=0"
                for t, c, x, y, w, h in rows
            )
            return EvidencePacket(
                EvidenceChannel.TOOL,
                "ocr",
                content,
                coverage=float(bool(rows)),
                confidence=sum(c for _, c, *_ in rows) / max(len(rows), 1) / 100,
                provenance=(f"tesseract:{pytesseract.get_tesseract_version()}:frame-0",),
                cost=float(config.get("router", {}).get("tool_costs", {}).get("ocr", 1)),
                metadata={"quality_convention": "OCR confidence proxy, not calibrated"},
            )

        providers["tool:ocr"] = ocr
    return PacketProvider(providers)
