"""Draw the verified current locations of the former Pytorch/ learning files.

Run with Python + Pillow. No experiment modules are imported or executed.
Only the PNG output is written. File names retain their original spelling.
Selection: seven topic folders, representative scripts, and two artifact groups.
The empty legacy folder is explicitly disclosed, not given invented children.
"""

from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SCALE = 2
W, H = 1920, 1080
NAVY = "#17365D"
BLUE = "#2F6192"
TEXT = "#263744"
MUTED = "#64758A"
LINE = "#B9C9D9"
PALE = "#EDF3F8"

STAGES = [
    {
        "path": "01_pytorch_basics",
        "label": "Stage 1  /  Tensors, gradients and network building blocks",
        "x": 72,
        "topics": [
            ("01_tensors_and_devices", "Tensor creation, shape, operations and device",
             ["test_torch.py", "tensor_shape.py"], []),
            ("02_autograd_and_optimization", "Autograd, manual updates and SGD",
             ["auto_grad1.py", "update_w.py", "opitimizer.py"], []),
            ("03_linear_layers", "Weights, biases and linear-layer training",
             ["weight_bias.py", "layer_hand.py", "nn_test"], []),
        ],
    },
    {
        "path": "02_deep_learning",
        "label": "Stage 2  /  MLP, supervised learning and image models",
        "x": 1000,
        "topics": [
            ("01_mlp_and_data_loading", "MLP regression, Dataset and DataLoader",
             ["mlp_test.py", "mlp_addDataset.py"], ["artifacts/figures/"]),
            ("02_classification", "Multiclass prediction, cross-entropy and accuracy",
             ["classfication_test.py"], []),
            ("03_mnist_mlp", "MNIST training, model saving and reloading",
             ["mnist_test.py", "mnist_test_load.py"], ["artifacts/checkpoints/from_pytorch_dir/"]),
            ("04_cnn", "Convolution, pooling, feature maps and MNIST CNN",
             ["cnn_test1.py", "cnn_mnist_train.py"], []),
        ],
    },
]


def validate_sources():
    legacy = ROOT / "Pytorch"
    if not legacy.is_dir():
        raise FileNotFoundError(legacy)
    if any(legacy.iterdir()):
        raise RuntimeError("Pytorch/ is no longer empty. Review the source selection before redrawing.")
    migration = (ROOT / "docs/migration/README.md").read_text(encoding="utf-8")
    verified = []
    for stage in STAGES:
        for topic, _, files, artifacts in stage["topics"]:
            folder = ROOT / stage["path"] / topic
            for filename in files:
                path = folder / "experiments" / filename
                if not path.is_file():
                    raise FileNotFoundError(path)
                if f"Pytorch/{filename}" not in migration or path.relative_to(ROOT).as_posix() not in migration:
                    raise RuntimeError(f"Missing migration provenance: {path}")
                verified.append(path)
            for artifact in artifacts:
                if not (folder / artifact).is_dir():
                    raise FileNotFoundError(folder / artifact)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("pytorch_learning_folder_structure.png"))
    args = parser.parse_args()
    verified = validate_sources()
    image = Image.new("RGB", (W * SCALE, H * SCALE), "white")
    draw = ImageDraw.Draw(image)
    fonts = {
        "regular": Path("C:/Windows/Fonts/segoeui.ttf"),
        "bold": Path("C:/Windows/Fonts/seguisb.ttf"),
        "mono": Path("C:/Windows/Fonts/consola.ttf"),
    }

    def font(size, kind="regular"):
        return ImageFont.truetype(str(fonts[kind]), round(size * SCALE))

    def text(x, y, value, size=22, color=TEXT, kind="regular", max_width=None):
        ft = font(size, kind)
        if max_width is not None and draw.textlength(value, font=ft) > max_width * SCALE:
            raise ValueError(f"Text exceeds its column: {value}")
        draw.text((x * SCALE, y * SCALE), value, font=ft, fill=color, anchor="lt")

    def line(points, color=LINE, width=2):
        draw.line([(int(x * SCALE), int(y * SCALE)) for x, y in points], fill=color, width=width * SCALE)

    def rect(x, y, w, h, color):
        draw.rectangle((x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE), fill=color)

    text(72, 45, "PyTorch Learning Folder Structure", 46, NAVY, "bold")
    text(72, 108, "Organized by learning stages and topics", 25, MUTED)

    rect(72, 172, 1776, 85, NAVY)
    text(96, 195, "pytorch-rl-study/", 31, "white", "mono")
    text(1020, 195, "Pytorch/ is now empty after the documented migration.", 23, "white", max_width=805)
    text(1020, 226, "Branches below show the files at their actual current paths.", 20, "#DCE7F1", max_width=805)

    # Directory connections: workspace -> stages -> topic folders.
    line([(960, 257), (960, 281), (96, 281), (96, 309)])
    line([(960, 281), (1024, 281), (1024, 309)])

    for stage in STAGES:
        x = stage["x"]
        rect(x, 309, 848, 70, PALE)
        text(x + 22, 320, stage["path"] + "/", 28, NAVY, "mono", 802)
        text(x + 22, 353, stage["label"], 20, BLUE, max_width=802)
        ys = [411, 587, 763] if len(stage["topics"]) == 3 else [397, 563, 695, 861]
        trunk = x + 24
        line([(trunk, 379), (trunk, ys[-1] + 15)])
        for y, (topic, purpose, files, artifacts) in zip(ys, stage["topics"]):
            tx = x + 54
            line([(trunk, y + 15), (tx - 13, y + 15)])
            text(tx, y, topic + "/", 25, NAVY, "mono", 780)
            text(tx, y + 34, purpose, 22, BLUE, max_width=780)
            text(tx + 18, y + 68, "experiments/", 21, MUTED, "mono", 760)
            text(tx + 42, y + 97, "  ·  ".join(files), 22, TEXT, "mono", 738)
            if artifacts:
                text(tx + 18, y + 125, artifacts[0], 20, MUTED, "mono", 760)

    line([(72, 1004), (1848, 1004)], "#DCE5EE", 1)
    text(72, 1021, "Selected topic folders and representative files. Original spellings are retained.", 21, MUTED)
    text(72, 1052, "Source: current workspace + docs/migration/README.md. Cache, empty placeholders and unrelated RL folders omitted.", 19, MUTED)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    print(f"Verified {len(verified)} representative scripts across 7 topic folders.")
    print(f"PNG: {args.output.resolve()} ({image.width} x {image.height}, 300 DPI)")


if __name__ == "__main__":
    main()
