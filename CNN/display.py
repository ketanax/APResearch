import os
import random
import tkinter as tk
from PIL import Image, ImageTk

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import ResNet50

# ---------- CONFIG ----------
DATASET_ROOT = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\imagenet2012\imagenet"
WEIGHTS_PATH = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\CNN\trained_models\CNN2.pth"
CLASS_NAMES_PATH = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\CNN\imagenet_classes.txt"

WINDOW_TITLE = "Random ImageNet Test Viewer"
WINDOW_SIZE = "1000x700"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ----------------------------

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD MODEL ----------
model = ResNet50(num_classes=1000)
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]

with open(CLASS_NAMES_PATH, "r") as f:
    IMAGENET_CLASSES = [line.strip() for line in f]

assert len(IMAGENET_CLASSES) == 1000, "Expected 1000 ImageNet class names"

model.load_state_dict(state_dict)
model.to(device)
model.eval()

print("Model loaded")

# ---------- TRANSFORMS ----------
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- DATASET ----------
test_dir = os.path.join(DATASET_ROOT, "test")
class_folders = [
    d for d in os.listdir(test_dir)
    if os.path.isdir(os.path.join(test_dir, d))
]

# ---------- GUI ----------
root = tk.Tk()
root.title(WINDOW_TITLE)
root.geometry(WINDOW_SIZE)

# ----- Layout Frames -----
top_frame = tk.Frame(root)
top_frame.pack(fill="x")

left_top = tk.Frame(top_frame)
left_top.pack(side="left", anchor="nw", padx=10, pady=5)

right_top = tk.Frame(top_frame)
right_top.pack(side="right", anchor="ne", padx=10, pady=5)

center_frame = tk.Frame(root)
center_frame.pack(expand=True)

bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

# ----- Title (Top-Left) -----
title_label = tk.Label(
    left_top,
    text="ap research CNN demonstration",
    font=("Arial", 16, "bold")
)
title_label.pack(anchor="w")

# ----- Image -----
image_label = tk.Label(center_frame)
image_label.pack(expand=True)

current_photo = None

# ----- Matplotlib Figure (Top-Right) -----
fig, ax = plt.subplots(figsize=(4.6, 3.4))
ax.set_title("Top-5 Predictions")

canvas = FigureCanvasTkAgg(fig, master=right_top)
canvas.get_tk_widget().pack(padx=5, pady=5)

# GT label under graph
gt_label = tk.Label(
    right_top,
    text="",
    font=("Arial", 11, "bold"),
    wraplength=380,
    justify="left"
)
gt_label.pack(anchor="w", pady=(2, 2))

top1_label = tk.Label(
    right_top,
    text="",
    font=("Arial", 10)
)
top1_label.pack(anchor="w")

top5_label = tk.Label(
    right_top,
    text="",
    font=("Arial", 10)
)
top5_label.pack(anchor="w", pady=(0, 5))

# ---------- LOAD + PROCESS FUNCTION ----------
def load_random_image():
    global current_photo

    selected_class = random.choice(class_folders)
    class_path = os.path.join(test_dir, selected_class)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not images:
        return

    selected_image = random.choice(images)
    image_path = os.path.join(class_path, selected_image)

    pil_image = Image.open(image_path).convert("RGB")
    input_tensor = transformations(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    pred_top1 = top5_idx[0][0].item()
    top5_preds = top5_idx[0].tolist()

    gt_index = int(selected_class)
    gt_name = IMAGENET_CLASSES[gt_index]
    pred_name = IMAGENET_CLASSES[pred_top1]

    top1_correct = gt_index == pred_top1
    top5_correct = gt_index in top5_preds

    # ---- Update labels ----
    gt_label.config(
        text=f"Randomly Selected Class:\n"
             f"{gt_index} — {gt_name}"
    )

    top1_label.config(
        text=f"Top-1 Correct: {'YES' if top1_correct else 'NO'}"
    )

    top5_label.config(
        text=f"Top-5 Correct: {'YES' if top5_correct else 'NO'}"
    )
    # ---- Console Output ----
    print("\n==============================")
    print(f"GT Class: {selected_class}")
    print(f"Image: {selected_image}")
    for i in range(5):
        print(
            f"{i+1}. Class {top5_idx[0][i].item()} "
            f"({top5_prob[0][i].item():.4f})"
        )

    # ---- Update Image ----
    display_img = pil_image.copy()
    display_img.thumbnail((700, 500), Image.LANCZOS)
    current_photo = ImageTk.PhotoImage(display_img)
    image_label.config(image=current_photo)

    # ---- Update Graph ----
    ax.clear()

    classes = [
        f"{c.item()} ({IMAGENET_CLASSES[c.item()]})"
        for c in top5_idx[0]
    ]
    confidences = top5_prob[0].cpu().numpy()

    ax.barh(classes[::-1], confidences[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Class")
    ax.set_title("Top-5 Predictions")

    fig.tight_layout()
    canvas.draw()

    gt_label.config(text=f"Randomly Selected Class: {selected_class}")

    root.title(
        f"{WINDOW_TITLE} | GT: {selected_class} | Pred: {top5_idx[0][0].item()}"
    )

# ---------- NEXT BUTTON ----------
next_button = tk.Button(
    bottom_frame,
    text="Next Image",
    font=("Arial", 12),
    command=load_random_image
)
next_button.pack()

# ---------- INITIAL LOAD ----------
load_random_image()

root.mainloop()
