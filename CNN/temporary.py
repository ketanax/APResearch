import os
import random
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model import ResNet50  # Assuming this is your custom ResNet50 implementation
import timm

# ---------- CONFIG ----------
DATASET_ROOT = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\imagenet2012\imagenet"
WEIGHTS_PATH = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\CNN\trained_models\CNN2.pth"
CLASS_NAMES_PATH = r"C:\Users\ErogluPC\PycharmProjects\alexnet-pytorch\CNN\imagenet_classes.txt"

WINDOW_TITLE = "Random ImageNet Test Viewer"
WINDOW_SIZE = "1000x700"
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# ----------------------------

# Assume test_dir is the validation set with class folders (wnids)
test_dir = os.path.join(DATASET_ROOT, 'test')  # Adjust if it's train or root directly

# Get class folders
class_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- LOAD RESNET MODEL ----------
model = ResNet50(num_classes=1000)
state_dict = torch.load(WEIGHTS_PATH, map_location=device)
if isinstance(state_dict, dict) and "state_dict" in state_dict:
    state_dict = state_dict["state_dict"]
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("ResNet model loaded")

# ---------- LOAD ViT TEACHER MODEL ----------
teacher = timm.create_model(
    "vit_base_patch16_224_in21k",
    pretrained=True,
    num_classes=1000
)
teacher = teacher.to(device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad = False
print("ViT teacher loaded")

# ---------- CLASS NAMES ----------
with open(CLASS_NAMES_PATH, "r") as f:
    IMAGENET_CLASSES = [line.strip() for line in f]
assert len(IMAGENET_CLASSES) == 1000, "Expected 1000 ImageNet class names"

# ---------- TRANSFORMS ----------
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_top5_predictions(pil_image, model, class_names, device, topk=5):
    """
    Run model on one PIL image → return top-k class names + probabilities
    """
    img_tensor = transformations(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
    probs = F.softmax(logits, dim=1)[0]
    top_probs, top_indices = torch.topk(probs, k=topk)
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = class_names[idx.item()]
        results.append((class_name, prob.item() * 100))  # percentage
    return results

# ---------- GUI SETUP ----------
root = tk.Tk()
root.title(WINDOW_TITLE)
root.geometry(WINDOW_SIZE)

# Image display
label_image = tk.Label(root)
label_image.pack(side=tk.LEFT)

# Text results
text_result = tk.Text(root, height=20, width=50)
text_result.pack(side=tk.RIGHT)

# Button for next image
button_next = tk.Button(root, text="Load Random Image", command=lambda: load_random_image())
button_next.pack()

# Global to keep photo reference
current_photo = None

def load_random_image():
    global current_photo

    selected_class = random.choice(class_folders)
    class_path = os.path.join(test_dir, selected_class)

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not images:
        print(f"No images in {class_path}")
        return

    selected_image = random.choice(images)
    image_path = os.path.join(class_path, selected_image)

    pil_image = Image.open(image_path).convert("RGB")

    # Display image (resize for GUI)
    display_image = pil_image.resize((500, 500), Image.LANCZOS)
    current_photo = ImageTk.PhotoImage(display_image)
    label_image.configure(image=current_photo)
    label_image.image = current_photo  # Keep reference

    # Get predictions for both models
    resnet_preds = get_top5_predictions(pil_image, model, IMAGENET_CLASSES, device)
    vit_preds = get_top5_predictions(pil_image, teacher, IMAGENET_CLASSES, device)

    # Clear text
    text_result.delete("1.0", tk.END)

    # Display info
    text_result.insert(tk.END, f"File: {image_path}\n")
    text_result.insert(tk.END, f"Ground Truth Class (WNID): {selected_class}\n\n")

    text_result.insert(tk.END, "ResNet50 Top-5 Predictions:\n")
    for i, (name, prob) in enumerate(resnet_preds, 1):
        text_result.insert(tk.END, f"{i}. {name:35} {prob:5.2f}%\n")

    text_result.insert(tk.END, "\nViT-B/16 Top-5 Predictions:\n")
    for i, (name, prob) in enumerate(vit_preds, 1):
        text_result.insert(tk.END, f"{i}. {name:35} {prob:5.2f}%\n")

# Optional: If you want to add a matplotlib bar plot for visualizations
# You can create a figure and canvas, but for simplicity, using text here.
# If needed, add code to plot bars for probs in the GUI.

print("Starting GUI...")
root.mainloop()