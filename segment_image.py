import tkinter as tk
from tkinter import filedialog
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import functional as F
import argparse
import sys
import json
import os
import glob



def load_model():
    """
    Carrega o modelo DeepLabv3+ pré-treinado.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    ).to(device).eval()
    return model, device

def preprocess_image(image, device):
    """
    Pre-processa a imagem para o modelo.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = torchvision.transforms.functional.to_tensor(image_rgb).to(device)
    input_image = input_image.unsqueeze(0)
    return image_rgb, input_image

def save_and_show_image(image_rgb, segmented_image, binary_mask, contour_mask, output_path='segmented_people.jpg'):
    """
    Salva e exibe a imagem segmentada junto com a máscara binária e o contorno da máscara.
    """
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    fig, axs = plt.subplots(1, 4, figsize=(20, 10))

    axs[0].imshow(image_rgb)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(segmented_image)
    axs[1].set_title('Segmented Image')
    axs[1].axis('off')

    axs[2].imshow(binary_mask, cmap='gray')
    axs[2].set_title('Binary Mask')
    axs[2].axis('off')

    axs[3].imshow(contour_mask, cmap='gray')
    axs[3].set_title('Contour Mask')
    axs[3].axis('off')

    plt.tight_layout()
    plt.show()

def process_image(image_path):
    model, device = load_model()
    image = cv2.imread(image_path)
    image_rgb, input_image = preprocess_image(image, device)
    output_predictions = segment_image(model, input_image)
    segmented_image, binary_mask, contour_mask, contours = apply_mask(image_rgb, output_predictions)
    save_and_show_image(image_rgb, segmented_image, binary_mask, contour_mask)
    return contours

def process_video(video_path):
    model, device = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        image_rgb, input_image = preprocess_image(frame, device)
        output_predictions = segment_image(model, input_image)
        segmented_image, binary_mask, contour_mask, contours = apply_mask(image_rgb, output_predictions)

        combined_frame = np.hstack((cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), segmented_image, cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2BGR)))

        progress_text = display_progress(frame_idx, total_frames, "Segmented Video")
        cv2.putText(combined_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Segmented Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def segment_image(model, input_image):
    """
    Realiza a segmentação da imagem usando o modelo.
    """
    with torch.no_grad():
        output = model(input_image)['out'][0]
    output_predictions = output.argmax(0)
    return output_predictions

def apply_mask(image_rgb, output_predictions):
    """
    Aplica a máscara de segmentação à imagem original e gera a máscara binária.
    """
    person_mask = output_predictions == 15
    person_mask = person_mask.byte().cpu().numpy()

    segmented_image = image_rgb.copy()
    segmented_image[~person_mask.astype(bool)] = 0

    binary_mask = (person_mask * 255).astype(np.uint8)

    # Redimensionar a máscara binária para 256x256
    binary_mask_resized = cv2.resize(binary_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

    # Calcular os contornos a partir da máscara redimensionada
    contours, _ = cv2.findContours(binary_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask_resized = np.zeros_like(binary_mask_resized)
    cv2.drawContours(contour_mask_resized, contours, -1, (255), thickness=1)

    # Redimensionar a imagem segmentada para 256x256
    segmented_image_resized = cv2.resize(segmented_image, (256, 256), interpolation=cv2.INTER_LINEAR)

    return segmented_image_resized, binary_mask_resized, contour_mask_resized, contours



def crop_img(img, init_point, end_point):
    return img[init_point[0]:end_point[0], init_point[1]:end_point[1], :]

def display_progress(frame_idx, total_frames, window_name):
    """
    Display the progress on the frame and terminal.
    """
    progress_text = f"Frame {frame_idx} de {total_frames}"
    print(progress_text, end='\r')
    return progress_text

def process_supervideo(supervideo_path, output_json_path):
    """
    Processa um supervídeo e salva os contornos em um arquivo JSON.
    """
    model, device = load_model()
    cap = cv2.VideoCapture(supervideo_path)

    if not cap.isOpened():
        print("Error: Could not open supervideo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dictionary to store contours for each subimage
    contours_dict = {os.path.basename(supervideo_path): {"subimage_1": [], "subimage_2": [], "subimage_3": [], "subimage_4": []}}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        if frame_idx ==10:
            cap.release()
            cv2.destroyAllWindows()

        imgs = [
            crop_img(frame, (0, 0), (720, 1280)),
            crop_img(frame, (0, 1320), (720, 2600)),
            crop_img(frame, (780, 0), (1500, 1280)),
            crop_img(frame, (780, 1320), (1500, 2600))
        ]

        combined_frame = np.zeros((512, 512, 3), dtype=np.uint8)  # Placeholder for combined frame

        for i, img in enumerate(imgs):
            image_rgb, input_image = preprocess_image(img, device)
            output_predictions = segment_image(model, input_image)
            segmented_image, binary_mask, contour_mask, contours = apply_mask(image_rgb, output_predictions)

            # Store contours in the dictionary
            subimage_key = f"subimage_{i+1}"
            contours_dict[os.path.basename(supervideo_path)][subimage_key].append([c.tolist() for c in contours])

            y_offset = 256 * (i // 2)
            x_offset = 256 * (i % 2)

            combined_frame[y_offset:y_offset+segmented_image.shape[0], x_offset:x_offset+segmented_image.shape[1]] = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

            # Overlay contour mask
            combined_frame[y_offset:y_offset+segmented_image.shape[0], x_offset:x_offset+segmented_image.shape[1]][contour_mask > 0] = [0, 255, 0]

        progress_text = display_progress(frame_idx, total_frames, "Segmented Video")
        cv2.putText(combined_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Segmented Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save contours to a JSON file
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(contours_dict, f)

def process_supervideos_in_directory(directory_path):
    """
    Processa todos os supervídeos em um diretório e suas subpastas.
    """
    supervideo_paths = glob.glob(os.path.join(directory_path, '**', '*.mkv'), recursive=True)

    for supervideo_path in supervideo_paths:
        # Parse the path to create the JSON output path
        path_parts = supervideo_path.split(os.sep)
        subject_id = path_parts[-3]  # subjectX
        activity_id = path_parts[-2]  # activity1 or activity2
        routine_id = path_parts[-1].split('.')[0]  # routine0X

        json_filename = f"{subject_id}_{activity_id}_{routine_id}.json"
        output_json_path = os.path.join('out', json_filename)

        process_supervideo(supervideo_path, output_json_path)

def main(image_path=None, video_path=None, supervideo_path=None, supervideo_superpath=None):
    if ((image_path is None and video_path is None and supervideo_path is None and supervideo_superpath is None) or
        (image_path is not None and (video_path is not None or supervideo_path is not None or supervideo_superpath is not None)) or
        (video_path is not None and (supervideo_path is not None or supervideo_superpath is not None)) or
        (supervideo_path is not None and supervideo_superpath is not None)):
        print("Error: You must specify either an image_path, a video_path, a supervideo_path, or a supervideo_superpath, but not multiple.")
        sys.exit()

    if image_path:
        process_image(image_path)
    elif video_path:
        process_video(video_path)
    elif supervideo_path:
        output_json_path = os.path.join('out', os.path.basename(supervideo_path) + '.json')
        process_supervideo(supervideo_path, output_json_path)
    elif supervideo_superpath:
        process_supervideos_in_directory(supervideo_superpath)

def open_file_dialog(file_type):
    root = tk.Tk()
    root.withdraw()
    if file_type == "image":
        return filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    elif file_type == "video":
        return filedialog.askopenfilename(title="Select a Video", filetypes=[("Video files", "*.mp4;*.mkv;*.avi")])
    elif file_type == "supervideo":
        return filedialog.askopenfilename(title="Select a Supervideo", filetypes=[("Supervideo files", "*.mkv")])
    elif file_type == "supervideo_path":
        return filedialog.askdirectory(title="Select Directory Containing Supervideos")
    else:
        return None

def run_gui():
    def on_select(event):
        file_type = combo.get()
        path = open_file_dialog(file_type)
        if path:
            root.quit()
            root.destroy()
            if file_type == "image":
                main(image_path=path)
            elif file_type == "video":
                main(video_path=path)
            elif file_type == "supervideo":
                main(supervideo_path=path)
            elif file_type == "supervideo_path":
                main(supervideo_superpath=path)

    root = tk.Tk()
    root.title("Select Input Type")

    label = tk.Label(root, text="Select input type:")
    label.pack(pady=10)

    combo = tk.StringVar()
    options = ["image", "video", "supervideo", "supervideo_path"]
    select = tk.OptionMenu(root, combo, *options, command=on_select)
    select.pack(pady=10)

    root.mainloop()

if __name__ == '__main__':
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"), "ta na mao")

    parser = argparse.ArgumentParser(description="Segmentação de pessoas usando DeepLabv3+")
    parser.add_argument('--image_path', type=str, help="Caminho para a imagem de entrada")
    parser.add_argument('--video_path', type=str, help="Caminho para o vídeo de entrada")
    parser.add_argument('--supervideo_path', type=str, help="Caminho para o supervídeo de entrada")
    parser.add_argument('--supervideo_superpath', type=str, help="Caminho para o diretório contendo supervídeos")
    args = parser.parse_args()

    if not any([args.image_path, args.video_path, args.supervideo_path, args.supervideo_superpath]):
        run_gui()
    else:
        main(args.image_path, args.video_path, args.supervideo_path, args.supervideo_superpath)