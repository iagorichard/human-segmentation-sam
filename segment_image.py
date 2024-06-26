import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import functional as F
import argparse
import sys

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
    input_image = F.to_tensor(image_rgb).to(device)
    input_image = input_image.unsqueeze(0)
    return image_rgb, input_image

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
    person_mask = cv2.resize(person_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    segmented_image = image_rgb.copy()
    segmented_image[~person_mask.astype(bool)] = 0

    binary_mask = (person_mask * 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(binary_mask)
    cv2.drawContours(contour_mask, contours, -1, (255), thickness=1)

    return segmented_image, binary_mask, contour_mask

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
    segmented_image, binary_mask, contour_mask = apply_mask(image_rgb, output_predictions)
    save_and_show_image(image_rgb, segmented_image, binary_mask, contour_mask)

def process_video(video_path):
    model, device = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb, input_image = preprocess_image(frame, device)
        output_predictions = segment_image(model, input_image)
        segmented_image, binary_mask, contour_mask = apply_mask(image_rgb, output_predictions)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

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
        plt.pause(0.001)
        plt.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def crop_img(img, init_point, end_point):
    return img[init_point[0]:end_point[0], init_point[1]:end_point[1], :]

def process_supervideo(supervideo_path):
    model, device = load_model()
    cap = cv2.VideoCapture(supervideo_path)

    if not cap.isOpened():
        print("Error: Could not open supervideo.")
        sys.exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        imgs = [
            crop_img(frame, (0, 0), (720, 1280)),
            crop_img(frame, (0, 1320), (720, 2600)),
            crop_img(frame, (780, 0), (1500, 1280)),
            crop_img(frame, (780, 1320), (1500, 2600))
        ]

        fig, axs = plt.subplots(len(imgs), 4, figsize=(20, 20))

        for i, img in enumerate(imgs):
            image_rgb, input_image = preprocess_image(img, device)
            output_predictions = segment_image(model, input_image)
            segmented_image, binary_mask, contour_mask = apply_mask(image_rgb, output_predictions)

            axs[i, 0].imshow(image_rgb)
            axs[i, 0].set_title(f'Original Image {i+1}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(segmented_image)
            axs[i, 1].set_title(f'Segmented Image {i+1}')
            axs[i, 1].axis('off')

            axs[i, 2].imshow(binary_mask, cmap='gray')
            axs[i, 2].set_title(f'Binary Mask {i+1}')
            axs[i, 2].axis('off')

            axs[i, 3].imshow(contour_mask, cmap='gray')
            axs[i, 3].set_title(f'Contour Mask {i+1}')
            axs[i, 3].axis('off')

        plt.tight_layout()
        plt.pause(0.001)
        plt.draw()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main(image_path=None, video_path=None, supervideo_path=None):
    if ((image_path is None and video_path is None and supervideo_path is None) or
        (image_path is not None and (video_path is not None or supervideo_path is not None)) or
        (video_path is not None and supervideo_path is not None)):
        print("Error: You must specify either an image_path, a video_path, or a supervideo_path, but not multiple.")
        sys.exit()

    if image_path:
        process_image(image_path)
    elif video_path:
        process_video(video_path)
    elif supervideo_path:
        process_supervideo(supervideo_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentação de pessoas usando DeepLabv3+")
    parser.add_argument('--image_path', type=str, help="Caminho para a imagem de entrada")
    parser.add_argument('--video_path', type=str, help="Caminho para o vídeo de entrada")
    parser.add_argument('--supervideo_path', type=str, help="Caminho para o supervídeo de entrada")
    args = parser.parse_args()
    main(args.image_path, args.video_path, args.supervideo_path)
