import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import functional as F
import argparse

def load_model():
    """
    Carrega o modelo DeepLabv3+ pré-treinado.
    """
    model = models.segmentation.deeplabv3_resnet101(
        weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
    ).eval()
    return model

def preprocess_image(image_path):
    """
    Carrega e pre-processa a imagem para o modelo.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = F.to_tensor(image_rgb)
    input_image = input_image.unsqueeze(0)
    return image, image_rgb, input_image

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

def main(image_path):
    model = load_model()
    image, image_rgb, input_image = preprocess_image(image_path)
    output_predictions = segment_image(model, input_image)
    segmented_image, binary_mask, contour_mask = apply_mask(image_rgb, output_predictions)
    save_and_show_image(image_rgb, segmented_image, binary_mask, contour_mask)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentação de pessoas usando DeepLabv3+")
    parser.add_argument('image_path', type=str, help="Caminho para a imagem de entrada")
    args = parser.parse_args()
    main(args.image_path)
