import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

# Escolher o modelo
model_type = "vit_h"  # Pode ser "vit_h", "vit_l", "vit_b"
sam_checkpoint = f'checkpoints/sam_{model_type}_4b8939.pth'

# Carregar a imagem
image_path = 'in/example.jpg'
image = cv2.imread(image_path)

# Carregar o modelo SAM
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Configurar a imagem no predictor
predictor.set_image(image)

# Realizar a segmentação
masks, scores, logits = predictor.predict()

# Selecionar a máscara correspondente às pessoas
# Supondo que o modelo SAM tenha uma maneira de identificar máscaras de pessoas. 
# Caso contrário, seria necessário um passo adicional para filtrar somente as pessoas.

# Aqui consideramos que masks já contém a segmentação de pessoas.
# Filtragem baseada em escores ou outro critério pode ser adicionada conforme necessário.

# Aplicar a máscara na imagem
segmented_image = image.copy()
for mask in masks:
    segmented_image[mask == 0] = 0  # Definir a área fora da máscara como preta

# Salvar ou mostrar a imagem segmentada
cv2.imwrite('segmented_people.jpg', segmented_image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
