import cv2
import json
import numpy as np
import os
import argparse

def crop_img(img, init_point, end_point):
    """
    Recorta a imagem dada as coordenadas de início e fim.
    """
    return img[init_point[0]:end_point[0], init_point[1]:end_point[1], :]

def draw_contours(image, contours):
    """
    Desenha os contornos na imagem.
    """
    for contour in contours:
        cv2.drawContours(image, [np.array(contour, dtype=np.int32)], -1, (0, 255, 0), 2)
    return image

def process_supervideo(supervideo_path, json_path):
    """
    Processa o supervídeo e desenha os contornos das pessoas conforme descrito no JSON.
    """
    cap = cv2.VideoCapture(supervideo_path)

    if not cap.isOpened():
        print("Error: Could not open supervideo.")
        return

    with open(json_path, 'r') as f:
        contours_dict = json.load(f)

    video_name = os.path.basename(supervideo_path)
    if video_name not in contours_dict:
        print(f"Error: JSON does not contain data for {video_name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        imgs = [
            cv2.resize(crop_img(frame, (0, 0), (720, 1280)), (256, 256), interpolation=cv2.INTER_LINEAR)    ,
            cv2.resize(crop_img(frame, (0, 1320), (720, 2600)), (256, 256), interpolation=cv2.INTER_LINEAR)    ,
            cv2.resize(crop_img(frame, (780, 0), (1500, 1280)), (256, 256), interpolation=cv2.INTER_LINEAR)    ,
            cv2.resize(crop_img(frame, (780, 1320), (1500, 2600)), (256, 256), interpolation=cv2.INTER_LINEAR)
        ]

        combined_frame = np.zeros((512, 512, 3), dtype=np.uint8)  # Placeholder for combined frame

        for i, img in enumerate(imgs):
            subimage_key = f"subimage_{i+1}"
            if subimage_key in contours_dict[video_name] and frame_idx <= len(contours_dict[video_name][subimage_key]):
                contours = contours_dict[video_name][subimage_key][frame_idx - 1]
                img = draw_contours(img, contours)
            
            # Redimensionar a subimagem para 256x256
            #img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

            y_offset = 256 * (i // 2)
            x_offset = 256 * (i % 2)
            combined_frame[y_offset:y_offset+img.shape[0], x_offset:x_offset+img.shape[1]] = img

        progress_text = f"Frame {frame_idx} de {total_frames}"
        cv2.putText(combined_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Segmented Supervideo', combined_frame)

        # Wait for 'N' key to proceed to the next frame
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()
    
    
def main():
    parser = argparse.ArgumentParser(description="Mostrar contornos de pessoas em um supervídeo usando dados de um JSON.")
    parser.add_argument('--supervideo_path', type=str, required=True, help="Caminho para o supervídeo de entrada")
    parser.add_argument('--json_path', type=str, required=True, help="Caminho para o arquivo JSON com os contornos")
    args = parser.parse_args()

    process_supervideo(args.supervideo_path, args.json_path)

if __name__ == '__main__':
    main()

