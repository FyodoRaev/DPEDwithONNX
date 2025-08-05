import os
import argparse
import numpy as np
import cv2
import onnxruntime as ort
from tqdm import tqdm # Добавим прогресс-бар для удобства

def run_onnx_inference(image_low_res_np, ort_session):
    """
    Выполняет инференс на УЖЕ загруженной ONNX модели.

    Args:
        image_low_res_np (np.array): Изображение низкого разрешения (H, W, 3), float32 [0, 1].
        ort_session (ort.InferenceSession): Активная сессия ONNX Runtime.

    Returns:
        np.array: Улучшенное изображение (H, W, 3), float32 [0, 1].
    """
    # Модель ожидает батч изображений, поэтому добавляем первую ось: (H, W, 3) -> (1, H, W, 3)
    image_for_model = np.expand_dims(image_low_res_np, axis=0)

    # Получаем имена входа и выхода из самой модели
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # Запускаем инференс
    # ort_session.run() возвращает список с результатами для каждого выхода
    enhanced_batch = ort_session.run([output_name], {input_name: image_for_model})[0]

    # Убираем лишнюю ось батча, возвращая изображение (1, H, W, 3) -> (H, W, 3)
    enhanced_image = np.squeeze(enhanced_batch, axis=0)

    return enhanced_image


def enhance_high_res_onnx(input_path, output_path, low_res_width, ort_session):
    """
    Улучшает изображение высокого разрешения, используя ONNX модель низкого разрешения
    и Guided Filter для апскейлинга.
    """
    A_bgr = cv2.imread(input_path)
    if A_bgr is None:
        print(f"\nПредупреждение: не удалось прочитать изображение {input_path}, пропуск.")
        return

    # --- 1. Подготовка изображений ---
    A = cv2.cvtColor(A_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w, _ = A.shape

    scale_factor = low_res_width / w
    low_res_height = int(h * scale_factor)
    a = cv2.resize(A, (low_res_width, low_res_height), interpolation=cv2.INTER_AREA)

    # --- 2. Инференс на ONNX модели ---
    b = run_onnx_inference(a, ort_session)
    # Конвертируем все в LAB. Важно, что данные - float32 [0,1]
    A_lab = cv2.cvtColor(A, cv2.COLOR_RGB2Lab)
    a_lab = cv2.cvtColor(a, cv2.COLOR_RGB2Lab)
    b_lab = cv2.cvtColor(b, cv2.COLOR_RGB2Lab)

    delta_lab = b_lab - a_lab

    print("Выполнение Guided Upsampling дельты...")
    full_h, full_w = A.shape[:2]
    delta_lab_upscaled = cv2.resize(delta_lab, (full_w, full_h), interpolation=cv2.INTER_CUBIC)


    # Извлекаем L канал из полноразмерного изображения (диапазон [0, 100])
    L_A = A_lab[:, :, 0]
    # Нормализуем его к диапазону [0, 1] для стабильной работы фильтра
    L_A_norm = L_A / 100.0
    # Убедимся, что тип данных остался float32
    L_A_norm = L_A_norm.astype(np.float32)

    # Параметры для guidedFilter. Возможно, eps придется подстроить для нормализованного гайда.
    # Например, (0.02)^2 = 4e-4 или (0.1)^2 = 1e-2. Начнем с вашего.
    radius = 24
    eps = 1e-5  # Можно попробовать увеличить, например до 1e-3, если результат будет шумным

    # Используем L_A_norm в качестве направляющего изображения
    print(f"Применение Guided Filter с radius={radius}, eps={eps}")
    delta_L_high = cv2.ximgproc.guidedFilter(guide=L_A_norm, src=delta_lab_upscaled[:, :, 0], radius=radius, eps=eps)
    delta_a_high = cv2.ximgproc.guidedFilter(guide=L_A_norm, src=delta_lab_upscaled[:, :, 1], radius=radius, eps=eps)
    delta_b_high = cv2.ximgproc.guidedFilter(guide=L_A_norm, src=delta_lab_upscaled[:, :, 2], radius=radius, eps=eps)
    # ----------------------------------------------------------------

    delta_high_lab = cv2.merge([delta_L_high, delta_a_high, delta_b_high])

    # --- Адаптивное ослабление ---
    # Берем оригинальную яркость (диапазон 0-100)
    L_A = A_lab[:, :, 0]

    # Создаем весовую карту. Где L_A близко к 100 (светло), вес будет близок к 0.
    # Где L_A близко к 0 (темно), вес будет близок к 1.
    # Функция (1 - x)^gamma позволяет управлять крутизной спада.
    gamma = 1.5  # Значение > 1 сильнее защищает света. Попробуйте 1.0, 1.5, 2.0
    highlight_protection_map = (1.0 - L_A / 100.0) ** gamma
    # Добавляем новую ось, чтобы маску можно было умножить на 3-канальную дельту
    highlight_protection_map = highlight_protection_map[:, :, np.newaxis]

    # Применяем маску только к дельте яркости
    delta_L, delta_a, delta_b = cv2.split(delta_high_lab)
    delta_L_adjusted = delta_L * highlight_protection_map[:, :, 0]  # Применяем 2D маску к 2D каналу

    # Собираем дельту обратно, но уже с ослабленной яркостью
    delta_high_lab_adjusted = cv2.merge([delta_L_adjusted, delta_a, delta_b])
    # -----------------------------

    # Теперь применяем эту скорректированную дельту
    B_lab = A_lab + delta_high_lab_adjusted
    print("Guided Upsampling завершено.")

    # Защитное отсечение (clipping) до валидного диапазона LAB ---
    # Это предотвратит артефакты при конвертации Lab -> RGB
    B_lab[:, :, 0] = np.clip(B_lab[:, :, 0], 0, 100)
    B_lab[:, :, 1] = np.clip(B_lab[:, :, 1], -128, 127)
    B_lab[:, :, 2] = np.clip(B_lab[:, :, 2], -128, 127)
    # --------------------------------------------------------------------------

    # Конвертируем обратно в RGB
    B_rgb = cv2.cvtColor(B_lab, cv2.COLOR_Lab2RGB)

    # Финальное отсечение для RGB (на всякий случай) и конвертация в uint8 для сохранения
    B_to_save = np.clip(B_rgb * 255.0, 0, 255).astype(np.uint8)
    B_bgr_to_save = cv2.cvtColor(B_to_save, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, B_bgr_to_save)
    print(f"Результат успешно сохранен в: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Улучшение папки с изображениями с помощью ONNX модели.")

    # parser.add_argument('--model', type=str, required=True, default="onnx_models/iphone_orig.onnx",
    #                     help="Путь к файлу .onnx модели.")
    model = "onnx_models/blackberry_orig.onnx"


    parser.add_argument('--input_dir', type=str, default="full_size_test_images",
                        help="Папка с исходными изображениями.")
    parser.add_argument('--output_dir', type=str, default="results",
                        help="Папка для сохранения результатов.")
    parser.add_argument('--width', type=int, default=1024,
                        help="Ширина low-res изображения для подачи в нейросеть.")
    args = parser.parse_args()

    # --- Проверки ---
    if not os.path.exists(model):
        print(f"Ошибка: Файл ONNX модели не найден: {model}")
        exit()
    if not os.path.isdir(args.input_dir):
        print(f"Ошибка: Входная папка не найдена: {args.input_dir}")
        exit()

    # --- Загрузка ONNX модели в память (один раз) ---
    print(f"Загрузка ONNX модели: {model}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(model, providers=providers)

    # Проверка, какой провайдер реально используется
    print("Используемые провайдеры:", ort_session.get_providers())
    print("Модель успешно загружена и готова к работе.")


    # Создаем уникальную папку для каждого запуска
    run_num = 1
    base_output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)
    while os.path.exists(os.path.join(base_output_dir, f'run_{run_num}')):
        run_num += 1
    output_run_dir = os.path.join(base_output_dir, f'run_{run_num}')
    os.makedirs(output_run_dir, exist_ok=True)
    print(f"Результаты будут сохранены в: {os.path.abspath(output_run_dir)}")

    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"В папке {args.input_dir} не найдено изображений.")
        exit()

    print(f"Найдено {len(image_files)} изображений для обработки.")

    for filename in tqdm(image_files, desc="Обработка изображений"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(output_run_dir, filename)
        enhance_high_res_onnx(input_path, output_path, args.width, ort_session)

    print("\nВсе изображения успешно обработаны!")
