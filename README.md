# Волга-IT 2026, полуфинал
Проект по распознованию нестандартных (желтые и квадртные) государственных (Российских) регистрационных знаков трантранспортных средств. 
`solution/` - решение, где лежит код, обученные веса и скрипты обучения. 
`dataset/` - датасет, который использовался для обучения, где 613 реальных фото + 5000 синтетических.


## Запуск решения

Нужен Python 3.11-3.13. Из папки `solution/`:

```bash
python -m venv .venv
.venv\Scripts\activate                 
pip install -r requirements.txt       
python -m src.run --input <каталог с изображениями> --output result.csv
```

Результат - `result.csv` (UTF-8, разделитель `;`, первая строка - заголовок):

```
image;plate_num;plate_type;confidence
img_001.jpg;A001MC778;type1a;0.93
```

Несколько знаков на кадре дают несколько строк, кадр без знаков строк не дает.

### Docker

Из папки `solution/`:

```bash
docker build -t alpr-ru .
docker run --rm -v <каталог с изображениями>:/data/images -v <каталог для результата>:/data/out alpr-ru
```

Результат - `/data/out/result.csv`. Вариант для GPU: `docker build -f Dockerfile.gpu -t alpr-ru-gpu .`,
запуск с `--gpus all`.

## Как устроено

```
кадр -> детектор (YOLOv9-t, 3 класса) -> выпрямление знака по 4 углам -> тип знака
     -> OCR (1А: две строки склеиваются в одну) -> декодирование по маске номера -> отсев other -> CSV
```

```
src/
  run.py           командная строка: каталог -> CSV
  pipeline.py      сборка стадий, замер времени по стадиям, отрисовка для отладки
  detect.py        детектор (ONNX Runtime) + NMS; необязательный проход по кускам кадра
  rectify.py       поиск 4 углов знака (Otsu, контуры) и перспективная коррекция; если углы не найдены - рамка
  classify.py      тип знака: метка детектора + поправки по доле жёлтого и пропорциям
  ocr.py           OCR (ONNX), склейка строк 1А, декодирование по маске позиций (только буквы ABEKMHOPCTYX)
  postprocess.py   проверка структуры номера, уверенность, перевод сомнительного в other, дедупликация
  vehicle.py       необязательная проверка «знак на ТС» (выключена)
  config.py        чтение config.yaml и переменных окружения
train/             обучение детектора и OCR
weights/           модели: детектор detector_416.onnx, OCR ocr_ru.onnx
config.yaml        пороги и параметры
```

## Скорость

Скорость (Intel Core i5-12400F, ONNX Runtime на CPU): в среднем 25.3 мс на
изображение (p95 31 мс), из них детектор около 22 мс, OCR около 3 мс.


## Обучение

Отдельное окружение (из папки `solution/`):

```bash
python -m venv .venv-train
.venv-train\Scripts\activate          
pip install -r requirements-train.txt
pip install --force-reinstall torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
```

Детектор (YOLOv9-t с COCO-претрейна, вход 416). Реальные фото в train повторяются 3 раза
(`--real-repeat 3`). В сдаче лежит модель последней эпохи (`last.pt`), поэтому
после обучения она экспортируется отдельной командой:

```bash
python train/train_detector.py --dataset ../dataset --out runs/detector --model yolov9t.pt --imgsz 416 --epochs 100 --batch 16 --seed 42 --real-repeat 3
python train/train_detector.py --dataset ../dataset --export-only runs/detector/train/weights/last.pt
```

Результат - `weights/detector_416.onnx` и `weights/detector_classes.yaml`.

OCR (дообучение `cct-xs-v2-global` из fast-plate-ocr, результат - `weights/ocr_ru.onnx`).
Переменная `KERAS_BACKEND=torch` нужна, чтобы обучение шло на видеокарте: команда `set` ниже для cmd,
в PowerShell вместо нее `$env:KERAS_BACKEND="torch"`.

```bash
set KERAS_BACKEND=torch               
python train/train_ocr.py --dataset ../dataset
```


## Лицензия

MIT (`LICENSE`): FastALPR (c) ankandrew, изменения и новый код мои.
Ultralytics (AGPL-3.0) используется только при обучении детектора и в работу решения не входит.