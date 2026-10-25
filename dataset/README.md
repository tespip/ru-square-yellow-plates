# Датасет российских номерных знаков 

Права: CC BY 4.0

Делала для полуфинала Волга-IT 2026, номинация «ИИ и анализ данных». Здесь знаки типов 1, 1А
(квадратные, двухстрочные) и 1Б (жёлтые) по ГОСТ Р 50577-2018, плюс
«other» - все, что на целевые типы не похоже (знаки прицепов, мотоциклов, иностранные).

Всего 5613 изображений и 5679 знаков: 613 реальных фото (679 знаков) и 5000 синтетических кадров.

## Что где лежит

```
images/real/         реальные фото
images/synthetic/    синтетика
labels/              разметка, один .txt на картинку (имя то же, что у картинки)
meta.csv             описание каждого знака, одна строка = один знак
generator/           скрипт, которым сделана синтетика
README.md, LICENSE
```

Файлы называются `<тип>_<номер>.jpg`, например `type1a_0001.jpg`. Тип в имени это тип первого
знака на фото на момент разметки. Нумерация у синтетики продолжает нумерацию реальных фото.

## Формат разметки

### meta.csv

Разделитель `;`, кодировка UTF-8, в первой строке заголовок:

```
image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions
```

| Поле | Что это |
|---|---|
| `image` | путь к картинке от корня датасета |
| `plate_num` | номер латиницей, заглавными; нечитаемая буква или цифра - `#` |
| `plate_type` | `type1`, `type1a`, `type1b` или `other` |
| `bbox` | `x,y,w,h` в пикселях, прямоугольник вокруг `quad` |
| `quad` | четыре угла знака `x1,y1,...,x4,y4` по часовой стрелке, начиная с левого верхнего |
| `is_vehicle` | 1 - знак на машине, 0 - все остальное |
| `is_synthetic` | 1 - синтетика, 0 - реальное фото |
| `source`, `license` | откуда фото и под какой лицензией |
| `conditions` | через запятую: `day`, `night`, `rain`, `snow`, `dirt`, `glare`, `motion_blur`, `angle` |

Маски номеров. Для `type1` и `type1a`: буква, три цифры, две буквы, регион из 2-3 цифр
(`A123BC777`). Для `type1b`: две буквы, три цифры, регион (`AB12377`). Буквы только из набора
`A B E K M H O P C T Y X`. Для `other` пишем текст со знака, если его можно прочитать
(например, знаки прицепов вида `5212KC42`).

### labels/*.txt

Формат YOLO, одна строка - один знак, все числа поделены на размер картинки:

```
<класс> <cx> <cy> <w> <h> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
```

Классы: `0` - type1, `1` - type1a, `2` - type1b, `3` - other. У синтетики строка короче:
`класс cx cy w h`, без четырех углов (углы для неё есть в `meta.csv`).

## Что в датасете

### По типам

| Тип | Реальных фото | Реальных знаков | Синтетика |
|---|---|---|---|
| type1 | 131 | 157 | 987 |
| type1a (квадратные) | 157 | 158 | 1787 |
| type1b (жёлтые) | 300 | 309 |  1739 |
| other | 55 | 55 | 487 |
| Итого | 613 | 679 | 5000 |

### Условия съёмки (по числу знаков)

| Тег | Реальные | Синтетика |
|---|---|---|
| day | 656 | 3991 |
| night | 23 | 1009 |
| angle | 42 | 1390 |
| motion_blur | 34 | 601 |
| glare | 13 | 932 |
| dirt | 3 | 995 |
| snow | 3 | 291 |
| rain | 0 | 408 |

## Источники и лицензии

### Реальные фото

| Источник | Знаков | Лицензия |
|---|---|---|
| Собственная съемка (`own_photo`) | 175 | CC BY 4.0 |
| Wikimedia Commons (ссылка на файл в поле `source`) | 240 | CC0; public domain; CC BY 4.0 |
| Roboflow: [platerecognition/licenseplates-wql9s](https://universe.roboflow.com/platerecognition/licenseplates-wql9s) | 91 | CC BY 4.0 |
| AUTO.RIA Numberplate Options Dataset (ARS Online OU), через [huggingface.co/datasets/AY000554/Car_plate_detecting_dataset](https://huggingface.co/datasets/AY000554/Car_plate_detecting_dataset) | 87 | CC BY 4.0 |
| Roboflow: [testcarplate/russian-license-plates-classification-by-this-type](https://universe.roboflow.com/testcarplate/russian-license-plates-classification-by-this-type) | 53 | CC BY 4.0 |
| Roboflow: [fverwfgerwf/two-line-russian-license-plates](https://universe.roboflow.com/fverwfgerwf/two-line-russian-license-plates) | 24 | CC BY 4.0 |
| Roboflow: [skat-wjz9d/russian-license-plates-detector-f4bzb](https://universe.roboflow.com/skat-wjz9d/russian-license-plates-detector-f4bzb) | 6 | CC BY 4.0 |
| Roboflow: [school-bzbp2/autoria-numberplate-dataset-s2xha](https://universe.roboflow.com/school-bzbp2/autoria-numberplate-dataset-s2xha) | 2 | CC BY 4.0 |
| Roboflow: [symbolplate/russian-license-plate-characters](https://universe.roboflow.com/symbolplate/russian-license-plate-characters) | 1 | CC BY 4.0 |

### Синтетика

Фоны взяты из AUTO.RIA Numberplate Options Dataset (ARS Online OU), CC BY 4.0, (c) 2018-2024 ARS Online OU. 
Шрифты, используемые на номерах, под лицензией SIL OFL 1.1: PT Sans Narrow Bold (ParaType Ltd.) и Roboto Condensed Bold
(The Roboto Project Authors). Код генератора под MIT, за основу взят deep-anpr (Matthew Earl, MIT)

## Методика

### Сбор и разметка реальных фото

Фото снимала сама, искала на Wikimedia Commons по свободным лицензиям, брала датасеты с Roboflow Universe и AUTO.RIA под CC BY 4.0.
Для разметки использовала свой инструмент `label_tool.html`: для каждого знака кликаю по четырем углам, `bbox` считается из них,
номер и тип ввожу руками. Для кадров из готовых датасетов рамки исходного датасета служили предразметкой (стартовые углы), а номер, 
тип и углы я проверяла и правила сама.

### Синтетика

Синтетику я делала скриптом `generator/gen.py`: он берет реальный кадр с машиной, закрашивает старый
номер и рисует на его месте новый случайный номер по ГОСТ. Потом добавляет эффекты (ночь, дождь,
снег, грязь, блик, смаз). Углы знака известны из самого рисования, поэтому разметка точная.

## Генератор синтетики

В папке `generator/`:

```
gen.py                    основной скрипт
plates.py                 рисует номер и плашку по размерам ГОСТ
effects.py                эффекты (ночь, дождь, снег, грязь, блик, смаз) и теги conditions
fonts/                    шрифты и их лицензии
backgrounds/              фоны и manifest.csv (источник и лицензия каждого фона)
requirements.txt          версии пакетов
THIRD_PARTY_NOTICES.md    чужой код и шрифты, которые использованы
```

Генератору нужен `opencv-python`, а решению `opencv-python-headless`. В одном окружении они
мешают друг другу, поэтому генератор ставится в отдельное. Команды из папки `dataset/`:

```bash
python -m venv .venv-gen
.venv-gen\Scripts\activate
pip install -r generator/requirements.txt
python generator/gen.py --n 5000 --seed 42 --merge --start-index type1=3,type1a=155,type1b=306,other=55
```

Последняя команда собирает именно этот набор. Seed фиксированный, поэтому повторный запуск с
версиями пакетов из `requirements.txt` дает те же файлы байт в байт. Перед запуском генератор
очищает `images/synthetic/` и заменяет строки синтетики в `meta.csv`, старый `meta.csv` сохраняется
в `meta_backup_<время>.csv`. Без `--merge` строки пишутся в отдельный `meta_synthetic.csv`.

Фоны - 2149 кадров AUTO.RIA ровно с одним номером. Из 2563 исходных кадров убраны 108, где
размечено два и более номеров, и 306, где детектор номеров нашел еще один, неразмеченный номер.
Лица на фонах размыты автоматически, вручную это не проверялось.





