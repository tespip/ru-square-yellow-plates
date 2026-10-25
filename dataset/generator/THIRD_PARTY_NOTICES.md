# Third-party notices

## deep-anpr (Matthew Earl) — MIT License

Генератор (`gen.py`, `plates.py`, `effects.py`) использует как каркас код проекта
deep-anpr: https://github.com/matthewearl/deep-anpr (файлы `gen.py`, `common.py`).
Из него взяты и переписаны: `euler_to_mat` (матрица поворота по углам рысканья,
тангажа и крена), `rounded_rect` (маска плашки со скруглёнными углами), схема
«отрисовка символов шрифтом Pillow -> наложение плашки на фон через маску ->
шум», загрузка шрифтов из папки. Код портирован на Python 3, вывод сделан
цветным, геометрия и рисунок плашек заменены на российские ГРЗ.
Фоны SUN397 и шрифт UKNumberPlate.ttf из deep-anpr не используются.

Текст лицензии (из репозитория deep-anpr; в заголовке `gen.py` указано
«Copyright (c) 2016 Matthew Earl», в файле LICENSE — «Copyright (c) 2015 matthewearl»):

```
The MIT License (MIT)

Copyright (c) 2015 matthewearl
Copyright (c) 2016 Matthew Earl

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Шрифты (`fonts/`) — SIL Open Font License 1.1

| Файл | Шрифт | Правообладатель | Лицензия |
|---|---|---|---|
| `2220.ttf` | PT Sans Narrow Bold | ParaType Ltd. | `fonts/OFL.txt` |
| `RobotoCondensed-Bold.ttf` | Roboto Condensed Bold | The Roboto Project Authors | `fonts/OFL-RobotoCondensed.txt` |

`RobotoCondensed-Bold.ttf` — статичный экземпляр (ось wght=700), полученный из
`RobotoCondensed[wght].ttf` (https://github.com/google/fonts/tree/main/ofl/robotocondensed)
командой `fonttools varLib.instancer "RobotoCondensed[wght].ttf" wght=700 --update-name-table`
(fonttools 4.60.1). Генератор принимает только статичные шрифты. Reserved Font Name у Roboto нет.
Генератор не запускается, если в папке шрифтов нет файла лицензии.

## Детектор номеров для отбора фонов (вне датасета)

Фоны с неразмеченными номерами других машин отсеяны детектором
yolo-v9-t-384-license-plates-end2end из open-image-models (https://github.com/ankandrew/open-image-models,
MIT), так исключено 306 фонов. Детектор нужен только при подготовке фонов, генератор его не использует.

## Фоны (`backgrounds/`)

Источник, лицензия и автор каждого фона записаны в `backgrounds/manifest.csv`
и переносятся в колонки `source` и `license` каждой строки `meta.csv`.
