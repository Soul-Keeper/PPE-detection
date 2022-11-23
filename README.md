# PPE detection
 
Данный репозиторий хранит код модуля «Распознавание человека и наличия СИЗ», являющегося частью платформы видеоаналитики для задач нефтегазовой отрасли.

## Запуск
Для запуска необходимо установить библиотеки. Это можно сделать выполнив команду, представленную ниже
```bash
pip install -r requirements.txt
```

Перед запуском модуля проверьте наличие весеов в папке /weigths
Для запуска модуля используйте команду из корня каталога
```bash
python main.py
```
- Для того, чтобы выбрать свой видео файл используйте флаг --video_path.
- Для сохранения обработанного видео испольузуйте флаг --save_video.
- Для теста производительности используйте флаг --perfomance_test.

Процесс выбора данных, результаты обучения моделей и многое другое представлены в [отчете](https://github.com/Soul-Keeper/PPE-detection/blob/main/отчёт.docx)

## Пример
Пример работы модуля представлен ниже


https://user-images.githubusercontent.com/60879213/203539565-6d996ab9-3d03-4175-9535-c4d95d67fdf5.mp4




