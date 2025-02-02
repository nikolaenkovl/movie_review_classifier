# Movie Review Classifier

## Описание
Этот проект представляет собой веб-приложение на Django для ввода и классификации отзывов о фильмах. Модель, основанная на TinyBERT, предсказывает рейтинг и категорию отзыва (положительный/отрицательный).

## Установка модели

Для использования проекта вам необходимо скачать модель `model_state.pth` и поместить её в корневую папку проекта. Вы можете скачать модель [здесь](https://drive.google.com/file/d/1BoFq9rNJhzGhLPHpwbQoSTZ7vYdZVPIR/view?usp=sharing).

## Установка

1. Клонируйте репозиторий:
    ```bash
    git clone https://github.com/nikolaenkovl/movie_review_classifier)](https://github.com/nikolaenkovl/movie_review_classifier
2. Перейдите в папку проекта:
   ```bash
   cd movie_review_classifier
3. Установите зависимости:
    ```bash
    pip install -r requirements.txt
## Запуск

    python manage.py runserver


## Использование
Перейдите на http://127.0.0.1:8000/ для ввода отзывов и получения их классификации.

