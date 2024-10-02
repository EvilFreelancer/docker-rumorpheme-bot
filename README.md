# ruMorphemeBot

ruMorphemeBot — это Telegram-бот для морфемного разбора русских слов. Бот использует обученную модель нейронной сети для
анализа слов и их разделения на морфемы (приставки, корни, суффиксы и т.д.).

Основан на модели [evilfreelancer/ruMorpheme](https://github.com/EvilFreelancer/ruMorpheme).

## Возможности

- отправьте боту русское слово или несколько слов, и он вернет их морфемный разбор;
- поддерживается вывод в формате plain text или JSON Lines;
- можно включить или отключить отображение типов морфем;
- пользователи могут оценивать качество разбора с помощью лайков и дизлайков.

## Примеры работы

| Ввод            | Режим           | Вывод                                                                                                                                                                                                                                                       |
|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| беспрекословный | /plain /types   | `беспрекословный: бес:PREF/прекослов:ROOT/н:SUFF/ый:END`                                                                                                                                                                                                    |
| беспрекословный | /plain /notypes | `беспрекословный: бес/прекослов/н/ый`                                                                                                                                                                                                                       |
| беспрекословный | /jsonl          | `jsonl{"word": "беспрекословный", "morphemes": [{"text": "бес", "type": "PREF", "prob": "99.99"}, {"text": "прекослов", "type": "ROOT", "prob": "94.22"}, {"text": "н", "type": "SUFF", "prob": "90.12"}, {"text": "ый", "type": "END", "prob": "100.0"}]}` |

## Команды

- /start — Показать приветственное сообщение и список доступных команд;
- /help — Показать помощь по командам;
- /plain — Установить формат вывода в plain text;
- /jsonl — Установить формат вывода в формате JSON Lines;
- /types — Включить отображение типов морфем;
- /notypes — Отключить отображение типов морфем;

Примечание: В групповых чатах используйте команды с упоминанием бота, например: `@ruMorphemeBot /jsonl`.

## Переменные окружения

Для настройки бота используются следующие переменные окружения:

- `BOT_TOKEN` — Токен вашего Telegram-бота. Обязательное поле;
- `ADMINS_ID` — ID администратора бота для получения уведомлений. Обязательное поле;
- `DATABASE_URL` — URL базы данных. По умолчанию: `sqlite+aiosqlite:///data/rumorpheme.db`;
- `MODEL_PATH` — Путь к обученной модели или имя репозитория на Hugging Face. По умолчанию:
  `evilfreelancer/ruMorpheme-v0.1`.

## Запуск с помощью Docker

По умолчанию используется Dockerfile на базе `nvidia/cuda:12.6.1-cudnn-runtime-ubuntu24.04` для использования GPU.
Также доступен Dockerfile.cpu на базе `python:3.12` для запуска на CPU.

Пример файла `docker-compose.yml`:

```yml
version: "3.9"
services:
  rumorpheme-bot:
    restart: "unless-stopped"
    build: .
    environment:
      BOT_TOKEN: 123456:xxx
      ADMINS_ID: 123
    volumes:
      - ./rumorpheme_data:/app/data
    logging:
      driver: "json-file"
      options:
        max-size: "10k"
```

Копируем из примера, cобираем, запускаем:

```shell
cp docker-compose.dist.yml docker-compose.yml
docker-compose build
docker-compose up -d
```

## Структура базы данных

Бот использует базу данных для хранения настроек пользователей и логирования сообщений.

Таблица `user_settings`:

- `user_id` — ID пользователя Telegram;
- `format` — Формат вывода (`plain` или `jsonl`). По умолчанию `plain`;
- `use_morpheme_types` — Отображать типы морфем (`yes` или `no`). По умолчанию `yes`.

Таблица `message_log`:

- `id` — Уникальный идентификатор записи;
- `user_id` — ID пользователя Telegram;
- `message_id` — ID сообщения в Telegram;
- `message_text` — Текст сообщения пользователя;
- `response_text` — Текст ответа бота;
- `debug` — Полная информация о разборе в формате JSONL;
- `feedback` — Оценка пользователя (`-1` — дизлайк, `0` — нет оценки, `1` — лайк);
- `timestamp` — Время создания записи.

## Ссылки

- https://github.com/EvilFreelancer/ruMorpheme
- https://huggingface.co/evilfreelancer/ruMorpheme-v0.1
- https://github.com/EvilFreelancer/docker-rumorpheme-bot
