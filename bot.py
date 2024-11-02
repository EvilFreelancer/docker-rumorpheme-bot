import logging
import asyncio
from asyncio import sleep
from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters import Command
import torch
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import select, Column, Integer, String, Text as SQLText, DateTime, func
from config import Config, load_config
import json
import re
import numpy as np

# Импортируем необходимые функции из модели
from rumorpheme.model import RuMorphemeModel
from rumorpheme.utils import labels_to_morphemes

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s:%(lineno)d #%(levelname)-8s [%(asctime)s] - %(name)s - %(message)s",
)

# Инициализируем бота и диспетчер для aiogram v3
config: Config = load_config()
bot = Bot(token=config.TOKEN, default=DefaultBotProperties(parse_mode='Markdown'))
dp = Dispatcher()
router = Router()
dp.include_router(router)

# Создаем глобальный процессный пул для обработки запросов
executor = ProcessPoolExecutor(max_workers=4)

# База данных
Base = declarative_base()


# Модель для хранения настроек пользователя
class UserSettings(Base):
    __tablename__ = 'user_settings'
    user_id = Column(Integer, primary_key=True)
    format = Column(String, default='plain')  # 'plain' или 'jsonl'
    use_morpheme_types = Column(String, default='yes')  # 'yes' или 'no'


# Модель для логирования сообщений
class MessageLog(Base):
    __tablename__ = 'message_log'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    message_id = Column(Integer)  # ID сообщения в Telegram
    message_text = Column(SQLText)
    response_text = Column(SQLText)
    debug = Column(SQLText)  # Полная информация в формате JSONL
    feedback = Column(Integer, default=0)  # -1: дизлайк, 0: нет оценки, 1: лайк
    timestamp = Column(DateTime, default=func.now())


# Настройка движка и сессии для базы данных
engine = create_async_engine(config.DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def load_model():
    # Settings
    model_path = config.MODEL_PATH

    # Загружаем модель
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RuMorphemeModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Сохраняем необходимые данные модели
    return model


async def update_user_setting(user_id, format=None, use_morpheme_types=None):
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.get(UserSettings, user_id)
            if not result:
                result = UserSettings(user_id=user_id)
                session.add(result)
            if format:
                result.format = format
            if use_morpheme_types:
                result.use_morpheme_types = use_morpheme_types
            await session.commit()


async def get_user_settings(user_id):
    async with AsyncSessionLocal() as session:
        result = await session.get(UserSettings, user_id)
        if result:
            return {
                'format': result.format,
                'use_morpheme_types': result.use_morpheme_types
            }
        else:
            return {
                'format': 'plain',
                'use_morpheme_types': 'yes'
            }


async def log_message(user_id, message_id, message_text, response_text, debug_info):
    async with AsyncSessionLocal() as session:
        async with session.begin():
            log_entry = MessageLog(
                user_id=user_id,
                message_id=message_id,
                message_text=message_text,
                response_text=response_text,
                debug=debug_info,
                feedback=0  # Изначально нет оценки
            )
            session.add(log_entry)
            await session.commit()


@router.message(Command("start"))
async def cmd_start(message: Message):
    help_text = (
        "Привет! Я бот для проведения морфемного анализа и сегментации слов русского языка. "
        "Просто отправь мне русские слова, и я разберу их по морфемам.\n\n"
        "Доступные команды:\n"
        "/start - Показать это сообщение\n"
        "/plain - Установить формат вывода в plain text\n"
        "/jsonl - Установить формат вывода в JSON Lines\n"
        "/types - Включить отображение типов морфем в режиме plain\n"
        "/notypes - Отключить отображение типов морфем в режиме plain\n\n"
        "Ссылки: "
        "[код обучения](https://github.com/EvilFreelancer/ruMorpheme), "
        "[веса модели](https://huggingface.co/evilfreelancer/ruMorpheme-v0.1), "
        "[код бота](https://github.com/EvilFreelancer/docker-rumorpheme-bot)"
    )
    await message.reply(help_text, disable_web_page_preview=True)


@router.message(Command("help"))
async def cmd_help(message: Message):
    await cmd_start(message)


@router.message(Command("plain"))
async def set_plain_format(message: Message):
    await update_user_setting(message.from_user.id, format='plain')
    await message.reply("Формат установлен на plain text.")


@router.message(Command("jsonl"))
async def set_jsonl_format(message: Message):
    await update_user_setting(message.from_user.id, format='jsonl')
    await message.reply("Формат установлен на JSON Lines.")


@router.message(Command("types"))
async def set_use_morpheme_types(message: Message):
    await update_user_setting(message.from_user.id, use_morpheme_types='yes')
    await message.reply("Морфемные типы будут отображаться.")


@router.message(Command("notypes"))
async def unset_use_morpheme_types(message: Message):
    await update_user_setting(message.from_user.id, use_morpheme_types='no')
    await message.reply("Морфемные типы не будут отображаться.")


@router.message(F.chat.type == 'private')
async def handle_private_message(message: Message):
    await process_user_message(message)


@router.message(F.chat.type.in_({'group', 'supergroup'}))
async def handle_group_message(message: Message, bot: Bot):
    bot_username = (await bot.get_me()).username
    # Проверяем, есть ли упоминание бота в сообщении
    if message.entities:
        for entity in message.entities:
            if entity.type in ('mention', 'text_mention'):
                mentioned_username = message.text[entity.offset + 1: entity.offset + entity.length]
                if mentioned_username == bot_username:
                    await process_user_message(message)
                    break
    # Проверяем, начинается ли сообщение с команды, адресованной боту
    elif message.text and message.text.startswith(f'@{bot_username}'):
        await process_user_message(message)


def process_text(text, format, use_morpheme_types):
    # Обработка текста с учетом настроек пользователя
    model = load_model()

    # Препроцессинг текста
    words = text.strip().split()
    russian_letter_pattern = re.compile(r'[а-яА-ЯёЁ]')
    processed_words = []
    for word in words:
        # Удаляем начальные символы, не являющиеся русскими буквами
        while word and not russian_letter_pattern.match(word[0]):
            word = word[1:]
        # Удаляем конечные символы, не являющиеся русскими буквами
        while word and not russian_letter_pattern.match(word[-1]):
            word = word[:-1]
        if word:
            processed_words.append(word.lower())

    if not processed_words:
        return "Нет русских слов для обработки.", None

    # Прогон модели
    all_predictions, all_log_probs = model.predict(processed_words)

    # Обработка и форматирование результатов
    results = []
    for idx, word in enumerate(processed_words):
        morphs, morph_types, morph_probs = labels_to_morphemes(
            word.lower(),
            all_predictions[idx],
            all_log_probs[idx]
        )

        # Собираем морфемы, их типы и вероятности в объекты
        morpheme_list = []
        for morpheme, morpheme_type, morpheme_prob in zip(morphs, morph_types, morph_probs):
            morpheme_info = {"text": morpheme}
            morpheme_info["type"] = morpheme_type
            morpheme_info["prob"] = str(np.round(morpheme_prob, 2))
            morpheme_list.append(morpheme_info)

        # Формируем результат для слова
        word_result = {"word": word, "morphemes": morpheme_list}
        results.append(word_result)

    # Формируем debug информацию (полный результат в формате JSONL)
    debug_info = '\n'.join(json.dumps(result, ensure_ascii=False) for result in results) + '\n'

    # Форматируем результат в соответствии с настройками пользователя
    if format == 'jsonl':
        output = debug_info
    else:
        output_lines = []
        for res in results:
            if use_morpheme_types == 'yes':
                morphemes_info = '/'.join([f"{m['text']}:{m['type']}" for m in res['morphemes']])
            else:
                morphemes_info = '/'.join([m['text'] for m in res['morphemes']])
            output_lines.append(f"{res['word']}: {morphemes_info}")
        output = '\n'.join(output_lines)

    return '```\n' + output + '\n```', debug_info


async def process_user_message(message: Message):
    text = message.text
    user_id = message.from_user.id

    # Убираем упоминание бота из текста (в группах)
    bot_username = (await bot.get_me()).username
    if message.entities:
        for entity in message.entities:
            if entity.type in ('mention', 'text_mention'):
                mentioned_username = message.text[entity.offset + 1: entity.offset + entity.length]
                if mentioned_username == bot_username:
                    # Удаляем упоминание из текста
                    text = message.text.replace(f'@{bot_username}', '').strip()
                    break

    # Получаем настройки пользователя
    user_settings = await get_user_settings(user_id)
    format = user_settings['format']
    use_morpheme_types = user_settings['use_morpheme_types']

    loop = asyncio.get_running_loop()
    # Запускаем обработку текста в отдельном процессе с учетом настроек
    result, debug_info = await loop.run_in_executor(
        executor, process_text, text, format, use_morpheme_types
    )

    # Отправляем результат пользователю с кнопками лайка/дизлайка
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="👍", callback_data=f"like:{message.message_id}:{user_id}"),
            InlineKeyboardButton(text="👎", callback_data=f"dislike:{message.message_id}:{user_id}")
        ]
    ])
    sent_message = await message.reply(result, reply_markup=keyboard)

    # Логируем сообщение и ответ
    await log_message(
        user_id=user_id,
        message_id=sent_message.message_id,
        message_text=text,
        response_text=result,
        debug_info=debug_info
    )


@router.callback_query(F.data.startswith('like:'))
async def handle_like(callback_query: CallbackQuery):
    await update_feedback(callback_query, 1)


@router.callback_query(F.data.startswith('dislike:'))
async def handle_dislike(callback_query: CallbackQuery):
    await update_feedback(callback_query, -1)


async def update_feedback(callback_query: CallbackQuery, feedback_value: int):
    user_id = callback_query.from_user.id
    data_parts = callback_query.data.split(':')
    message_id = int(data_parts[1])
    original_user_id = int(data_parts[2])  # Получаем ID оригинального пользователя

    # Проверяем, что только отправитель может оставить оценку
    if user_id != original_user_id:
        await callback_query.answer("Вы не можете оценить этот ответ.", show_alert=True)
        return

    async with AsyncSessionLocal() as session:
        async with session.begin():
            # Ищем запись в MessageLog по user_id и message_id
            stmt = select(MessageLog).where(
                MessageLog.user_id == user_id,
                MessageLog.message_id == callback_query.message.message_id
            )
            result = await session.execute(stmt)
            entry = result.scalar_one_or_none()
            if entry:
                # Обновляем feedback в зависимости от предыдущего значения
                if entry.feedback == feedback_value:
                    entry.feedback = 0  # Отменяем предыдущий выбор
                    response_text = "Ваша оценка отменена."
                else:
                    entry.feedback = feedback_value  # Устанавливаем новую оценку
                    response_text = "Спасибо за вашу оценку!"

                await session.commit()

                # Обновляем кнопки
                keyboard = generate_feedback_keyboard(entry.feedback, original_user_id, message_id)
                await callback_query.message.edit_reply_markup(reply_markup=keyboard)
            else:
                response_text = "Сообщение не найдено."

    # Уведомляем пользователя
    await callback_query.answer(response_text, show_alert=False)


def generate_feedback_keyboard(feedback_value: int, user_id: int, message_id: int):
    # Генерируем клавиатуру с учетом текущего значения feedback
    if feedback_value == 1:
        like_button_text = "👍✅"
        dislike_button_text = "👎"
    elif feedback_value == -1:
        like_button_text = "👍"
        dislike_button_text = "👎✅"
    else:
        like_button_text = "👍"
        dislike_button_text = "👎"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text=like_button_text, callback_data=f"like:{message_id}:{user_id}"),
            InlineKeyboardButton(text=dislike_button_text, callback_data=f"dislike:{message_id}:{user_id}")
        ]
    ])
    return keyboard


async def notify_admins(bot: Bot, admin_ids: list, message: str = None):
    logger.info("Notify admin...")
    for admin_id in admin_ids:
        try:
            await bot.send_message(admin_id, message, disable_notification=True)
            logger.debug(f"Message sent {admin_id}")
        except Exception:
            logger.debug("Chat with admin not found")

        await sleep(0.3)


# Инициализируем базу данных и запускаем бота
async def main():
    logger.info("Starting bot")

    # Send notification to admin
    await notify_admins(bot, admin_ids=config.ADMINS_IDS, message="Bot successfully started")

    # Init bot
    await init_db()
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped")
