import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Update
from openai import AsyncOpenAI
from fastapi import FastAPI, Request

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RENDER_URL = os.getenv("RENDER_URL")

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

MODEL_NAME = "llama-3.1-8b-instant"
# Максимальное количество сообщений в истории (будет удалять старые)
MAX_CONTEXT_MESSAGES = 10

SYSTEM_PROMPT = """
Ты — Павел... (ТВОЙ ДЛИННЫЙ ПРОМПТ БЕЗ ИЗМЕНЕНИЙ)
"""

# Хранилище контекстов
user_contexts = {}

def trim_context(context):
    """Обрезает контекст, оставляя только последние MAX_CONTEXT_MESSAGES сообщений."""
    # Не считаем системный промпт
    system_messages = [msg for msg in context if msg["role"] == "system"]
    other_messages = [msg for msg in context if msg["role"] != "system"]
    
    if len(other_messages) > MAX_CONTEXT_MESSAGES:
        # Оставляем системные сообщения и последние MAX_CONTEXT_MESSAGES других
        context = system_messages + other_messages[-MAX_CONTEXT_MESSAGES:]
    
    return context

@asynccontextmanager
async def lifespan(app: FastAPI):
    bot = Bot(token=TELEGRAM_TOKEN)
    dp = Dispatcher()

    @dp.message(Command("start"))
    async def start_cmd(message: types.Message):
        await message.answer("Привет! Я Павел. Спрашивай, отвечу 😎")

    @dp.message(Command("clear"))
    async def clear_cmd(message: types.Message):
        user_contexts.pop(message.from_user.id, None)
        await message.answer("История диалога очищена. Начинаю с чистого листа.")

    @dp.message()
    async def handle_message(message: types.Message):
        user_id = message.from_user.id
        user_text = message.text
        
        logger.info(f"Получено сообщение от {user_id}: {user_text[:50]}...")

        if user_id not in user_contexts:
            user_contexts[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

        context = user_contexts[user_id]
        context.append({"role": "user", "content": user_text})

        await bot.send_chat_action(chat_id=message.chat.id, action="typing")

        try:
            # Обрезаем контекст перед отправкой, чтобы не превысить лимиты
            context = trim_context(context)
            
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=context,
                temperature=0.9,
                max_tokens=500,
            )
            answer = response.choices[0].message.content
            context.append({"role": "assistant", "content": answer})
            user_contexts[user_id] = context
            await message.answer(answer)
            logger.info(f"Ответ отправлен пользователю {user_id}")

        except Exception as e:
            error_str = str(e)
            logger.error(f"Ошибка Groq для пользователя {user_id}: {error_str}")
            
            # Обработка ошибки лимита токенов
            if "token" in error_str.lower() or "rate_limit" in error_str.lower():
                # Очищаем контекст, оставляя только системный промпт
                user_contexts[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
                await message.answer("Слишком много сообщений. История диалога очищена для продолжения работы.")
                # Повторяем запрос без контекста
                try:
                    context = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}]
                    response = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=context,
                        temperature=0.9,
                        max_tokens=500,
                    )
                    answer = response.choices[0].message.content
                    user_contexts[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}, 
                                             {"role": "user", "content": user_text},
                                             {"role": "assistant", "content": answer}]
                    await message.answer(answer)
                except Exception as retry_error:
                    logger.error(f"Повторная ошибка: {retry_error}")
                    await message.answer("Что-то пошло не так. Попробуй позже или начни с команды /clear")
            else:
                await message.answer(f"Что-то пошло не так. Попробуй еще раз чуть позже. (Код: {type(e).__name__})")

    # Устанавливаем вебхук
    if RENDER_URL:
        await bot.set_webhook(f"{RENDER_URL}/webhook")
        logger.info("Вебхук установлен успешно")
    else:
        logger.warning("RENDER_URL не задан! Вебхук не установлен.")

    app.state.bot = bot
    app.state.dp = dp

    yield

    await bot.delete_webhook()
    await bot.session.close()

app = FastAPI(lifespan=lifespan)

@app.post("/webhook")
async def webhook(request: Request):
    bot = request.app.state.bot
    dp = request.app.state.dp
    data = await request.json()
    update = Update.model_validate(data)
    await dp.feed_update(bot, update)
    return "ok"

@app.get("/")
async def root():
    return {"status": "ok"}
