import os, time
import logging
import threading

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
from PIL import Image, ImageOps
from dataclasses import dataclass
from methods import gan, nst

PORT = int(os.environ.get('PORT', 5000))
MODE, PHOTO, NST, GAN, GAN_STYLE = range(5)

# Tokens is saved on the Heroku. HerokuApp -> Settings ->  Config Vars
TOKEN = os.environ.get('TG_TOKEN')
HEROKU_URL = os.environ.get('HEROKU_URL')

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
content_files = {} #dictionary of content files: userId -> file_name
listOfTasks = []

os.makedirs("outputs", exist_ok = True)
os.makedirs("inputs", exist_ok = True)

@dataclass
class StyleTask:
    method: str
    target: str
    style:  str
    update: Update

def help(update: Update, context: CallbackContext):
    
    update.message.reply_text(
        'ФотоБот умеет переносить менять стили фотографии при помощи нейросетей.\n'
        'Реализовано два метода: NST и GAN.\n'
        'Для начала работы используйте команду /start\n'
        'Для завершения работы - команду /cancel\n',
        reply_markup = ReplyKeyboardRemove())
    return

def start(update: Update, context: CallbackContext) -> int:

    update.message.reply_text(
        'Привет! Я ФотоБот. Я умею изменять стили фотографий.\n'
        'Для начала отправь мне фото, которое нужно изменить.', 
        reply_markup = ReplyKeyboardRemove())
    return PHOTO

def photo_content(update: Update, context: CallbackContext) -> int:
    
    photo_file = update.message.photo[-1].get_file()
    content_filename = f'inputs/{str(photo_file["file_unique_id"])}.jpg'
    photo_file.download(content_filename)

    content_files[update.message.chat_id] = content_filename #запоминаем файл привязанный к пользователю
   
    reply_keyboard = [['NST', 'GAN']]
    update.message.reply_text(
        'Итак, контентное фото у нас есть! '
        'Как будем переносить стиль? NST или GAN?',
        reply_markup = ReplyKeyboardMarkup(reply_keyboard))

    return MODE

def mode(update: Update, context: CallbackContext) -> int:
    
    if(update.message.text == 'NEW'):
        update.message.reply_text(
            'Ok! Отправь мне новое фото контента',
            reply_markup = ReplyKeyboardRemove())

        return PHOTO

    if(update.message.text == 'GAN'):
        styles_keyboard = [['CUPHEAD', 'STARRY NIGHT', 'MOSAIC']]

        update.message.reply_text(
            'Замечательно! Дальше нам нужно выбрать один из трех стилей: ',
            reply_markup = ReplyKeyboardMarkup(styles_keyboard))

        return GAN
    
    if(update.message.text == 'NST'):
        update.message.reply_text(
            'Супер! Тогда мне нужна еще картинка стиля...\n'
            'Отправь еще одно фото, c которого будем копировать стиль',
            reply_markup = ReplyKeyboardRemove(),
        )
        return NST

def run_style_transfer(update: Update, context: CallbackContext) -> int:

    photo_file = update.message.photo[-1].get_file()
    style_filename = f'inputs/{str(photo_file["file_unique_id"])}.jpg'
    photo_file.download(style_filename)

    update.message.reply_text(
        'Отличный стиль, думаю получится что-то интересное.\n'
        'Но NST не самый быстрый метод - придется немного подождать...')

    listOfTasks.append(StyleTask("NST", content_files[update.message.chat_id], style_filename, update))
    print('NST added task')
    return show_task_added(update, len(listOfTasks))

def run_GAN_transfer(update: Update, context: CallbackContext) -> int:

    update.message.reply_text(
        f'Принято в работу, применяю стиль {update.message.text}', 
        reply_markup = ReplyKeyboardRemove())

    listOfTasks.append(StyleTask("GAN", content_files[update.message.chat_id], update.message.text, update))
    print('GAN added task')
    return show_task_added(update, len(listOfTasks))

def send_result(update: Update, result_file, task_title):
    update.message.reply_text(f'Готова задачка {task_title}!')
    update.message.reply_photo(photo = open(result_file, "rb"))
    update.message.reply_text('Шикарно же?!\n')

def show_task_added(update, num) -> int:

    update.message.reply_text(f'Задача добавлена в очередь под номером {num}\n'
                               'Когда будет готово, я пришлю результат')

    modes_keyboard = [['NST', 'GAN'],['NEW']]
    update.message.reply_text(
        'Можем добавить еще задач! Что будем делать дальше?\n'
        'Продолжим "мучать" наше фото? (Выбери NST или GAN...)\n'
        'Или попробуем на другой фотке? (Нажимай NEW...) ',
        reply_markup = ReplyKeyboardMarkup(modes_keyboard))

    return MODE

def cancel(update: Update, context: CallbackContext) -> int:
    user = update.message.from_user
    logger.info("User %s canceled the conversation.", user.first_name)
    update.message.reply_text(
        'Пока! Будет скучно - приходи еще...', reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END

def run_tasks(queue):

    while True:
        if len(queue) > 0:
            print('task found')
            task = queue[0] #.pop(0)

            if task.method == "GAN":
                print('do gan')
                styled_file = gan.apply_style(task.target, task.style)
            else:
                print('do nst')
                styled_file = nst.transfer_style(task.target, task.style)

            queue.pop(0)
            send_result(task.update, styled_file, f'{task.method} ({task.style})')
        else:
            print('waiting for task')

        time.sleep(1 if len(queue) > 0 else 10)

def main() -> None:
    # Create the Updater and pass it your bot's token.
    updater = Updater(TOKEN)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # Add conversation handler with the states MODE, PHOTO
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start), CommandHandler('help', help)],
        states = {
            MODE: [MessageHandler(Filters.regex('^(NST|GAN|NEW)$'), mode)],
            PHOTO: [MessageHandler(Filters.photo, photo_content)],
            NST: [MessageHandler(Filters.photo, run_style_transfer)],
            GAN: [MessageHandler(Filters.regex('^(CUPHEAD|STARRY NIGHT|MOSAIC)$'), run_GAN_transfer)]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    
    dispatcher.add_handler(conv_handler)
    updater.start_webhook(listen="0.0.0.0",
                            port=int(PORT),
                            url_path=TOKEN)

    updater.bot.set_webhook(HEROKU_URL + TOKEN)
    updater.idle()

if __name__ == '__main__':
    threading.Thread(target=run_tasks, args = (listOfTasks,), daemon = True).start()
    main()