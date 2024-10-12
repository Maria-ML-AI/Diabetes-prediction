import logging
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, CallbackContext, ConversationHandler, MessageHandler, Filters
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import os

# Настраиваем логирование
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']

# Стадії для запиту фічей
ASKING_FEATURES = 1
ASKING_AGE_BMI = 2

# Глобальный словарь для хранения данных пользователя
user_data = {}

# Загружаем модель XGBoost с обработкой ошибки
try:
    model = xgb.Booster()
    model.load_model('model_diabetes.pkl')
    logger.info("Модель XGBoost успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {str(e)}")

# Функция для вывода кнопок с вариантами ответов
def send_question(update: Update, context: CallbackContext, question: str, options: list):
    keyboard = [[InlineKeyboardButton(option, callback_data=option) for option in options]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text(question, reply_markup=reply_markup)

# Стартовая команда
def start(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    user_data[chat_id] = []
    logger.info(f"Пользователь {chat_id} начал сессию.")
    send_question(update, context, "Наявність гіпертонії (0 - немає, 1 - є):", ["0", "1"])
    return ASKING_FEATURES

# Обработка нажатий на кнопки
def button_handler(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()
    
    chat_id = query.message.chat_id
    current_feature = len(user_data[chat_id])
    
    # Сохраняем выбранное пользователем значение
    user_data[chat_id].append(float(query.data))
    logger.info(f"Пользователь {chat_id} ввел значение {query.data} для {FEATURES[current_feature]}.")

    # Переходим к следующему вопросу
    current_feature += 1
    if current_feature < len(FEATURES):
        feature_name = FEATURES[current_feature]
        logger.info(f"Запрашиваем у пользователя {chat_id} информацию для {feature_name}.")
        
        if feature_name == 'BMI':
            query.message.reply_text("Введи значення для ІМТ (вага в кг поділена на квадрат зросту в метрах):")
            return ASKING_AGE_BMI
        
        elif feature_name == 'Age':
            query.message.reply_text("Введи свій вік:")
            return ASKING_AGE_BMI
        
        # Для остальных вопросов предлагаем кнопки
        elif feature_name == 'HighChol':
            send_question(query, context, "Наявність високого холестерину (0 - немає, 1 - є):", ["0", "1"])
        
        elif feature_name == 'Smoker':
            send_question(query, context, "Чи курите ви? (0 - ні, 1 - так):", ["0", "1"])
        
        elif feature_name == 'Stroke':
            send_question(query, context, "Чи був у вас інсульт? (0 - ні, 1 - так):", ["0", "1"])
        
        # Добавьте аналогично для остальных вопросов...
        
        return ASKING_FEATURES
    else:
        return make_prediction(query, context)

# Обработка ввода числовых данных
def handle_numeric_input(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    value = update.message.text
    try:
        value = float(value)
        logger.info(f"Пользователь {chat_id} ввел значение {value} для {FEATURES[len(user_data[chat_id])]}") 
    except ValueError:
        update.message.reply_text("Будь ласка, введи числове значення.")
        return ASKING_AGE_BMI
    
    user_data[chat_id].append(value)
    
    # После ввода числового значения сразу переходим к следующему вопросу
    current_feature = len(user_data[chat_id])
    if current_feature < len(FEATURES):
        feature_name = FEATURES[current_feature]
        
        if feature_name == 'Age':
            update.message.reply_text("Введи свій вік:")
            return ASKING_AGE_BMI
        
        if feature_name == 'Smoker':
            send_question(update, context, "Чи курите ви? (0 - ні, 1 - так):", ["0", "1"])
            return ASKING_FEATURES

    else:
        return make_prediction(update, context)

def feature_engineering(df):
    logger.info("Выполняется feature engineering.")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['KMeans_Cluster'] = kmeans.fit_predict(df[['Age', 'Income']])

    df = pd.get_dummies(df, columns=['KMeans_Cluster'], drop_first=True)
    df['High_Income_Flag'] = np.where(df['Income'].isin([7, 8]), 1, 0)
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obesity'])
    df.drop(columns=['BMI'], inplace=True)

    df = pd.get_dummies(df, columns=['GenHlth', 'DiffWalk', 'BMI_Category'], drop_first=True)

    scaler = StandardScaler()
    df[['MentHlth', 'PhysHlth', 'Age', 'Income']] = scaler.fit_transform(df[['MentHlth', 'PhysHlth', 'Age', 'Income']])

    logger.info("Feature engineering завершен.")
    return df

# Предсказание результата
def make_prediction(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    user_features = user_data[chat_id]
    
    features_df = pd.DataFrame([user_features], columns=FEATURES)
    features_df = feature_engineering(features_df)
    
    prediction = model.predict(xgb.DMatrix(features_df))
    logger.info(f"Прогноз для пользователя {chat_id}: {prediction[0]}")
    
    update.message.reply_text(f"Ймовірність наявності діабету: {prediction[0]}")
    
    user_data.pop(chat_id, None)
    return ConversationHandler.END

# Обработка команды /cancel
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Операцію скасовано.")
    return ConversationHandler.END

def main():
       
    token = "7882573984:AAHJ7ZQBZGOTIDnTZnm40DrLdRMpTXE8YDU" 
    #os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("Токен не найден. Проверьте переменные окружения.")
        return
    
    updater = Updater(token, use_context=True)
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASKING_FEATURES: [CallbackQueryHandler(button_handler)],
            ASKING_AGE_BMI: [MessageHandler(Filters.text & ~Filters.command, handle_numeric_input)]
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    logger.info("Запуск бота.")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
