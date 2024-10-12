import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
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

# Завантажуємо модель XGBoost
try:
    model = xgb.Booster()
    model.load_model('model_diabetes.pkl')  # Загрузка модели напрямую через XGBoost
    logger.info("Модель XGBoost успешно загружена.")
except Exception as e:
    logger.error(f"Ошибка при загрузке модели: {str(e)}")
    model = joblib.load('model_diabetes.pkl')  # Загрузка через joblib для других моделей

# Стадії для запиту фічей
ASKING_FEATURES = 1

# Глобальный словарь для хранения данных пользователя
user_data = {}

# Стартова команда
def start(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    logger.info(f"Пользователь {chat_id} начал сессию.")
    user_data[chat_id] = []  # Инициализируем пустой список для пользователя
    update.message.reply_text("Привіт! Я допоможу передбачити наявність діабету. Дай відповідь на кілька питань.")
    return ask_feature(update, context)

def ask_feature(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    current_feature = len(user_data[chat_id])
    
    if current_feature < len(FEATURES):
        feature_name = FEATURES[current_feature]
        logger.info(f"Запрашиваем у пользователя {chat_id} информацию для {feature_name}.")
        
        if feature_name == 'BMI':
            update.message.reply_text("Введи значення для ІМТ(вага в кг поділена на квадрат зросту в метрах):")
        
        elif feature_name == 'Age':
            update.message.reply_text("Введи свій вік:")
        
        elif feature_name == 'HighBP':
            update.message.reply_text("Наявність гіпертонії (0 - немає, 1 - є):")
        
        # Добавьте остальные вопросы сюда...
        
        return ASKING_FEATURES
    else:
        return make_prediction(update, context)

# Збір даних від користувача
def collect_data(update: Update, context: CallbackContext) -> int:
    chat_id = update.message.chat_id
    value = update.message.text
    try:
        value = float(value)
        logger.info(f"Пользователь {chat_id} ввел значение {value} для {FEATURES[len(user_data[chat_id])]}") 
    except ValueError:
        update.message.reply_text("Будь ласка, введи числове значення.")
        logger.warning(f"Пользователь {chat_id} ввел некорректное значение: {value}.")
        return ASKING_FEATURES
    
    user_data[chat_id].append(value)
    return ask_feature(update, context)

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

# Обробка помилок
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
            ASKING_FEATURES: [MessageHandler(Filters.text & ~Filters.command, collect_data)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    logger.info("Запуск бота.")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()

