import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
import joblib  # для загрузки модели
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Налаштування для 20 фічей, які ти будеш запитувати у користувача
FEATURES = ['HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'GenHlth',
       'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education',
       'Income']  

# Завантажуємо модель
model = joblib.load('model_diabetes.pkl') 

# Логи для відстеження
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Стадії для запиту фічей
ASKING_FEATURES = range(len(FEATURES))

# Стартова команда
def start(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Привіт! Я допоможу передбачити наявність діабету. Дай відповідь на кілька питань.")
    user_data[update.message.chat_id] = []
    return ASKING_FEATURES
def ask_feature(update: Update, context: CallbackContext) -> int:
    current_feature = len(user_data[update.message.chat_id])
    
    if current_feature < len(FEATURES):
        feature_name = FEATURES[current_feature]
        
        if feature_name == 'BMI':
            update.message.reply_text("Введи значення для ІМТ(вага в кг поділена на квадрат зросту в метрах):")
        
        elif feature_name == 'Age':
            update.message.reply_text("Введи свій вік:")
        
        elif feature_name == 'HighBP':
            update.message.reply_text("Наявність гіпертонії (0 - немає, 1 - є:")
        
        elif feature_name == 'CholCheck':
            update.message.reply_text("Чи робився аналіз крові на холестерин за останні 5 років(0 - ні, 1 - так):")
            
        elif feature_name == 'Smoker':
            update.message.reply_text("Чи куриш(0 - ні, 1 - так):")
        
        elif feature_name == 'Stroke':
            update.message.reply_text("Чи був інсульт (0 - ні, 1 - так):")
        
        elif feature_name == 'HeartDiseaseorAttack':
            update.message.reply_text("Чи був серцевий напад або серцеве захворювання (0 - ні, 1 - так):")
        
        elif feature_name == 'PhysActivity':
            update.message.reply_text("Фізична активність протягом останнього місяця (0 - не було, 1 - була):")

        elif feature_name == 'Fruits':
            update.message.reply_text("Споживання фруктів принаймні раз на день (0 - ні, 1 - так):")
        
        elif feature_name == 'Veggies':
            update.message.reply_text("Споживання овочів принаймні раз на день (0 - ні, 1 - так):")
        
        elif feature_name == 'HvyAlcoholConsump':
            update.message.reply_text("Надмірне споживання алкоголю (0 - ні, 1 - так):")
        
        elif feature_name == 'AnyHealthcare':
            update.message.reply_text("Наявність медичного страхування (0 - ні, 1 - так):")
        
        elif feature_name == 'NoDocbcCost':
            update.message.reply_text("Чи пропускав візит до лікаря/не звертався до лікаря через високу вартість медичних послуг (0 - ні, 1 - так):")
        
        elif feature_name == 'GenHlth':
            update.message.reply_text("Оцінка загального здоров'я на основі шкали від 1 (відмінне) до 5 (погане):")
        
        elif feature_name == 'MentHlth':
            update.message.reply_text("Кількість днів за останній місяць, коли психічне здоров'я було поганим (числовий показник).:")
        
        elif feature_name == 'PhysHlth':
            update.message.reply_text("Кількість днів за останній місяць, коли фізичне здоров'я було поганим (числовий показник):")
        
        elif feature_name == 'DiffWalk':
            update.message.reply_text("Чи є труднощі з ходьбою (0 - ні, 1 - так):")
        
        elif feature_name == 'Sex':
            update.message.reply_text("Стать (0 - жіноча, 1 - чоловіча):")
        
        elif feature_name == 'Education':
            update.message.reply_text("Рівень освіти (шкала від 1 до 6, де 1 – відсутність освіти, 6 – вища освіта):")
        
        elif feature_name == 'Income':
            update.message.reply_text("Введи рівень доходу від 1 до 8:")
        

        return ASKING_FEATURES
    else:
        return make_prediction(update, context)


# Збір даних від користувача
def collect_data(update: Update, context: CallbackContext) -> int:
    value = update.message.text
    try:
        value = float(value)
    except ValueError:
        update.message.reply_text("Будь ласка, введи числове значення.")
        return ASKING_FEATURES
    
    user_data[update.message.chat_id].append(value)
    return ask_feature(update, context)

def feature_engineering(df):
    # Використовуємо KMeans для створення нових кластерів
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['KMeans_Cluster'] = kmeans.fit_predict(df[['Age', 'BMI', 'Income']])

    # Перетворення кластерів на категорійні фічі
    df = pd.get_dummies(df, columns=['KMeans_Cluster'], drop_first=True)

    # Створюємо нову ознаку для високого доходу
    df['High_Income_Flag'] = np.where(df['Income'].isin([7, 8]), 1, 0)

    # Створюємо нову ознаку: взаємодія віку та BMI
    df['Age_BMI_Interaction'] = df['Age'] * df['BMI']

    # Створюємо категорію BMI
    df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obesity'])
    df.drop(columns=['BMI'], inplace=True)

    # Перетворення категорійних ознак у dummy variables
    df = pd.get_dummies(df, columns=['GenHlth', 'DiffWalk', 'BMI_Category'], drop_first=True)

    # Масштабування числових ознак
    scaler = StandardScaler()
    df[['MentHlth', 'PhysHlth', 'Age', 'Income']] = scaler.fit_transform(df[['MentHlth', 'PhysHlth', 'Age', 'Income']])

    return df

# Передбачення результату
def make_prediction(update: Update, context: CallbackContext) -> int:
    user_features = user_data[update.message.chat_id]
    # Преобразуємо дані як необхідно для твоєї моделі (включаючи feature engineering)
    features_df = pd.DataFrame([user_features], columns=FEATURES)
        
    # Виклик feature engineering
    features_df = feature_engineering(features_df)
    
    # Передбачаємо результат
    prediction = model.predict(features_df)
    update.message.reply_text(f"Ймовірність наявності діабету: {prediction[0]}")
    
    # Очищуємо дані користувача після передбачення
    user_data.pop(update.message.chat_id, None)
    return ConversationHandler.END

# Обробка помилок
def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Операцію скасовано.")
    return ConversationHandler.END

def main():
    updater = Updater("YOUR_BOT_API_TOKEN", use_context=True)  # Замініть на токен вашого бота
    dispatcher = updater.dispatcher

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            ASKING_FEATURES: [MessageHandler(Filters.text & ~Filters.command, collect_data)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    dispatcher.add_handler(conv_handler)

    # Запуск бота
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
