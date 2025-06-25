import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# офф варнинги
warnings.simplefilter(action='ignore', category=FutureWarning)

#подготовка данных
file_path = '/kaggle/input/car-price-prediction-dataset/cardekho.csv'
df = pd.read_csv(file_path)
df = pd.read_csv(file_path)
df = df.dropna()  # удаление пропусков
#формат вывода чисел
pd.set_option('display.float_format', '{:.2f}'.format)

#основная инфа
print("Предпросмотр данных:")
print(df.head())
print("\nИнформация о данных:")
df.info()
print("\nОсновные статистики:")
print(df.describe())
print("\nУникальные значения в столбце 'fuel':", df['fuel'].unique())
print("Уникальные значения в столбце 'seller_type':", df['seller_type'].unique())
print("Уникальные значения в столбце 'transmission':", df['transmission'].unique())
print("Уникальные значения в столбце 'owner':", df['owner'].unique())


#ГРАФИКИ
data = pd.read_csv(file_path)

data['brand'] = data['name'].apply(lambda x: x.split()[0].lower()) #нашел бренд
data_2015 = data[data['year'] == 2015] #выбираю год
data_2019 = data[data['year'] == 2019]
brand_counts_2015 = data_2015['brand'].value_counts()  #считаю число машин по году
brand_counts_2019 = data_2019['brand'].value_counts()

plt.figure(figsize=(16, 6))

# 2015
plt.subplot(1, 2, 1)
brand_counts_2015.plot(kind='bar', color='skyblue')
plt.title('Количество проданных машин по брендам (2015)')
plt.xlabel('Бренд')
plt.ylabel('Количество продаж')
plt.xticks(rotation=45)

# 2019 
plt.subplot(1, 2, 2)
brand_counts_2019.plot(kind='bar', color='lightgreen')
plt.title('Количество проданных машин по брендам (2019)')
plt.xlabel('Бренд')
plt.ylabel('Количество продаж')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#ДИАГРАММЫ
plt.figure(figsize=(16, 8))  

# скрыть <5%
def autopct_format(pct):
    return f'{pct:.1f}%' if pct >= 5 else ''

# 2015
plt.subplot(1, 2, 1)

# скрыть <5% внутри
labels_2015 = [label if (value / brand_counts_2015.sum() * 100) >= 5 else '' 
               for label, value in zip(brand_counts_2015.index, brand_counts_2015.values)]

brand_counts_2015.plot(
    kind='pie',
    autopct=autopct_format,
    startangle=90,
    pctdistance=0.85,
    textprops={'fontsize': 10},
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    labels=labels_2015 
)
plt.title('Доля продаж по брендам (2015)', pad=20)
plt.ylabel('')
plt.legend(
    labels=brand_counts_2015.index,  
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)
# 2019
plt.subplot(1, 2, 2)

labels_2019 = [label if (value / brand_counts_2019.sum() * 100) >= 5 else '' 
               for label, value in zip(brand_counts_2019.index, brand_counts_2019.values)]

brand_counts_2019.plot(
    kind='pie',
    autopct=autopct_format,
    startangle=90,
    pctdistance=0.85,
    textprops={'fontsize': 10},
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    labels=labels_2019
)
plt.title('Доля продаж по брендам (2019)', pad=20)
plt.ylabel('')
plt.legend(
    labels=brand_counts_2019.index,
    loc='center left',
    bbox_to_anchor=(1, 0.5),
    fontsize=10
)
plt.tight_layout()
plt.show()
#ГРАФИКИ




# графички МНОГА
# 6 штук
numeric_columns = ['selling_price', 'year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'seats']
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col])
    plt.title(f'Распределение {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')
plt.tight_layout()  
plt.show()

# 4 штуки
categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner']
plt.figure(figsize=(15, 8)) 
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(2, 2, i) 
    sns.countplot(data=df, x=col)
    plt.title(f'Количество объектов: {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# матрица корреляции
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Матрица корреляций")
plt.show()



# ПРЕДОБРАБОТКА
def convert_float(x):     #скип пропуксков
    try:
        return float(x)
    except:
        return np.nan

df["max_power"] = df["max_power"].apply(convert_float)
missing = ["mileage(km/ltr/kg)", "engine", "max_power", "seats"]
for i in missing:
    df[i].fillna(df[i].median(), inplace=True)                     #пропуск медианой заполняется


# подготовка для модели 
categoricals = ["fuel", "seller_type", "transmission", "owner"]
numericals = ["year", "mileage(km/ltr/kg)", "engine", "max_power", "seats", "km_driven", "selling_price"]   #распределение на категории и числа

# категориии ы число
le = LabelEncoder()
for i in categoricals:
    df[i] = le.fit_transform(df[i])

# масштабированеи для диапазона 0-1, шоб не сломалось
data = df[categoricals + numericals].values
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
# оси задать
x = data[:, :-1]
y = data[:, -1]

# выборка учение/тест - 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)


# ЛИНЕЙКА
linear_regression = LinearRegression()


#ЛЕС
# настриваю лес
param_grid = {
    'n_estimators': [50, 100],  # деревья
    'max_depth': [10, 20],      # глубина
    'min_samples_split': [2, 5], # мин для разделения
    'min_samples_leaf': [1, 2],  # мин в листе
    'max_features': [1.0, 'sqrt'] # признаки число
}

# поиск оптимальных параметров
# rf = RandomForestRegressor(random_state=0)

# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     scoring='r2',
#     cv=3,
#     n_jobs=-1,
#     verbose=0
# )

# grid_search.fit(x_train, y_train)      #пуск обучения
# best_rf = grid_search.best_estimator_  #вывод резов че лучш
# best_params = grid_search.best_params_

# print("\nЛучшие параметры для случайного леса:")
# print(best_params)
# print()


# сравнение регресии леса
models = [linear_regression, best_rf]
names = ["Линейная регрессия", "Оптимизированный случайный лес"]
predictions = []

# реп батл моделей
def training(model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    pred = np.maximum(pred, 0)  # предсказания ток +
    r2 = r2_score(y_test, pred)  
    mse = mean_squared_error(y_test, pred) 
    return r2, mse, pred

# оценка
for i, j in zip(models, names):
    print(j, "\n")
    r2, mse, pred = training(i)
    predictions.append(pred)
    print("Среднеквадратичная ошибка (MSE): ", mse)
    print("Коэффициент детерминации (R^2): ", r2)
    print("\n\n")


# графички
plt.figure(figsize=(18, 6))
for i, j in enumerate(names):
    plt.subplot(1, 2, i + 1)
    plt.scatter(y_test, predictions[i], alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r')
    plt.title(f'Предсказанные vs Фактические цены: {j}')
    plt.xlabel('Фактические цены')
    plt.ylabel('Предсказанные цены')
    plt.xlim(y.min(), y.max())
    plt.ylim(y.min(), y.max())
    plt.grid()
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.show()

#+линии регрессии
plt.figure(figsize=(18, 6))
for i, j in enumerate(names):
    plt.subplot(1, 2, i + 1)
    plt.scatter(predictions[i], y_test, alpha=0.5, label='Фактическая цена')
    m, b = np.polyfit(predictions[i], y_test, 1)
    plt.plot(predictions[i], m * predictions[i] + b, color='orange', label='Предсказанная цена')
    for x_val, y_val in zip(predictions[i], y_test):
        predicted_y = m * x_val + b
        plt.vlines(x_val, y_val, predicted_y, colors='gray', linestyles='dotted')
    plt.title(f'Предсказанные vs Фактические цены: {j}')
    plt.ylabel('Фактические цены')
    plt.xlabel('Предсказанные цены')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.subplots_adjust(wspace=0.5)
plt.show()
