import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Фиксация случайных значений
np.random.seed(42)
torch.manual_seed(42)

# =============================================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ (CROP WATER MODEL)
# =============================================================================

class CropWaterModel:
      
    def __init__(self, crop_type='wheat', season_days=120):
        self.crop_type = crop_type
        self.season_days = season_days
        
        # Параметры культуры
        self.params = {
            'wheat': {
                'base_yield': 50,  # ц/га без стресса
                'water_requirement': 500,  # мм за сезон
                'critical_stages': [30, 60, 90],  # Дни критических фаз
                'drought_sensitivity': 0.8
            },
            'corn': {
                'base_yield': 80,
                'water_requirement': 600,
                'critical_stages': [40, 70, 100],
                'drought_sensitivity': 0.9
            },
            'rice': {
                'base_yield': 60,
                'water_requirement': 800,
                'critical_stages': [20, 50, 80],
                'drought_sensitivity': 0.7
            }
        }
        
    def simulate_season(self, irrigation_schedule, weather_data):
        """
        Симулирует урожайность при заданном графике полива.
        
        Args:
            irrigation_schedule: массив объёмов полива по дням (мм)
            weather_data: dict с данными (temp, precip, evapotranspiration)
        
        Returns:
            yield: урожайность (ц/га)
            water_used: общий объём использованной воды (мм)
            stress_days: количество дней водного стресса
        """
        params = self.params[self.crop_type]
        daily_water_need = params['water_requirement'] / self.season_days
        
        soil_moisture = 50  # Начальная влажность почвы (%)
        water_stress = 0
        total_irrigation = 0
        stress_days = 0
        
        for day in range(self.season_days):
            # Испарение
            et = weather_data['et'][day] if 'et' in weather_data else 5
            
            # Осадки
            precip = weather_data['precip'][day] if 'precip' in weather_data else 0
            
            # Полив
            irrigation = irrigation_schedule[day] if day < len(irrigation_schedule) else 0
            total_irrigation += irrigation
            
            # Баланс влаги
            soil_moisture = soil_moisture + precip + irrigation - et
            soil_moisture = np.clip(soil_moisture, 0, 100)
            
            # Водный стресс
            if soil_moisture < 30:
                stress = (30 - soil_moisture) / 30 * params['drought_sensitivity']
                water_stress += stress
                stress_days += 1
            else:
                stress = 0
            
            # Критические фазы более чувствительны
            for critical_day in params['critical_stages']:
                if abs(day - critical_day) <= 5 and stress > 0:
                    water_stress += stress * 0.5
        
        # Расчёт урожайности
        stress_factor = max(0, 1 - water_stress / self.season_days)
        yield_value = params['base_yield'] * stress_factor
        
        # Перелив тоже вреден
        if total_irrigation > params['water_requirement'] * 1.3:
            overwatering_penalty = (total_irrigation - params['water_requirement'] * 1.3) / 100
            yield_value = yield_value * max(0.8, 1 - overwatering_penalty)
        
        return {
            'yield': yield_value,
            'water_used': total_irrigation,
            'stress_days': stress_days,
            'water_efficiency': yield_value / max(total_irrigation, 1)
        }
    
    def generate_optimal_schedule(self, weather_data):
        """Генерирует оптимальный график полива (oracle)"""
        params = self.params[self.crop_type]
        schedule = np.zeros(self.season_days)
        
        soil_moisture = 50
        
        for day in range(self.season_days):
            et = weather_data['et'][day] if 'et' in weather_data else 5
            precip = weather_data['precip'][day] if 'precip' in weather_data else 0
            
            soil_moisture = soil_moisture + precip - et
            
            # Полив при низкой влажности
            if soil_moisture < 35:
                irrigation = min(20, 50 - soil_moisture)
                schedule[day] = irrigation
                soil_moisture += irrigation
            
            soil_moisture = np.clip(soil_moisture, 0, 100)
        
        return schedule


def generate_weather_data(season_days=120, n_seasons=100):
    """Генерирует погодные данные для多个 сезонов"""
    data = []
    
    for season in range(n_seasons):
        season_data = {
            'season_id': season,
            'temp': np.random.normal(25, 5, season_days),
            'precip': np.random.exponential(3, season_days),
            'et': np.random.normal(5, 1.5, season_days)  # Evapotranspiration
        }
        data.append(season_data)
    
    return data


def generate_training_dataset(n_samples=1000, season_days=120):
    """
    Генерирует датасет для обучения моделей. Локальный датасет развёрнут на домашнем ПК.
    Включает различные стратегии полива.
    """
    crop_model = CropWaterModel(crop_type='wheat', season_days=season_days)
    weather_data_list = generate_weather_data(season_days, n_samples)
    
    X = []  # Признаки
    y_irrigation = []  # Оптимальный полив (для регрессии)
    y_yield = []  # Урожайность
    metadata = []
    
    for i, weather in enumerate(weather_data_list):
        # Оптимальный график (целевая переменная для RL)
        optimal_schedule = crop_model.generate_optimal_schedule(weather)
        optimal_result = crop_model.simulate_season(optimal_schedule, weather)
        
        # Разные стратегии для обучения
        strategies = [
            ('optimal', optimal_schedule),
            ('minimal', np.ones(season_days) * 2),
            ('maximal', np.ones(season_days) * 15),
            ('random', np.random.uniform(0, 20, season_days)),
            ('fixed_interval', create_fixed_interval_schedule(season_days, 10))
        ]
        
        for strategy_name, schedule in strategies:
            result = crop_model.simulate_season(schedule, weather)
            
            # Признаки: текущий день, погода, влажность почвы
            for day in range(0, season_days, 5):  # Каждый 5 день
                features = [
                    day / season_days,  # Нормализованный день
                    weather['temp'][day] / 35,  # Нормализованная температура
                    weather['precip'][day] / 20,  # Нормализованные осадки
                    weather['et'][day] / 10,  # Нормализованное испарение
                    np.mean(schedule[max(0, day-10):day+1]) / 20,  # История полива
                    season / n_samples  # Номер сезона
                ]
                
                X.append(features)
                y_irrigation.append(optimal_schedule[day] / 20)  # Нормализованный полив
                y_yield.append(result['yield'] / 80)  # Нормализованная урожайность
                
                metadata.append({
                    'season': season,
                    'day': day,
                    'strategy': strategy_name,
                    'actual_yield': result['yield'],
                    'water_used': result['water_used']
                })
    
    return np.array(X), np.array(y_irrigation), np.array(y_yield), metadata, crop_model


def create_fixed_interval_schedule(season_days, interval):
    """Создаёт график полива с фиксированным интервалом"""
    schedule = np.zeros(season_days)
    for day in range(0, season_days, interval):
        schedule[day] = 10
    return schedule


print("Генерация данных для обучения...")
X, y_irrigation, y_yield, metadata, crop_model = generate_training_dataset(n_samples=200, season_days=120)
print(f"Размер датасета: {X.shape}")
print(f"Признаков: {X.shape[1]}")

# =============================================================================
# 2. НЕЙРОСЕТЕВАЯ РЕГРЕССИЯ (MLP MODEL)
# =============================================================================

class IrrigationMLP(nn.Module):
    """
    Многослойный перцептрон для предсказания оптимального объёма полива.
    """
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16], output_dim=1):
        super(IrrigationMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # Выход от 0 до 1
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class IrrigationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Подготовка данных
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y_irrigation.reshape(-1, 1)).flatten()

# Разделение на train/test
train_size = int(0.8 * len(X_scaled))
indices = np.random.permutation(len(X_scaled))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

train_dataset = IrrigationDataset(X_train, y_train)
test_dataset = IrrigationDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Обучение модели
MLP_MODEL = IrrigationMLP(input_dim=X.shape[1], hidden_dims=[64, 32, 16])
criterion = nn.MSELoss()
optimizer = optim.Adam(MLP_MODEL.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

print("\n" + "="*60)
print("ОБУЧЕНИЕ НЕЙРОСЕТЕВОЙ МОДЕЛИ (MLP)")
print("="*60)

n_epochs = 100
train_losses = []
test_losses = []

for epoch in range(n_epochs):
    MLP_MODEL.train()
    epoch_loss = 0
    
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = MLP_MODEL(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # Валидация
    MLP_MODEL.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = MLP_MODEL(X_batch)
            test_loss += criterion(outputs, y_batch).item()
    
    train_losses.append(epoch_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    scheduler.step(test_loss / len(test_loader))
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{n_epochs}], Train Loss: {epoch_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}')

# График обучения
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses, label='Test Loss', color='red')
plt.title('Обучение MLP модели')
plt.xlabel('Эпоха')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('mlp_training.png', dpi=150)
plt.show()

# =============================================================================
# 3. REINFORCEMENT LEARNING (PPO-LIKE)
# =============================================================================

class IrrigationEnv:
    """
    Среда для RL: управление поливом.
    """
    
    def __init__(self, crop_model, weather_data, season_days=120):
        self.crop_model = crop_model
        self.weather_data = weather_data
        self.season_days = season_days
        self.reset()
    
    def reset(self):
        self.current_day = 0
        self.soil_moisture = 50
        self.total_irrigation = 0
        self.cumulative_yield = 0
        return self._get_state()
    
    def _get_state(self):
        """Возвращает текущее состояние среды"""
        state = [
            self.current_day / self.season_days,
            self.soil_moisture / 100,
            self.weather_data['temp'][self.current_day] / 35,
            self.weather_data['precip'][self.current_day] / 20,
            self.weather_data['et'][self.current_day] / 10,
            self.total_irrigation / (self.crop_model.params['wheat']['water_requirement'])
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """
        Выполняет действие (полив) и возвращает новое состояние и награду.
        
        action: объём полива (0-1, нормализованный)
        """
        irrigation = action * 20  # 0-20 мм
        
        # Обновление состояния
        et = self.weather_data['et'][self.current_day]
        precip = self.weather_data['precip'][self.current_day]
        
        self.soil_moisture = self.soil_moisture + precip + irrigation - et
        self.soil_moisture = np.clip(self.soil_moisture, 0, 100)
        self.total_irrigation += irrigation
        
        # Расчёт награды
        # 1. Награда за поддержание оптимальной влажности
        moisture_reward = 1 - abs(self.soil_moisture - 50) / 50
        
        # 2. Штраф за использование воды
        water_penalty = irrigation / 20 * 0.3
        
        # 3. Штраф за стресс
        stress_penalty = 0
        if self.soil_moisture < 30:
            stress_penalty = (30 - self.soil_moisture) / 30 * 0.5
        
        reward = moisture_reward - water_penalty - stress_penalty
        
        # Переход к следующему дню
        self.current_day += 1
        done = self.current_day >= self.season_days
        
        # Финальная награда за урожайность
        if done:
            result = self.crop_model.simulate_season(
                np.zeros(self.season_days),  # Уже учтено в процессе
                self.weather_data
            )
            yield_reward = result['yield'] / self.crop_model.params['wheat']['base_yield']
            reward += yield_reward * 2  # Усиленная финальная награда
        
        return self._get_state(), reward, done, {}


class PPOAgent:
    """
    Упрощённая реализация PPO (Proximal Policy Optimization).
    """
    
    def __init__(self, state_dim, action_dim=1, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor Network (политика)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Действие от 0 до 1
        )
        
        # Critic Network (ценность состояния)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        self.gamma = 0.99
        self.clip_epsilon = 0.2
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            action_mean = self.actor(state)
            if deterministic:
                return action_mean.numpy()
            else:
                # Добавляем шум для исследования
                noise = torch.normal(0, 0.1, action_mean.shape)
                action = torch.clamp(action_mean + noise, 0, 1)
                return action.numpy()
    
    def evaluate(self, state):
        state = torch.FloatTensor(state)
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Расчёт целевых значений
        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + self.gamma * next_values * (1 - dones)
        
        # Обновление Critic
        values = self.critic(states)
        critic_loss = nn.MSELoss()(values, targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Обновление Actor (Policy Gradient)
        advantages = targets - values.detach()
        
        action_means = self.actor(states)
        log_probs = -((actions - action_means) ** 2) / (2 * 0.1 ** 2)  # Gaussian log prob
        
        actor_loss = -(log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()


# Обучение RL агента
print("\n" + "="*60)
print("ОБУЧЕНИЕ RL АГЕНТА (PPO)")
print("="*60)

rl_agent = PPOAgent(state_dim=6, action_dim=1, hidden_dim=64)

# Генерация погоды для RL
rl_weather = generate_weather_data(season_days=120, n_seasons=1)[0]
env = IrrigationEnv(crop_model, rl_weather, season_days=120)

n_episodes = 500
episode_rewards = []
actor_losses = []
critic_losses = []

for episode in range(n_episodes):
    state = env.reset()
    episode_reward = 0
    states, actions, rewards, next_states, dones = [], [], [], [], []
    
    for t in range(120):
        action = rl_agent.select_action(state)
        next_state, reward, done, _ = env.step(action[0][0])
        
        states.append(state)
        actions.append(action[0])
        rewards.append(reward)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # Обновление модели
    if len(states) > 0:
        actor_loss, critic_loss = rl_agent.update(
            states, actions, rewards, next_states, dones
        )
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
    
    episode_rewards.append(episode_reward)
    
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        print(f'Episode [{episode+1}/{n_episodes}], Avg Reward: {avg_reward:.2f}')

# График наград RL
plt.figure(figsize=(10, 5))
plt.plot(np.convolve(episode_rewards, np.ones(50)/50, mode='valid'), 
         label='Скользящее среднее (50)', color='green')
plt.title('Награды RL агента по эпизодам')
plt.xlabel('Эпизод')
plt.ylabel('Суммарная награда')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('rl_rewards.png', dpi=150)
plt.show()

# =============================================================================
# 4. СРАВНЕНИЕ СТРАТЕГИЙ ПОЛИВА
# =============================================================================

print("\n" + "="*60)
print("СРАВНЕНИЕ СТРАТЕГИЙ ПОЛИВА")
print("="*60)

def evaluate_strategy(strategy_name, schedule, weather, crop_model):
    """Оценивает стратегию полива"""
    result = crop_model.simulate_season(schedule, weather)
    return {
        'strategy': strategy_name,
        'yield': result['yield'],
        'water_used': result['water_used'],
        'water_efficiency': result['water_efficiency'],
        'stress_days': result['stress_days']
    }


# Тестовая погода
test_weather = generate_weather_data(season_days=120, n_seasons=1)[0]

# Различные стратегии
strategies = {
    'Без полива': np.zeros(120),
    'Минимальный': np.ones(120) * 2,
    'Максимальный': np.ones(120) * 15,
    'Фиксированный (10 дней)': create_fixed_interval_schedule(120, 10),
    'Фиксированный (5 дней)': create_fixed_interval_schedule(120, 5),
    'Оптимальный (Oracle)': crop_model.generate_optimal_schedule(test_weather)
}

# Прогноз MLP модели
mlp_schedule = np.zeros(120)
for day in range(0, 120, 5):
    features = [
        day / 120,
        test_weather['temp'][day] / 35,
        test_weather['precip'][day] / 20,
        test_weather['et'][day] / 10,
        np.mean(mlp_schedule[max(0, day-10):day+1]) / 20,
        0.5
    ]
    features = scaler_X.transform([features])[0]
    with torch.no_grad():
        pred = MLP_MODEL(torch.FloatTensor(features).unsqueeze(0)).numpy()[0][0]
    mlp_schedule[day] = pred * 20

strategies['MLP Модель'] = mlp_schedule

# Прогноз RL агента
rl_schedule = np.zeros(120)
state = env.reset()
env.weather_data = test_weather
for day in range(120):
    action = rl_agent.select_action(state, deterministic=True)
    rl_schedule[day] = action[0][0] * 20
    if day < 119:
        state, _, _, _ = env.step(action[0][0])

strategies['RL Агент (PPO)'] = rl_schedule

# Оценка всех стратегий
results = []
for name, schedule in strategies.items():
    result = evaluate_strategy(name, schedule, test_weather, crop_model)
    results.append(result)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('water_efficiency', ascending=False)

print(results_df.to_string(index=False))

# =============================================================================
# 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

# 1. График политик полива
plt.figure(figsize=(14, 7))
for i, (name, schedule) in enumerate(strategies.items()):
    if name in ['Оптимальный (Oracle)', 'MLP Модель', 'RL Агент (PPO)']:
        plt.plot(schedule, label=name, linewidth=2)
plt.xlabel('День сезона')
plt.ylabel('Объём полива (мм)')
plt.title('Сравнение политик полива')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('irrigation_policies.png', dpi=150)
plt.show()

# 2. Урожайность vs Вода
plt.figure(figsize=(10, 6))
plt.scatter(results_df['water_used'], results_df['yield'], 
           s=200, c=range(len(results_df)), cmap='viridis', alpha=0.6)
for i, row in results_df.iterrows():
    plt.annotate(row['strategy'], (row['water_used'], row['yield']), 
                fontsize=9, ha='center')
plt.xlabel('Использованная вода (мм)')
plt.ylabel('Урожайность (ц/га)')
plt.title('Эффективность стратегий: Урожайность vs Водопотребление')
plt.grid(True, alpha=0.3)
plt.savefig('yield_vs_water.png', dpi=150)
plt.show()

# 3. Эффективность по стратегиям
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].bar(x, results_df['yield'], width, label='Урожайность', color='green', alpha=0.7)
axes[0].set_ylabel('Урожайность (ц/га)')
axes[0].set_title('Урожайность по стратегиям')
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['strategy'], rotation=45, ha='right')
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].bar(x, results_df['water_used'], width, label='Вода', color='blue', alpha=0.7)
axes[1].set_ylabel('Вода (мм)')
axes[1].set_title('Водопотребление по стратегиям')
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['strategy'], rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('strategy_comparison.png', dpi=150)
plt.show()

# 4. Экономия воды
baseline_water = results_df[results_df['strategy'] == 'Фиксированный (10 дней)']['water_used'].values[0]
baseline_yield = results_df[results_df['strategy'] == 'Фиксированный (10 дней)']['yield'].values[0]

print("\n" + "="*60)
print("ЭКОНОМИЯ ВОДЫ")
print("="*60)

for _, row in results_df.iterrows():
    if row['strategy'] not in ['Фиксированный (10 дней)', 'Без полива']:
        water_saving = (baseline_water - row['water_used']) / baseline_water * 100
        yield_change = (row['yield'] - baseline_yield) / baseline_yield * 100
        print(f"{row['strategy']}:")
        print(f"  Экономия воды: {water_saving:.1f}%")
        print(f"  Изменение урожайности: {yield_change:+.1f}%")
        print()

# =============================================================================
# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

results_df.to_csv('irrigation_results.csv', index=False, encoding='utf-8-sig')
print("Результаты сохранены в irrigation_results.csv")

# Сохранение моделей
torch.save(MLP_MODEL.state_dict(), 'mlp_irrigation_model.pth')
torch.save(rl_agent.actor.state_dict(), 'rl_actor_model.pth')
print("Модели сохранены")

# Итоговая таблица
summary_table = {
    'Метрика': ['Урожайность (ц/га)', 'Вода (мм)', 'Эффективность', 'Стресс-дни'],
    'MLP': [
        f"{results_df[results_df['strategy']=='MLP Модель']['yield'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='MLP Модель']['water_used'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='MLP Модель']['water_efficiency'].values[0]:.3f}",
        f"{results_df[results_df['strategy']=='MLP Модель']['stress_days'].values[0]:.0f}"
    ],
    'RL (PPO)': [
        f"{results_df[results_df['strategy']=='RL Агент (PPO)']['yield'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='RL Агент (PPO)']['water_used'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='RL Агент (PPO)']['water_efficiency'].values[0]:.3f}",
        f"{results_df[results_df['strategy']=='RL Агент (PPO)']['stress_days'].values[0]:.0f}"
    ],
    'Оптимальный': [
        f"{results_df[results_df['strategy']=='Оптимальный (Oracle)']['yield'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='Оптимальный (Oracle)']['water_used'].values[0]:.2f}",
        f"{results_df[results_df['strategy']=='Оптимальный (Oracle)']['water_efficiency'].values[0]:.3f}",
        f"{results_df[results_df['strategy']=='Оптимальный (Oracle)']['stress_days'].values[0]:.0f}"
    ]
}

summary_df = pd.DataFrame(summary_table)
print("\n" + "="*60)
print("ИТОГОВАЯ ТАБЛИЦА")
print("="*60)
print(summary_df.to_string(index=False))

summary_df.to_csv('irrigation_summary.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print("ПРОЕКТ ЗАВЕРШЁН")
print("="*60)
