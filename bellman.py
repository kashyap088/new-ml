import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import poisson

# --- Generate Training Data ---
def generate_train_data(B=100):
    def rank(a, p):
        return np.argsort(np.hstack((a, p)))[:,0]
    
    our_price = 10 + np.random.uniform(0, 10, (B, 1))
    competitor_prices = 10 + np.random.uniform(0, 10, (B, 5))
    our_rank = np.reshape(rank(our_price, competitor_prices), (B, 1))
    X = np.hstack((our_price, competitor_prices, our_rank))
    
    Y = np.round(np.random.uniform(0, 1, our_rank.shape) * (1 - our_rank / 11)).ravel()
    
    return (X, Y)

# --- Train Sales Prediction Model ---
X, Y = generate_train_data()
def make_model(X, Y):
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    def predict(x):
        return np.maximum(0, regr.predict(x))
    return predict

sales_model = make_model(X, Y)

# --- Parameters ---
T = 20
MAX_AGE = 3
initial_inventory = [0, 0, 5]  # 5 fresh units
price_range = np.arange(10, 20, 0.1)
L = 0.01
delta = 0.99
Z = 1
competitor_prices = np.array([14, 15, 16, 17, 18])
computed_values = {}

# --- Helper Functions ---
def rank(a, p):
    _rank = p.shape[0]
    for i in range(p.shape[0]):
        if a < p[i]:
            _rank = _rank - 1
    return _rank

def sell_and_age(inventory, sales):
    inventory = list(inventory)
    sold = 0
    for i in range(len(inventory)):
        if inventory[i] > 0:
            sell = min(inventory[i], sales - sold)
            inventory[i] -= sell
            sold += sell
            if sold == sales:
                break
    waste = inventory[0]
    new_inventory = inventory[1:] + [0]
    return new_inventory, sold, waste

# --- Bellman Equation ---
def _V(price, t, inventory):
    x = np.hstack((price, competitor_prices, rank(price, competitor_prices))).reshape(1, -1)
    sales_prob = sales_model(x)[0]

    _sum = 0
    for i in range(int(poisson.ppf(0.9999, sales_prob)) + 1):
        pi = poisson.pmf(i, sales_prob)
        next_inventory, sold, waste = sell_and_age(inventory, i)
        _, V_future = V(t + 1, next_inventory)

        today_profit = sold * price
        waste_cost = waste * 0.5
        holding_cost = sum(next_inventory) * L

        total = pi * (today_profit - holding_cost - waste_cost + delta * V_future)
        _sum += total
    return _sum

def V(t, inventory):
    inventory_key = tuple(inventory)
    if (t, inventory_key) in computed_values:
        return computed_values[(t, inventory_key)]

    if t >= T or sum(inventory) == 0:
        computed_values[(t, inventory_key)] = (0, sum(inventory) * Z)
        return (0, sum(inventory) * Z)

    price_opt, V_opt = None, -float('inf')
    for price in price_range:
        v = _V(price, t, inventory)
        if v > V_opt:
            price_opt = price
            V_opt = v

    computed_values[(t, inventory_key)] = (price_opt, V_opt)
    return (price_opt, V_opt)

# --- Evaluate Policy ---
V(0, initial_inventory)

# --- Plot Pricing Over Time ---
price_policy = []
inventory_state = initial_inventory.copy()

for t in range(T + 1):
    key = (t, tuple(inventory_state))
    if key in computed_values:
        p, _ = computed_values[key]
        price_policy.append(p)
        # Simulate fixed 2-unit sales per step
        inventory_state, _, _ = sell_and_age(inventory_state, 2)
    else:
        price_policy.append(None)

plt.figure(figsize=(10, 4))
plt.plot(price_policy, label="Price over time")
plt.ylabel('Optimal Price')
plt.xlabel('Time')
plt.grid(True)
plt.legend()
plt.title("Optimal Pricing Policy with Perishability")
plt.tight_layout()
plt.show()




