import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pulp

# ==============================================================================
# 【基础部分：你的原代码，已修正单位 + 你的专属数据 ZHOUY】
# ==============================================================================
# 1. 初始化线性规划问题
prob = pulp.LpProblem("The_Diet_Problem", pulp.LpMinimize)

# 2. 9种食物（决策变量：每种食物花多少钱，单位：美元）
foods = [
    "Wheat_Flour", "Evaporated_Milk", "Cheddar_Cheese", "Beef_Liver",
    "Cabbage", "Spinach", "Sweet_Potatoes", "Dried_Lima_Beans", "Product_X"
]
x = pulp.LpVariable.dicts("Spend", foods, lowBound=0, cat='Continuous')

# 3. 目标函数：最小化总花费
prob += pulp.lpSum([x[f] for f in foods]), "Total_Cost"

# 4. 你的专属营养需求（ZHOUY）
requirements = {
    "Calories": 8600,      # 10² 卡路里
    "Calcium": 0.3,       # 10⁻² 克
    "Vit_A": 1000,         # 10² IU
    "Riboflavin": 7.8,    # 10⁻¹ 毫克
    "Ascorbic_Acid": 0   # 毫克
}

# 5. 食物营养数据（统一单位：每1美元对应的营养值）
nutrition_data = {
    "Wheat_Flour":      [44700, 2,   0, 33.3,   0],
    "Evaporated_Milk":  [ 8400,15.1, 26000, 23.5,  60],
    "Cheddar_Cheese":   [ 7400,16.4, 28100, 10.3,   0],
    "Beef_Liver":       [ 2200,  0.2,169200, 50.8, 525],
    "Cabbage":          [ 2600, 4,  7200,  4.5,5369],
    "Spinach":          [ 1100,   0,918400, 13.8,2755],
    "Sweet_Potatoes":   [ 9600, 2.7,290700,  5.4,1912],
    "Dried_Lima_Beans": [17400, 3.7,  5100, 38.2,   0],
    "Product_X":        [ 8300,  6.9,  45000,  5.4,  39]
}

# 6. 添加营养约束
for i, nutrient in enumerate(requirements.keys()):
    prob += pulp.lpSum([x[f] * nutrition_data[f][i] for f in foods]) >= requirements[nutrient], f"Min_{nutrient}"

# ==============================================================================
# 【问题 a】最优饮食方案 & 最小成本
# ==============================================================================
prob.solve(pulp.PULP_CBC_CMD(msg=False))
print("===== (a) 最优饮食方案与最小成本 =====")
print(f"状态: {pulp.LpStatus[prob.status]}")
print(f"最小总成本: ${pulp.value(prob.objective):.4f}")
for f in foods:
    if x[f].varValue > 0.0001:
        print(f"- {f}: ${x[f].varValue:.4f}")

# ==============================================================================
# 【问题 b】维生素A & 核黄素 的最高支付意愿（影子价格）
# ==============================================================================
print("\n===== (b) 纯维生素A / 核黄素 的最高支付意愿 =====")
shadow_prices = {name: c.pi for name, c in prob.constraints.items()}
print(f"维生素A 影子价格(支付意愿): {shadow_prices['Min_Vit_A']:.6f}")
print(f"核黄素 影子价格(支付意愿): {shadow_prices['Min_Riboflavin']:.6f}")

# ==============================================================================
# 【问题 c】是否加入新食物 GLUNK
# 营养值：83,17,25,93,07（每美元）
# ==============================================================================
print("\n===== (c) 是否加入新食物 GLUNK =====")
prob_c = pulp.LpProblem("Prob_C", pulp.LpMinimize)
x_c = pulp.LpVariable.dicts("Spend", foods, lowBound=0)
x_glunk = pulp.LpVariable("Spend_GLUNK", lowBound=0)
# 重建目标函数：原食物 + GLUNK
prob_c += pulp.lpSum([x_c[f] for f in foods]) + x_glunk
# 添加营养约束（含GLUNK）
glunk_nut = [8300,1.7,25000,9.3,7]
for i, nut in enumerate(requirements.keys()):
    prob_c += pulp.lpSum([x_c[f] * nutrition_data[f][i] for f in foods]) + x_glunk * glunk_nut[i] >= requirements[nut], f"Min_{nut}"
prob_c.solve(pulp.PULP_CBC_CMD(msg=False))
cost_c = pulp.value(prob_c.objective)
cost_a = pulp.value(prob.objective)
print(f"原成本: {cost_a:.4f} | 加入GLUNK后成本: {cost_c:.4f}")
if cost_c < cost_a - 1e-6:
    print("=> 结论：应该加入 GLUNK")
else:
    print("=> 结论：不需要加入 GLUNK")

# ==============================================================================
# 【问题 d】利马豆成本变动多少会进入/离开方案（约简成本）
# ==============================================================================
print("\n===== (d) 利马豆成本变动分析 =====")
lima = x["Dried_Lima_Beans"]
reduced_cost_lima = lima.dj
print(f"利马豆 约简成本(Reduced Cost): {reduced_cost_lima:.4f}")
if reduced_cost_lima > 1e-6:
    print(f"=> 利马豆需要降价 {reduced_cost_lima:.4f} 美元才会被选用")
else:
    print("=> 利马豆已在最优方案中")

# ==============================================================================
# 【问题 e】牛肝成本的有效范围（在什么价格区间会被保留）
# 方法：在当前模型中，牛肝的目标系数为1（每美元花费成本为1）。
# 若牛肝价格变动k倍，则目标系数变为k（或等效地将营养值除以k）。
# 我们通过改变目标系数来寻找牛肝仍在方案中的范围。
# ==============================================================================
print("\n===== (e) 牛肝成本有效范围 =====")

def solve_with_liver_cost(cost_multiplier):
    """以给定的牛肝成本倍数求解LP，返回(总成本, 牛肝是否在解中, 牛肝用量)"""
    p = pulp.LpProblem("Sensitivity", pulp.LpMinimize)
    x_e = pulp.LpVariable.dicts("Spend", foods, lowBound=0)
    # 目标函数：牛肝成本 = cost_multiplier，其他食物成本 = 1
    p += (pulp.lpSum([x_e[f] for f in foods if f != "Beef_Liver"])
          + cost_multiplier * x_e["Beef_Liver"]), "Total_Cost"
    for i, nut in enumerate(requirements.keys()):
        p += pulp.lpSum([x_e[f] * nutrition_data[f][i] for f in foods]) >= requirements[nut], f"Min_{nut}"
    p.solve(pulp.PULP_CBC_CMD(msg=False))
    liver_used = x_e["Beef_Liver"].varValue or 0
    return pulp.value(p.objective), liver_used > 0.0001, liver_used

# 当前牛肝在方案中，系数=1
# 下限：成本降低 → 牛肝更有吸引力，不会离开。下限为0。
# 上限：成本升高 → 牛肝会被替代。用二分查找。
print("正在计算牛肝成本上限...")
# 先找到上界的大致范围
hi = 1.0
for _ in range(20):
    hi *= 2
    _, in_diet, _ = solve_with_liver_cost(hi)
    if not in_diet:
        break

# 二分查找精确上限
lo = 1.0
for _ in range(40):  # 40次迭代精度足够
    mid = (lo + hi) / 2
    _, in_diet, _ = solve_with_liver_cost(mid)
    if in_diet:
        lo = mid
    else:
        hi = mid

upper_bound = lo  # 牛肝仍保留的最大成本倍数

# 下限：检查成本为0时是否仍在方案中
_, zero_in, _ = solve_with_liver_cost(0)
lower_bound = 0.0  # 成本为0必然在方案中

print(f"牛肝当前成本系数: 1.0（基准）")
print(f"=> 牛肝成本有效范围: [{lower_bound:.4f}, {upper_bound:.4f}]")
print(f"=> 即只要牛肝的价格不超过当前价格的 {upper_bound:.4f} 倍，就会保留在饮食中")
print(f"=> 若价格超过 {upper_bound:.4f} 倍，牛肝将被其他食物替代")

# ==============================================================================
# 【问题 f】食物1,3,5,7,9涨价10%，但你继续使用(a)中的饮食方案
# 关键解释："continued on the diet found in (a)" 意味着你**不改变**食物选择
# 和数量，而不是重新优化。但我们也给出重新优化的结果作为对比。
# ==============================================================================
print("\n===== (f) 涨价10%后的维生素A/核黄素支付意愿 =====")

# 食物索引: 1=Wheat_Flour, 2=Evaporated_Milk, 3=Cheddar_Cheese, 4=Beef_Liver,
#            5=Cabbage, 6=Spinach, 7=Sweet_Potatoes, 8=Dried_Lima_Beans, 9=Product_X
# 涨价食物: 1,3,5,7,9
price_up_foods = ["Wheat_Flour", "Cheddar_Cheese", "Cabbage", "Sweet_Potatoes", "Product_X"]

# --- 解释1：坚持(a)方案，不重新优化 ---
print("--- 解释1：坚持(a)饮食方案不变 ---")
original_cost_a = pulp.value(prob.objective)
extra_cost = sum(x[f].varValue * 0.1 for f in price_up_foods if x[f].varValue > 0.0001)
new_total_cost = original_cost_a + extra_cost
print(f"原总成本: ${original_cost_a:.4f}")
print(f"因涨价增加的成本: ${extra_cost:.4f}")
print(f"新总成本: ${new_total_cost:.4f}")
print(f"维生素A 支付意愿: {shadow_prices['Min_Vit_A']:.6f} (与(a)相同，因方案未变)")
print(f"核黄素 支付意愿: {shadow_prices['Min_Riboflavin']:.6f} (与(a)相同，因方案未变)")

# --- 解释2：重新优化（供参考） ---
print("\n--- 解释2：允许重新优化饮食方案（供对比） ---")
prob_f = pulp.LpProblem("Prob_F", pulp.LpMinimize)
x_f = pulp.LpVariable.dicts("Spend", foods, lowBound=0)
price_multiplier = {
    "Wheat_Flour":1.1, "Cheddar_Cheese":1.1, "Cabbage":1.1,
    "Sweet_Potatoes":1.1, "Product_X":1.1
}
prob_f += pulp.lpSum([x_f[f] * price_multiplier.get(f, 1) for f in foods])
for i, nut in enumerate(requirements.keys()):
    prob_f += pulp.lpSum([x_f[f]*nutrition_data[f][i] for f in foods]) >= requirements[nut], f"Min_{nut}"
prob_f.solve(pulp.PULP_CBC_CMD(msg=False))
sp_f = {name:c.pi for name,c in prob_f.constraints.items()}
print(f"重新优化后总成本: ${pulp.value(prob_f.objective):.4f}")
print(f"新维生素A影子价格: {sp_f['Min_Vit_A']:.6f}")
print(f"新核黄素影子价格: {sp_f['Min_Riboflavin']:.6f}")
print("新方案食物选择:")
for f in foods:
    if x_f[f].varValue > 0.0001:
        print(f"  - {f}: ${x_f[f].varValue:.4f}")

# ==============================================================================
# 【问题 g】小麦粉维生素A +10 → 方案是否改变
# ==============================================================================
print("\n===== (g) 小麦粉维生素A+10，方案是否改变 =====")
prob_g = pulp.LpProblem("Prob_G", pulp.LpMinimize)
x_g = pulp.LpVariable.dicts("Spend", foods, lowBound=0)
prob_g += pulp.lpSum(x_g[f] for f in foods)
# 使用深拷贝避免修改原始数据
new_nut = {k: v.copy() for k, v in nutrition_data.items()}
new_nut["Wheat_Flour"][2] += 10  # 维A+10
for i, nut in enumerate(requirements.keys()):
    prob_g += pulp.lpSum([x_g[f]*new_nut[f][i] for f in foods]) >= requirements[nut], f"Min_{nut}"
prob_g.solve(pulp.PULP_CBC_CMD(msg=False))
cost_g = pulp.value(prob_g.objective)

# 比较方案
orig_foods = set(f for f in foods if x[f].varValue > 0.0001)
new_foods = set(f for f in foods if x_g[f].varValue > 0.0001)

print(f"原方案食物: {orig_foods}")
print(f"新方案食物: {new_foods}")
print(f"原成本: ${cost_a:.4f} | 修改后成本: ${cost_g:.4f}")

foods_changed = orig_foods != new_foods
cost_changed = abs(cost_g - cost_a) > 1e-4

if foods_changed or cost_changed:
    print("=> 结论：饮食方案 **会改变**")
    if foods_changed:
        print(f"   食物构成变化: {orig_foods} -> {new_foods}")
    if cost_changed:
        print(f"   成本变化: ${cost_a:.4f} -> ${cost_g:.4f}")
else:
    print("=> 结论：饮食方案 **不变**")
    print("   原因：小麦粉原本维生素A=0，维生素A约束不紧张（影子价格=0），")
    print("   增加10单位维生素A不改变任何紧约束")
