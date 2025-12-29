from deap import creator, base, tools, algorithms
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

# 加载数据并预处理
data = pd.read_excel('/Users/macaroni/Downloads/pythonProject1/doi_txt/L_arc-melting_HV_Element.xlsx')
X = data.drop('HV', axis=1)
y = data['HV']

model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X, y)

# 定义遗传算法
# 创建适应度类和个体类
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 初始化工具箱
toolbox = base.Toolbox()


def create_individual():
    individual = np.zeros(13)  # 创建一个长度为13的个体，所有值初始化为0
    remaining = 100  # 初始剩余可分配总量为100

    indices = random.sample(range(13), 5)  # 随机选择5个位置

    for idx in indices:
        individual[idx] = random.uniform(4, min(remaining, 35))  # 在剩余的可分配总量和40之间分配一个随机值
        remaining -= individual[idx]  # 更新剩余可分配总量

    return individual

# 定义如何进行个体的变异操作
# 定义如何进行个体的变异操作
def mutate(individual, indpb):
    if random.random() < indpb:
        # 除了Mo以外的非零元素索引
        non_zero_indices = [i for i, val in enumerate(individual) if val > 0 and i != 1]
        mutation_index = random.choice(non_zero_indices)
        individual[mutation_index] = random.uniform(5, 50)

        # 调整其他非零元素以确保总和加上Mo的值等于100
        remaining = 100 - individual[5] - individual[mutation_index]
        adjust_indices = [i for i in non_zero_indices if i != mutation_index]
        adjust_values = np.random.uniform(0, 1, len(adjust_indices))
        adjust_values /= adjust_values.sum()
        adjust_values *= remaining

        for idx, value in zip(adjust_indices, adjust_values):
            individual[idx] = max(5, individual[idx] + value)  # 确保不小于5

        individual[5] = max(individual[5], 5)  # 确保Mo不小于20

        # 确保未参与变异的位置仍然为0
        for i in range(len(individual)):
            if i not in non_zero_indices and i != 5:
                individual[i] = 0

    return individual,


# 定义个体如何进行交叉操作
def crossover(ind1, ind2):
    # 获取两个父代的非零元素位置
    non_zero_indices_1 = [i for i, val in enumerate(ind1) if val > 0]
    non_zero_indices_2 = [i for i, val in enumerate(ind2) if val > 0]

    # 选择交叉点，这里简化为只在共同的非零元素位置进行交叉
    common_non_zero_indices = list(set(non_zero_indices_1) & set(non_zero_indices_2))
    if common_non_zero_indices:
        cxpoint = random.choice(common_non_zero_indices)

        # 交换两个父代在交叉点的基因
        ind1[cxpoint], ind2[cxpoint] = ind2[cxpoint], ind1[cxpoint]

    # 交换独有的非零元素位置上的基因
    unique_non_zero_indices_1 = list(set(non_zero_indices_1) - set(non_zero_indices_2))
    unique_non_zero_indices_2 = list(set(non_zero_indices_2) - set(non_zero_indices_1))
    for idx1, idx2 in zip(unique_non_zero_indices_1, unique_non_zero_indices_2):
        ind1[idx1], ind2[idx2] = ind2[idx2], ind1[idx1]

    return ind1, ind2


def evaluate_individual(individual):
    # 将个体转换为DataFrame，以符合模型的输入要求
    individual_df = pd.DataFrame([individual], columns=X.columns)

    # 使用模型预测硬度
    hardness = model.predict(individual_df)[0]

    # 返回硬度的负值作为适应度
    return hardness,


# 注册遗传算法所需的各种操作
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


# 遗传算法的主函数
def main():
    pop = toolbox.population(n=50)  # 创建初始种群
    hof = tools.HallOfFame(1)  # 用于记录最佳个体
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 应用遗传算法
    algorithms.eaSimple(pop, toolbox, cxpb=0.2, mutpb=0.2, ngen=100, stats=stats, halloffame=hof, verbose=True)

    return hof


def normalize_individual(individual):
    non_zero_indices = [i for i, val in enumerate(individual) if val > 0]
    non_zero_values = [individual[i] for i in non_zero_indices]

    # 归一化这些非零值，使得它们的总和为100
    total_non_zero = sum(non_zero_values)
    non_zero_values = [val * 100 / total_non_zero for val in non_zero_values]

    # 分配归一化后的值回个体，确保总和为100
    # 由于四舍五入可能导致总和略有偏差，我们在最后一个元素上进行调整
    for i in range(len(non_zero_values) - 1):
        individual[non_zero_indices[i]] = non_zero_values[i]
    # 确保最后一个元素调整后非零值之和为100
    individual[non_zero_indices[-1]] = 100 - sum(non_zero_values[:-1])

    return individual

import json
from deap import creator, base, tools, algorithms
import numpy as np
import random

# 假设 model 是一个训练好的模型，这里只是为了示例
# 在实际使用中，您应该将 model 替换为您的机器学习模型

# 其他遗传算法相关函数的定义

if __name__ == "__main__":
    results = []
    for i in range(2200):
        best_individual = main()
        normalized_best_individual = normalize_individual(best_individual[0])
        print("Best Individual: ", normalized_best_individual)

        # 将 normalized_best_individual 转换为带有正确列名的 DataFrame
        individual_df = pd.DataFrame([normalized_best_individual], columns=X.columns)

        best_hv = model.predict(individual_df)[0]  # 直接传递 individual_df，不放入列表

        print("Predicted HV: ", best_hv)
        if best_hv > 600:
            results.append({
                "normalized_best_individual": list(normalized_best_individual),  # 转换为列表以便于JSON序列化
                "best_hv": float(best_hv)
            })


    # 将结果写入 JSON 文件
    with open("/Users/macaroni/Downloads/pythonProject1/HEA_Abstract/result_genetic_algorithm_results_400.json", "a") as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')  # 在每个结果后添加换行符


