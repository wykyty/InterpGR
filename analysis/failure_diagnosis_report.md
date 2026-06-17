# 检索失败诊断报告：SAE 特征层面的分析

## 核心问题

检索失败时，模型的内部表征出了什么问题？
是路由特征失活（走错分支），还是细粒度区分特征失活（前缀对了但最后错了）？

## 1. 数据概览

- 成功样本（top-1 检索正确）: 3483 条
- 失败样本（top-1 检索错误）: 3483 条
- SAE 特征维度: 8192
- 每个 token 激活的特征数: ~100 (k=100)

## 2. 成功特征：成功时更活跃，失败时 "熄火"

筛选标准: Cohen's d > 0.3（成功组均值更高）且激活率差异 > 5%

共找到 **22** 个成功特征。

### Top-30 成功特征

| 排名 | 特征ID | Cohen's d | 成功组均值 | 失败组均值 | 成功激活率 | 失败激活率 | 激活率差 | top-k选中率差 |
|------|--------|-----------|-----------|-----------|-----------|-----------|---------|-------------|
| 1 | 1155 | 0.558 | 8.73 | 2.05 | 35.9% | 16.2% | +19.7% | +0.0% |
| 2 | 4613 | 0.550 | 10.58 | 3.18 | 40.0% | 21.5% | +18.4% | +0.1% |
| 3 | 2719 | 0.509 | 7.88 | 2.77 | 40.3% | 23.8% | +16.6% | +0.0% |
| 4 | 3791 | 0.493 | 10.02 | 4.04 | 49.0% | 33.9% | +15.1% | +0.0% |
| 5 | 2908 | 0.486 | 5.75 | 1.68 | 32.5% | 15.2% | +17.3% | +0.0% |
| 6 | 6519 | 0.476 | 4.25 | 1.03 | 27.9% | 10.1% | +17.9% | +0.0% |
| 7 | 2754 | 0.408 | 4.33 | 1.35 | 27.4% | 11.6% | +15.7% | +0.0% |
| 8 | 5076 | 0.397 | 4.57 | 1.61 | 33.2% | 15.8% | +17.4% | +0.0% |
| 9 | 7675 | 0.375 | 4.42 | 1.47 | 26.3% | 12.8% | +13.5% | +0.0% |
| 10 | 6973 | 0.351 | 2.35 | 0.61 | 18.5% | 5.9% | +12.5% | +0.0% |
| 11 | 1370 | 0.348 | 3.28 | 0.91 | 21.6% | 8.0% | +13.6% | +0.0% |
| 12 | 6901 | 0.347 | 2.55 | 0.75 | 21.4% | 9.1% | +12.2% | +0.0% |
| 13 | 8093 | 0.344 | 2.87 | 0.72 | 17.4% | 6.4% | +11.0% | +0.0% |
| 14 | 1343 | 0.343 | 2.49 | 0.80 | 21.5% | 9.5% | +12.0% | +0.0% |
| 15 | 1934 | 0.342 | 5.17 | 2.22 | 30.8% | 20.6% | +10.2% | +0.0% |
| 16 | 161 | 0.341 | 4.54 | 1.85 | 27.7% | 16.2% | +11.5% | +0.0% |
| 17 | 6785 | 0.337 | 5.49 | 2.42 | 33.0% | 22.0% | +11.0% | +0.1% |
| 18 | 7013 | 0.328 | 3.97 | 1.62 | 32.3% | 17.9% | +14.4% | -0.1% |
| 19 | 6831 | 0.305 | 1.26 | 0.40 | 16.5% | 6.4% | +10.0% | +0.0% |
| 20 | 7816 | 0.304 | 2.66 | 1.25 | 27.5% | 16.0% | +11.4% | +0.0% |
| 21 | 767 | 0.303 | 2.05 | 0.68 | 18.9% | 8.1% | +10.8% | +0.0% |
| 22 | 8083 | 0.302 | 2.29 | 0.87 | 19.9% | 11.0% | +8.9% | +0.0% |

### Top-10 成功特征详细分析

#### Feature 1155 (Cohen's d = 0.558)

**激活率**: 成功 35.9% vs 失败 16.2% (差 +19.7%)
**均值**: 成功 8.73 vs 失败 2.05
**DocID 前缀一致性**: 20% (最常见前缀 (27, 0), 3/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 102.20 | where do characters live in this is us | [10, 24, 0, 0] |
| 2 | 102.16 | when is this is us season 2 released on dvd | [27, 0, 12, 0] |
| 3 | 101.59 | where did the legend of the easter bunny come from | [18, 7, 19, 0] |
| 4 | 99.11 | how did they find the jersey shore cast | [27, 0, 28, 0] |
| 5 | 89.46 | who will take the throne after the queen dies | [16, 0, 16, 0] |
| 6 | 87.23 | who made the song we are the world | [15, 21, 20, 0] |
| 7 | 85.82 | what kind of beast is the beast from beauty and the beast | [27, 28, 25, 1] |
| 8 | 84.93 | how old were jersey shore cast first season | [27, 0, 28, 0] |
| 9 | 81.24 | what are the main crops grown in the united states | [28, 16, 14, 0] |
| 10 | 79.67 | who sang first line of we are the world | [15, 21, 20, 0] |

#### Feature 4613 (Cohen's d = 0.550)

**激活率**: 成功 40.0% vs 失败 21.5% (差 +18.4%)
**均值**: 成功 10.58 vs 失败 3.18
**DocID 前缀一致性**: 13% (最常见前缀 (26, 12), 2/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 112.56 | which greek god flew too close to the sun | [4, 0, 4, 0] |
| 2 | 105.54 | where is the amazon river located in south america | [16, 7, 23, 0] |
| 3 | 102.45 | who sold the most records elvis or the beatles | [26, 12, 5, 0] |
| 4 | 92.48 | where does the ohio river and the mississippi river meet | [9, 17, 0, 0] |
| 5 | 92.30 | where is dia de los muertos celebrated in mexico | [18, 26, 7, 0] |
| 6 | 87.63 | what were the article of confederation and what powers did they gra... | [16, 4, 1, 0] |
| 7 | 84.64 | been through the desert on a horse with no name neil young | [19, 12, 12, 0] |
| 8 | 84.24 | where is nathan's hotdog eating contest held | [4, 11, 12, 0] |
| 9 | 83.05 | who is the most selling music artist of all time | [26, 12, 5, 0] |
| 10 | 82.84 | what is the purpose of the national do not call registry | [23, 4, 5, 0] |

#### Feature 2719 (Cohen's d = 0.509)

**激活率**: 成功 40.3% vs 失败 23.8% (差 +16.6%)
**均值**: 成功 7.88 vs 失败 2.77
**DocID 前缀一致性**: 33% (最常见前缀 (27, 17), 5/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 73.96 | as a nigerian do i need a visa to visit jamaica | [14, 14, 12, 0] |
| 2 | 73.87 | what is billy last name in where the red fern grows | [8, 1, 22, 0] |
| 3 | 73.02 | who plays faith on when calls the heart | [27, 17, 18, 0] |
| 4 | 70.29 | when was the last time kentucky won ncaa | [14, 11, 3, 0] |
| 5 | 68.08 | who won the first medal in olympics for india | [11, 20, 7, 0] |
| 6 | 65.65 | who plays sofia in when calls the heart | [27, 17, 18, 0] |
| 7 | 65.35 | where does new crust come from in sea floor spreading | [4, 1, 10, 0] |
| 8 | 64.74 | who played mr thatcher in when calls the heart | [27, 17, 18, 0] |
| 9 | 63.34 | where do polar bears live and what's their habitat | [22, 22, 8, 0] |
| 10 | 62.99 | who played charles on when calls the heart | [27, 17, 18, 0] |

#### Feature 3791 (Cohen's d = 0.493)

**激活率**: 成功 49.0% vs 失败 33.9% (差 +15.1%)
**均值**: 成功 10.02 vs 失败 4.04
**DocID 前缀一致性**: 27% (最常见前缀 (2, 13), 4/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 105.92 | kuch rang pyar ke aise bhi colors tv | [23, 14, 19, 0] |
| 2 | 102.49 | when does the curious incident of the dog in the nighttime take place | [10, 14, 24, 0] |
| 3 | 100.42 | can you carry a pocket knife in canada | [26, 4, 26, 0] |
| 4 | 93.12 | which is the ring finger for male in india | [26, 0, 2, 0] |
| 5 | 89.89 | how many episodes is season 4 of the flash | [2, 13, 12, 0] |
| 6 | 84.55 | who won the battle of the first battle of bull run | [7, 2, 0, 0] |
| 7 | 83.51 | when did the battle of bull run start | [7, 2, 0, 0] |
| 8 | 82.81 | different ways to say bless you in french | [18, 17, 2, 0] |
| 9 | 81.50 | who is the most selling music artist of all time | [26, 12, 5, 0] |
| 10 | 79.37 | who won the 1st battle of bull run | [7, 2, 0, 0] |

#### Feature 2908 (Cohen's d = 0.486)

**激活率**: 成功 32.5% vs 失败 15.2% (差 +17.3%)
**均值**: 成功 5.75 vs 失败 1.68
**DocID 前缀一致性**: 20% (最常见前缀 (27, 17), 3/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 73.64 | who was the last nba player to get drafted out of high school | [28, 21, 7, 0] |
| 2 | 59.16 | what are the main crops grown in the united states | [28, 16, 14, 0] |
| 3 | 58.88 | what is the width of the mississippi river | [16, 14, 15, 0] |
| 4 | 58.69 | can you carry a pocket knife in canada | [26, 4, 26, 0] |
| 5 | 54.96 | what did the cast of stranger things make | [27, 17, 8, 0] |
| 6 | 54.85 | who was the killer in the movie i know what you did last summer | [10, 17, 8, 0] |
| 7 | 54.07 | when does season 3 of strnger things come out | [27, 17, 8, 0] |
| 8 | 53.96 | where does captain america civil war take place | [27, 10, 13, 0] |
| 9 | 52.62 | when was the minimum wage established in the united states | [6, 29, 5, 0] |
| 10 | 51.79 | where do you think most farming is done in the united states and why | [28, 16, 14, 0] |

#### Feature 6519 (Cohen's d = 0.476)

**激活率**: 成功 27.9% vs 失败 10.1% (差 +17.9%)
**均值**: 成功 4.25 vs 失败 1.03
**DocID 前缀一致性**: 13% (最常见前缀 (14, 20), 2/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 83.36 | what was the population of the roman empire at its height | [14, 20, 17, 0] |
| 2 | 68.44 | when did the 12 tribes of israel form | [14, 20, 12, 0] |
| 3 | 64.64 | when did cristiano ronaldo go to manchester united | [25, 4, 0, 0] |
| 4 | 60.27 | i'm not a robot episode 25 summary | [3, 28, 14, 0] |
| 5 | 54.29 | when did cat on a hot tin roof take place | [25, 0, 8, 0] |
| 6 | 53.49 | where can the mona lisa be found today | [25, 17, 4, 0] |
| 7 | 53.45 | which greek god flew too close to the sun | [4, 0, 4, 0] |
| 8 | 52.19 | who sings you're welcome in moana credits | [3, 12, 18, 0] |
| 9 | 51.14 | when does the chinese new year begin and end | [18, 26, 27, 0] |
| 10 | 49.49 | when is chinese new year and what year is it | [18, 26, 27, 0] |

#### Feature 2754 (Cohen's d = 0.408)

**激活率**: 成功 27.4% vs 失败 11.6% (差 +15.7%)
**均值**: 成功 4.33 vs 失败 1.35
**DocID 前缀一致性**: 20% (最常见前缀 (28, 0), 3/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 97.13 | when did cat on a hot tin roof take place | [25, 0, 8, 0] |
| 2 | 68.09 | what is earth's magnetic field responsible for | [28, 7, 8, 0] |
| 3 | 66.80 | what is billy last name in where the red fern grows | [8, 1, 22, 0] |
| 4 | 66.72 | what is the function of red bone marrow | [28, 0, 9] |
| 5 | 61.69 | some fungi are components of normal human microflora | [28, 0, 2, 1] |
| 6 | 58.31 | using illustration give detail account of cell division in living o... | [28, 0, 29] |
| 7 | 54.06 | when did the days of the week get named | [26, 28, 29, 0] |
| 8 | 53.79 | what kind of metric system does the us use | [28, 19, 22, 0] |
| 9 | 51.81 | when does star trek discovery air on tv | [27, 17, 17, 0] |
| 10 | 51.23 | naa peru surya naa illi india mp3 songs | [10, 19, 28, 0] |

#### Feature 5076 (Cohen's d = 0.397)

**激活率**: 成功 33.2% vs 失败 15.8% (差 +17.4%)
**均值**: 成功 4.57 vs 失败 1.61
**DocID 前缀一致性**: 13% (最常见前缀 (10, 27), 2/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 84.28 | when did one child policy end in china | [16, 13, 8, 0] |
| 2 | 74.38 | when was child benefit paid for the first child | [18, 13, 13, 0] |
| 3 | 72.36 | where is the island in and then there were none | [10, 27, 7, 0] |
| 4 | 70.06 | who was mr.owen in and then there were none | [10, 27, 7, 0] |
| 5 | 63.39 | when was united nations convention on the rights of the child created | [6, 20, 26, 0] |
| 6 | 62.94 | why was medicare part c put into effect | [6, 26, 18, 4] |
| 7 | 62.77 | who created the convention on the rights of the child | [6, 20, 26, 0] |
| 8 | 62.68 | when does the implantation of the embryo occur | [24, 14, 10, 0] |
| 9 | 62.62 | who propounded the idea of basic education in india | [6, 3, 10, 3] |
| 10 | 61.99 | why did walt disney create the walt disney company | [16, 21, 16, 2] |

#### Feature 7675 (Cohen's d = 0.375)

**激活率**: 成功 26.3% vs 失败 12.8% (差 +13.5%)
**均值**: 成功 4.42 vs 失败 1.47
**DocID 前缀一致性**: 13% (最常见前缀 (25, 20), 2/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 97.30 | who is president of india in present time | [16, 0, 7, 0] |
| 2 | 92.31 | history of development of nuclear energy in india | [16, 17, 10, 0] |
| 3 | 77.92 | who won the first medal in olympics for india | [11, 20, 7, 0] |
| 4 | 73.77 | who has the rights to alice in wonderland | [10, 1, 7, 0] |
| 5 | 73.64 | where is the emerald ash borer found in the us | [7, 27, 20] |
| 6 | 64.15 | what's on each level of the eiffel tower | [25, 20, 0, 0] |
| 7 | 63.10 | who has won the eurovision song contest the most times | [16, 27, 22, 0] |
| 8 | 59.56 | who sang the most number of songs in the world | [28, 23, 29, 0] |
| 9 | 59.43 | who hoisted indian flag abroad for the first time | [25, 25, 10, 1] |
| 10 | 58.31 | who is known as father of green revolution in india | [23, 12, 13, 0] |

#### Feature 6973 (Cohen's d = 0.351)

**激活率**: 成功 18.5% vs 失败 5.9% (差 +12.5%)
**均值**: 成功 2.35 vs 失败 0.61
**DocID 前缀一致性**: 20% (最常见前缀 (16, 5), 3/15)

**类型判断**: **语义/细粒度特征** — top 样本主题相关但 DocID 前缀分散

| 排名 | 激活值 | Query | DocID (语义ID) |
|------|--------|-------|---------------|
| 1 | 66.61 | who will take the throne after the queen dies | [16, 0, 16, 0] |
| 2 | 60.34 | who is next in line to inherit the british throne | [16, 0, 16, 0] |
| 3 | 56.23 | a political leader during the roman empire was called | [14, 20, 17, 0] |
| 4 | 50.74 | who plays patroclus in troy fall of a city | [2, 10, 22] |
| 5 | 48.32 | what was the population of the roman empire at its height | [14, 20, 17, 0] |
| 6 | 46.06 | when was the last time michigan basketball won the championship | [16, 5, 9, 0] |
| 7 | 45.57 | what was the climate like in ancient egypt | [14, 10, 17, 13] |
| 8 | 41.79 | who won last year's ncaa women's basketball | [16, 5, 16, 0] |
| 9 | 41.33 | why were the wars between rome and carthage called the punic wars | [25, 21, 2, 0] |
| 10 | 40.33 | when's the last time michigan won a national championship in basket... | [16, 5, 9, 0] |


## 3. 失败特征：失败时更活跃，可能是 "错误信号"

筛选标准: Cohen's d < -0.3（失败组均值更高）且激活率差异 < -5%

共找到 **0** 个失败特征。

### Top-30 失败特征

| 排名 | 特征ID | Cohen's d | 成功组均值 | 失败组均值 | 成功激活率 | 失败激活率 | 激活率差 | top-k选中率差 |
|------|--------|-----------|-----------|-----------|-----------|-----------|---------|-------------|

### Top-10 失败特征详细分析


## 4. 中等效应量特征 (|d| > 0.5)

成功特征 (d > 0.5, 激活率差 > 10%): **3** 个
失败特征 (d < -0.5, 激活率差 < -10%): **0** 个

### 成功特征 (中等效应量)

| 特征ID | Cohen's d | 成功激活率 | 失败激活率 | 激活率差 |
|--------|-----------|-----------|-----------|---------|
| 1155 | 0.558 | 35.9% | 16.2% | +19.7% |
| 4613 | 0.550 | 40.0% | 21.5% | +18.4% |
| 2719 | 0.509 | 40.3% | 23.8% | +16.6% |

## 5. 总结：检索失败时，模型内部出了什么问题？

### 成功特征的类型分布（top-20）

- 路由特征（DocID 前缀一致 > 50%）: 0 个
- 语义/细粒度特征（DocID 前缀分散）: 20 个

### 失败特征的类型分布（top-20）

- 路由特征: 0 个
- 语义/细粒度特征: 0 个

### 关键发现

1. **成功特征**在成功样本中显著更活跃。这些特征可能编码了正确检索所需的关键信息——当它们 "熄火" 时，模型就会犯错。
2. **失败特征**在失败样本中更活跃。这些可能是 "干扰信号"——模型被错误的模式吸引，导致走错分支。
3. 通过观察特征的 top 激活样本，可以判断特征是编码 DocID 路由（前缀一致）还是编码细粒度语义（主题相关但前缀分散）。
4. 如果成功特征主要是路由特征，说明失败是因为走错了大分支；如果成功特征主要是细粒度特征，说明失败是因为在最后一步区分能力不足。