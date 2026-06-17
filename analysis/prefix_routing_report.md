# DocID 层次结构在模型内部的存储分析

## 核心问题

是否存在一组 SAE 特征，在模型准备生成特定前缀时一致地、排他地激活？
如果存在，这些就是 **前缀路由特征**——直接证明模型内部存储了树结构的离散路由信号。

## 1. 数据概览

- SAE 特征维度: 8192
- 样本数: 7830
- 第一层前缀数: 30（Semantic ID 第一个 token, 取值 0-29）

| 前缀 | 样本数 | 占比 |
|------|--------|------|
| 0 | 248 | 3.2% |
| 1 | 188 | 2.4% |
| 2 | 248 | 3.2% |
| 3 | 323 | 4.1% |
| 4 | 412 | 5.3% |
| 5 | 239 | 3.1% |
| 6 | 259 | 3.3% |
| 7 | 233 | 3.0% |
| 8 | 147 | 1.9% |
| 9 | 155 | 2.0% |
| 10 | 286 | 3.7% |
| 11 | 430 | 5.5% |
| 12 | 184 | 2.3% |
| 13 | 173 | 2.2% |
| 14 | 513 | 6.6% |
| 15 | 164 | 2.1% |
| 16 | 564 | 7.2% |
| 17 | 149 | 1.9% |
| 18 | 271 | 3.5% |
| 19 | 243 | 3.1% |
| 20 | 161 | 2.1% |
| 21 | 217 | 2.8% |
| 22 | 185 | 2.4% |
| 23 | 234 | 3.0% |
| 24 | 183 | 2.3% |
| 25 | 174 | 2.2% |
| 26 | 420 | 5.4% |
| 27 | 402 | 5.1% |
| 28 | 276 | 3.5% |
| 29 | 149 | 1.9% |

## 2. 前缀路由特征

**筛选标准**: 对某前缀 z-score > 2.0，且排他性 > 1.5（最高 z 比第二高 z 高 50% 以上）

共找到 **67** 个前缀路由特征。

### 完整列表

| 排名 | 特征ID | 目标前缀 | z-score | 第二高z | 排他性 | 目标前缀样本数 |
|------|--------|---------|---------|---------|--------|-------------|
| 1 | 2125 | 1 | 3.70 | 0.12 | 30.83x | 188 |
| 2 | 7760 | 8 | 3.59 | 0.97 | 3.72x | 147 |
| 3 | 7568 | 2 | 3.14 | 1.76 | 1.78x | 248 |
| 4 | 7771 | 9 | 2.99 | 0.28 | 10.60x | 155 |
| 5 | 3369 | 15 | 2.99 | 1.07 | 2.79x | 164 |
| 6 | 2236 | 25 | 2.80 | 0.23 | 12.14x | 174 |
| 7 | 3407 | 2 | 2.68 | 0.56 | 4.83x | 248 |
| 8 | 4329 | 15 | 2.62 | 0.89 | 2.94x | 164 |
| 9 | 1301 | 12 | 2.57 | 0.38 | 6.74x | 184 |
| 10 | 7530 | 28 | 2.53 | 1.53 | 1.66x | 276 |
| 11 | 2680 | 8 | 2.50 | 1.27 | 1.97x | 147 |
| 12 | 5059 | 9 | 2.48 | 0.47 | 5.32x | 155 |
| 13 | 7113 | 22 | 2.47 | 1.07 | 2.30x | 185 |
| 14 | 817 | 2 | 2.43 | 1.20 | 2.03x | 248 |
| 15 | 3806 | 8 | 2.43 | 0.57 | 4.26x | 147 |
| 16 | 1028 | 9 | 2.38 | 1.02 | 2.32x | 155 |
| 17 | 1460 | 21 | 2.37 | 0.29 | 8.09x | 217 |
| 18 | 1717 | 1 | 2.37 | 0.49 | 4.88x | 188 |
| 19 | 5018 | 0 | 2.35 | 0.43 | 5.52x | 248 |
| 20 | 6632 | 20 | 2.34 | 0.54 | 4.31x | 161 |
| 21 | 2809 | 2 | 2.31 | 0.74 | 3.12x | 248 |
| 22 | 18 | 10 | 2.31 | 0.04 | 59.82x | 286 |
| 23 | 7154 | 8 | 2.29 | 1.00 | 2.28x | 147 |
| 24 | 2069 | 2 | 2.28 | 0.83 | 2.75x | 248 |
| 25 | 6437 | 25 | 2.27 | 0.03 | 74.48x | 174 |
| 26 | 6796 | 10 | 2.27 | 0.83 | 2.72x | 286 |
| 27 | 1321 | 2 | 2.26 | 0.32 | 7.08x | 248 |
| 28 | 7505 | 1 | 2.26 | 0.51 | 4.40x | 188 |
| 29 | 6972 | 10 | 2.26 | 0.01 | 300.61x | 286 |
| 30 | 5630 | 2 | 2.24 | 0.49 | 4.63x | 248 |
| 31 | 371 | 20 | 2.24 | 1.22 | 1.84x | 161 |
| 32 | 3002 | 8 | 2.24 | 1.15 | 1.95x | 147 |
| 33 | 6864 | 8 | 2.23 | 1.00 | 2.23x | 147 |
| 34 | 2487 | 10 | 2.23 | 0.62 | 3.61x | 286 |
| 35 | 5103 | 10 | 2.18 | 0.34 | 6.33x | 286 |
| 36 | 5829 | 0 | 2.18 | 0.82 | 2.65x | 248 |
| 37 | 2602 | 8 | 2.18 | 0.69 | 3.17x | 147 |
| 38 | 7969 | 2 | 2.17 | 0.48 | 4.53x | 248 |
| 39 | 4742 | 2 | 2.16 | 0.90 | 2.40x | 248 |
| 40 | 6529 | 10 | 2.16 | 0.65 | 3.32x | 286 |
| 41 | 4521 | 2 | 2.15 | 0.18 | 12.12x | 248 |
| 42 | 4064 | 23 | 2.15 | 0.12 | 18.65x | 234 |
| 43 | 4035 | 24 | 2.15 | 0.13 | 16.67x | 183 |
| 44 | 7451 | 1 | 2.13 | 0.46 | 4.61x | 188 |
| 45 | 5581 | 25 | 2.13 | 0.09 | 23.29x | 174 |
| 46 | 4357 | 29 | 2.11 | 0.34 | 6.23x | 149 |
| 47 | 6966 | 9 | 2.11 | 0.62 | 3.41x | 155 |
| 48 | 5065 | 15 | 2.09 | 1.06 | 1.98x | 164 |
| 49 | 5047 | 22 | 2.09 | 0.43 | 4.89x | 185 |
| 50 | 1690 | 15 | 2.08 | 1.23 | 1.69x | 164 |

### 按前缀分组

#### 前缀 0（248 个样本，5 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 5018 | 2.35 | 5.52x |
| 5829 | 2.18 | 2.65x |
| 6163 | 2.03 | 5.19x |
| 1975 | 2.02 | 2.24x |
| 1868 | 2.01 | 5.70x |

**Feature 5018** (z=2.35, 排他性=5.52x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 42.91 | what is meant by thin film in physics | [0, 23, 1, 8] | 0 |
| 35.09 | how can the sarb use the cash reserves requirements and o... | [0, 6, 11, 2] | 0 |
| 34.57 | which type of fire detector uses the effect of smoke on a... | [0, 23, 13, 3] | 0 |
| 34.31 | what are the active materials of a lead acid battery | [0, 13, 28, 2] | 0 |
| 33.02 | where is the slide placed on the microscope | [26, 3, 28, 4] | 26 |

**Feature 5829** (z=2.18, 排他性=2.65x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 91.85 | where is carbohydrate converted to glucose through the pr... | [0, 20, 3, 3] | 0 |
| 85.22 | where does the digestion of food take place | [0, 20, 18, 0] | 0 |
| 83.33 | what does the cytoplasm do for the animal cell | [0, 20, 24, 1] | 0 |
| 78.20 | which of these is best description of passive transport | [0, 20, 20, 4] | 0 |
| 74.67 | how many co2 molecules are produced in aerobic respiration | [0, 20, 20, 0] | 0 |

**Feature 6163** (z=2.03, 排他性=5.19x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 67.58 | tools made from high-speed tool steel are generally used ... | [0, 13, 12, 1] | 0 |
| 61.33 | who brought the idea of castles to england | [0, 10, 10, 1] | 0 |
| 53.97 | when did beds become popular in france and germany | [0, 13, 4, 0] | 0 |
| 50.37 | what type of speed does a speedometer measure | [0, 7, 27, 2] | 0 |
| 50.14 | when was the first documented case of tool mark identific... | [0, 22, 10, 5] | 0 |

#### 前缀 1（188 个样本，5 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 2125 | 3.70 | 30.83x |
| 1717 | 2.37 | 4.88x |
| 7505 | 2.26 | 4.40x |
| 7451 | 2.13 | 4.61x |
| 3569 | 2.05 | 3.16x |

**Feature 2125** (z=3.70, 排他性=30.83x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 89.35 | when was the draft of the national assembly constitution ... | [1, 6, 8, 0] | 1 |
| 77.69 | who had created the second bank of the united states | [1, 6, 20, 1] | 1 |
| 64.48 | who wrote most of the declaration of independance | [1, 6, 2, 0] | 1 |
| 63.07 | word that means separation of church and state | [1, 23, 3, 0] | 1 |
| 53.25 | summary on the i have a dream speech | [1, 18, 21, 1] | 1 |

**Feature 1717** (z=2.37, 排他性=4.88x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 84.74 | who made the poppies at tower of london | [1, 19, 7, 0] | 1 |
| 80.08 | 17th century conceptions of liberty and freedom included ... | [1, 10, 14, 3] | 1 |
| 75.23 | who was the declaration of independence written for | [1, 6, 2, 0] | 1 |
| 72.96 | when did it become law to stand for the national anthem | [1, 8, 9, 0] | 1 |
| 70.82 | who is mostly responsible for writing the declaration of ... | [1, 6, 2, 0] | 1 |

**Feature 7505** (z=2.26, 排他性=4.40x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 47.03 | what is the meaning of the book of proverbs | [1, 21, 11, 4] | 1 |
| 36.51 | what is the meaning of peter piper picked a peck of pickl... | [1, 11, 1, 2] | 1 |
| 36.05 | what is the oath that new citizens take | [1, 23, 11, 0] | 1 |
| 34.97 | meaning of peter piper picked a peck of pickled peppers | [1, 11, 1, 2] | 1 |
| 30.86 | when did the three little pigs come out | [1, 11, 1, 5] | 1 |

#### 前缀 2（248 个样本，15 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 7568 | 3.14 | 1.78x |
| 3407 | 2.68 | 4.83x |
| 817 | 2.43 | 2.03x |
| 2809 | 2.31 | 3.12x |
| 2069 | 2.28 | 2.75x |
| 1321 | 2.26 | 7.08x |
| 5630 | 2.24 | 4.63x |
| 7969 | 2.17 | 4.53x |
| 4742 | 2.16 | 2.40x |
| 4521 | 2.15 | 12.12x |

**Feature 7568** (z=3.14, 排他性=1.78x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 95.85 | actress who plays brad pitts wife in war machine | [2, 1, 19, 1] | 2 |
| 89.53 | who plays drew's boyfriend on the night shift | [2, 19, 27] | 2 |
| 88.52 | when does season 2 of limitless come out | [2, 13, 4, 1] | 2 |
| 85.56 | how many episodes are in season 7 of pretty little liars | [2, 13, 3, 2] | 2 |
| 84.71 | how many episodes in season 3 of good witch | [2, 13, 3, 3] | 2 |

**Feature 3407** (z=2.68, 排他性=4.83x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 74.97 | when is the next tangled the series episode coming out | [2, 13, 13] | 2 |
| 44.52 | how many episodes of supergirl is superman in | [2, 13, 3, 1] | 2 |
| 43.19 | will there be a 2nd season of monster musume | [2, 14, 1, 2] | 2 |
| 42.79 | will there be a third season of the durells in corfu | [2, 27, 15] | 2 |
| 41.80 | where is prokaryotic life found around hydrothermal vents | [26, 13, 17, 7] | 26 |

**Feature 817** (z=2.43, 排他性=2.03x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 83.20 | what is a wrinkle in time based on | [2, 22, 1, 0] | 2 |
| 80.19 | forney's sister in where the heart is | [2, 1, 7, 0] | 2 |
| 80.04 | who plays young lydia in one day at a time | [8, 22, 10, 0] | 8 |
| 78.42 | will there be a third season of the durells in corfu | [2, 27, 15] | 2 |
| 76.77 | who voices randy in f is for family | [2, 25, 22, 3] | 2 |

#### 前缀 3（323 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 5672 | 2.01 | 7.01x |

**Feature 5672** (z=2.01, 排他性=7.01x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 51.65 | top 10 most viewed youtube videos in india | [3, 12, 25, 0] | 3 |
| 45.85 | who played santa in the santa clause movies | [3, 3, 25] | 3 |
| 44.79 | whats the most liked picture on instagram 2018 | [3, 25, 2, 1] | 3 |
| 43.47 | ben 10 ultimate alien episode 2 season 1 | [3, 9, 0, 8] | 3 |
| 42.49 | who won the fifth season of america's got talent | [3, 16, 14, 6] | 3 |

#### 前缀 6（259 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 6104 | 2.08 | 5.24x |

**Feature 6104** (z=2.08, 排他性=5.24x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 87.23 | what ethnic group celebrates its contribution to america ... | [6, 15, 13] | 6 |
| 62.34 | with a land area of 54 314 square miles where does wiscon... | [6, 22, 15] | 6 |
| 60.54 | texas uses what kind of voter registration system | [6, 25, 11, 4] | 6 |
| 58.64 | what is the number of total presidential electoral votes | [6, 23, 0, 0] | 6 |
| 58.43 | when does new model year start for cars | [6, 13, 2, 3] | 6 |

#### 前缀 8（147 个样本，7 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 7760 | 3.59 | 3.72x |
| 2680 | 2.50 | 1.97x |
| 3806 | 2.43 | 4.26x |
| 7154 | 2.29 | 2.28x |
| 3002 | 2.24 | 1.95x |
| 6864 | 2.23 | 2.23x |
| 2602 | 2.18 | 3.17x |

**Feature 7760** (z=3.59, 排他性=3.72x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 95.45 | what book of the bible is the song of solomon in | [1, 21, 11, 1] | 1 |
| 90.33 | where is the villa in call me by your name | [8, 11, 9, 0] | 8 |
| 87.96 | where did the book small steps take place | [8, 20, 15, 1] | 8 |
| 85.12 | where have you been where are you going short story | [8, 20, 21, 2] | 8 |
| 84.20 | what is the short story the gift of the magi about | [8, 20, 21, 0] | 8 |

**Feature 2680** (z=2.50, 排他性=1.97x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 57.52 | bill the galactic hero on the planet of robot slaves | [8, 20, 15, 3] | 8 |
| 57.20 | where did the book small steps take place | [8, 20, 15, 1] | 8 |
| 53.59 | description of greg from diary of a wimpy kid | [13, 25, 28, 3] | 13 |
| 42.07 | justify the title of the novel sense and sensibility | [25, 0, 14, 6] | 25 |
| 38.72 | who played harley in harley davidson and the marlboro man | [8, 2, 8, 3] | 8 |

**Feature 3806** (z=2.43, 排他性=4.26x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 57.50 | what is the short story the gift of the magi about | [8, 20, 21, 0] | 8 |
| 56.67 | around the world in 80 days book pages | [8, 14, 8, 2] | 8 |
| 54.79 | the old man and the sea page count | [8, 20, 21, 1] | 8 |
| 48.10 | what is the story of around the world in 80 days | [8, 14, 8, 2] | 8 |
| 41.15 | the fire and the rain summary by girish karnad | [8, 17, 6, 0] | 8 |

#### 前缀 9（155 个样本，4 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 7771 | 2.99 | 10.60x |
| 5059 | 2.48 | 5.32x |
| 1028 | 2.38 | 2.32x |
| 6966 | 2.11 | 3.41x |

**Feature 7771** (z=2.99, 排他性=10.60x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 118.90 | what continents does the ring of fire touch | [9, 27, 24, 0] | 9 |
| 94.09 | how much greater is the average temperature on earth than... | [9, 27, 0, 4] | 9 |
| 89.85 | where does the coral sea meet the pacific ocean | [9, 3, 13, 11] | 9 |
| 81.00 | what color is the golden gate bridge in san francisco | [9, 10, 1, 0] | 9 |
| 74.83 | name of volcano that erupted in iceland in 2010 | [9, 18, 1, 2] | 9 |

**Feature 5059** (z=2.48, 排他性=5.32x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 82.04 | who lives on 50th floor of trump tower | [9, 10, 8, 0] | 9 |
| 50.13 | who has played in the most masters tournaments | [9, 4, 17, 0] | 9 |
| 49.38 | how many wins does tiger woods have on the pga tour | [13, 9, 29, 0] | 13 |
| 48.01 | what mall did they use in back to the future | [9, 6, 12, 2] | 9 |
| 45.93 | which river separates the bronx in new york city from man... | [9, 27, 12, 0] | 9 |

**Feature 1028** (z=2.38, 排他性=2.32x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 83.16 | marsupials are found in north america and australia | [9, 12, 0] | 9 |
| 68.68 | where is kruger national park in south africa | [9, 4, 25, 1] | 9 |
| 67.14 | one of the global hottest not spots of biodiversity in in... | [26, 13, 26, 4] | 26 |
| 58.96 | where are the giant redwoods located in california | [9, 27, 23, 0] | 9 |
| 57.12 | what is the current population of bora bora | [9, 18, 8, 4] | 9 |

#### 前缀 10（286 个样本，7 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 18 | 2.31 | 59.82x |
| 6796 | 2.27 | 2.72x |
| 6972 | 2.26 | 300.61x |
| 2487 | 2.23 | 3.61x |
| 5103 | 2.18 | 6.33x |
| 6529 | 2.16 | 3.32x |
| 435 | 2.02 | 2.28x |

**Feature 18** (z=2.31, 排他性=59.82x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 57.24 | what was tom hanks character name in castaway | [10, 10, 22, 0] | 10 |
| 48.97 | who has the rights to alice in wonderland | [10, 1, 7, 0] | 10 |
| 36.20 | who is the queen of hearts in alice in wonderland | [10, 2, 27, 0] | 10 |
| 34.43 | where was the last scene of goonies filmed | [10, 28, 25, 0] | 10 |
| 30.87 | where did they film woody the woodpecker movie | [10, 12, 20, 3] | 10 |

**Feature 6796** (z=2.27, 排他性=2.72x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 41.77 | when was the movie the wizard of oz made | [10, 1, 1, 0] | 10 |
| 34.44 | what happens in the movie the boy in the striped pajamas | [10, 22, 6, 1] | 10 |
| 31.88 | release date of ready player one movie in india | [10, 24, 15, 2] | 10 |
| 31.88 | tujhe dekha toh yeh jana sanam movie name | [10, 13, 28] | 10 |
| 31.35 | a description of the history and meaning of the 1st amend... | [16, 16, 7, 0] | 16 |

**Feature 6972** (z=2.26, 排他性=300.61x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 80.35 | war for the planet of the apes in india release | [10, 24, 7, 2] | 10 |
| 71.06 | who played caesar in planet of the apes war | [10, 24, 7, 2] | 10 |
| 67.56 | where was the war of the planet of the apes filmed | [10, 24, 7, 2] | 10 |
| 62.44 | when does planet of the apes come out 2017 | [10, 24, 7, 2] | 10 |
| 56.25 | when do subtitles start in the passion of the christ | [10, 24, 1] | 10 |

#### 前缀 12（184 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 1301 | 2.57 | 6.74x |

**Feature 1301** (z=2.57, 排他性=6.74x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 97.36 | who was the first executive president of guyana | [12, 6, 13, 5] | 12 |
| 81.61 | when did the movie karate kid come out | [12, 8, 6, 18] | 12 |
| 78.32 | what does the boy from karate kid look like now | [12, 23, 15, 1] | 12 |
| 69.26 | which brazilian player is known as the black diamond of f... | [12, 9, 4, 8] | 12 |
| 61.82 | who played the mom in the partridge family | [12, 17, 26, 0] | 12 |

#### 前缀 15（164 个样本，4 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 3369 | 2.99 | 2.79x |
| 4329 | 2.62 | 2.94x |
| 5065 | 2.09 | 1.98x |
| 1690 | 2.08 | 1.69x |

**Feature 3369** (z=2.99, 排他性=2.79x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 81.85 | who sang heard it thru the grapevine first | [15, 13, 11, 0] | 15 |
| 79.11 | where do you go to my lovely full version | [15, 13, 27, 0] | 15 |
| 76.31 | where do you you go to my lovely | [15, 13, 27, 0] | 15 |
| 75.74 | who is the book of galatians written to | [25, 18, 10, 0] | 25 |
| 69.25 | what is the song i heard it through the grapevine about | [15, 13, 11, 0] | 15 |

**Feature 4329** (z=2.62, 排他性=2.94x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 44.82 | country song when we get behind closed doors | [15, 18, 13, 0] | 15 |
| 38.44 | baby please don't go down to new orleans song | [15, 24, 28, 1] | 15 |
| 35.75 | the story of lover's leap in jamaica | [11, 16, 19, 15] | 11 |
| 35.69 | bruce springsteen we shall overcome the seeger sessions s... | [21, 29, 24] | 21 |
| 32.46 | what is the song ticket to ride about | [15, 14, 29, 0] | 15 |

**Feature 5065** (z=2.09, 排他性=1.98x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 77.36 | where does the last name roman come from | [7, 14, 8, 28] | 7 |
| 43.17 | what is the meaning of the name baba | [7, 14, 8, 27] | 7 |
| 42.42 | what is the short story the gift of the magi about | [8, 20, 21, 0] | 8 |
| 39.31 | the old man and the sea page count | [8, 20, 21, 1] | 8 |
| 36.94 | where have you been where are you going short story | [8, 20, 21, 2] | 8 |

#### 前缀 17（149 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 4187 | 2.03 | 2.26x |

**Feature 4187** (z=2.03, 排他性=2.26x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 93.71 | who propounded the concept of cross elasticity of demand | [17, 22, 4, 0] | 17 |
| 64.28 | difference between half rate and full rate in gsm | [17, 21, 6] | 17 |
| 58.38 | when does law of diminishing returns set in | [17, 22, 8, 1] | 17 |
| 58.25 | many many happy returns of the day meaning | [18, 22, 7, 0] | 18 |
| 53.92 | what is the short story the gift of the magi about | [8, 20, 21, 0] | 8 |

#### 前缀 19（243 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 1264 | 2.05 | 2.00x |

**Feature 1264** (z=2.05, 排他性=2.00x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 38.72 | who wrote he ain't heavy he's my brother lyrics | [19, 2, 15, 6] | 19 |
| 34.27 | who sings the song loving you is easy | [19, 10, 22, 8] | 19 |
| 32.85 | when was i look at the world poem written | [19, 23, 12, 15] | 19 |
| 29.32 | who did the original spirit in the sky | [19, 9, 19, 0] | 19 |
| 28.12 | who sang put the lime in the coconut in practical magic | [19, 2, 12, 10] | 19 |

#### 前缀 20（161 个样本，2 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 6632 | 2.34 | 4.31x |
| 371 | 2.24 | 1.84x |

**Feature 6632** (z=2.34, 排他性=4.31x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 72.32 | can a woman carry twins from two different fathers | [20, 15, 20, 2] | 20 |
| 62.08 | non-disjunction can occur during either anaphase i or ii | [20, 12, 7, 0] | 20 |
| 57.65 | what allows chyme to enter the small intestine | [20, 18, 1] | 20 |
| 57.36 | the biceps brachii muscle extends the arm at the elbow | [20, 25, 1, 2] | 20 |
| 54.80 | what is the average height of a chinese man | [17, 0, 2] | 17 |

**Feature 371** (z=2.24, 排他性=1.84x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 40.64 | glycogen and amylopectin are long chains of which simple ... | [24, 23, 4] | 24 |
| 34.61 | what is the name of the protease which is released in the... | [20, 24, 26] | 20 |
| 33.98 | in what part of the digestive tube do you expect the init... | [20, 18, 15] | 20 |
| 31.29 | what is the result of electrical stimulation to the retic... | [20, 3, 13, 2] | 20 |
| 28.93 | what is the rate limiting enzyme of kreb's cycle | [24, 23, 21] | 24 |

#### 前缀 21（217 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 1460 | 2.37 | 8.09x |

**Feature 1460** (z=2.37, 排他性=8.09x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 81.34 | bruce springsteen we shall overcome the seeger sessions s... | [21, 29, 24] | 21 |
| 81.29 | where does the story the great gatsby take place | [21, 12, 18] | 21 |
| 52.68 | who plays mr wilson in dennis the menace | [21, 3, 4, 1] | 21 |
| 49.57 | what is the second book in the alchemyst series | [21, 23, 28, 2] | 21 |
| 48.33 | who won the last fight in million dollar baby | [21, 4, 1, 4] | 21 |

#### 前缀 22（185 个样本，2 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 7113 | 2.47 | 2.30x |
| 5047 | 2.09 | 4.89x |

**Feature 7113** (z=2.47, 排他性=2.30x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 74.85 | where does the thames river begin and end | [22, 1, 0, 7] | 22 |
| 69.06 | where did the river thames start and end | [22, 1, 4, 1] | 22 |
| 65.40 | a town in west yorkshire on the river aire home to a rugb... | [22, 19, 27, 3] | 22 |
| 64.59 | where does the brazos river start and stop | [22, 1, 15, 0] | 22 |
| 58.58 | where does the paraguay river start and end | [22, 1, 2, 4] | 22 |

**Feature 5047** (z=2.09, 排他性=4.89x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 53.37 | should governments take actions to directly influence the... | [22, 0, 26, 0] | 22 |
| 50.78 | when was the first electronic cash register invented | [22, 23, 10, 7] | 22 |
| 35.93 | the fertile crescent is located between what two bodies o... | [4, 29, 25, 3] | 4 |
| 35.05 | in 1973 the first patient bill of rights was established | [22, 25, 16, 0] | 22 |
| 34.74 | texas uses what kind of voter registration system | [6, 25, 11, 4] | 6 |

#### 前缀 23（234 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 4064 | 2.15 | 18.65x |

**Feature 4064** (z=2.15, 排他性=18.65x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 76.20 | where did the idea of mickey mouse come from | [23, 20, 5, 0] | 23 |
| 70.89 | who does the voice of stewie family guy | [23, 23, 29, 1] | 23 |
| 61.74 | is kermit the frog part of sesame street | [23, 23, 5, 4] | 23 |
| 60.12 | what channel is big 10 network on fios | [23, 7, 1, 0] | 23 |
| 59.78 | what type of dog was laika the spacedog | [23, 1, 5, 3] | 23 |

#### 前缀 24（183 个样本，3 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 4035 | 2.15 | 16.67x |
| 4697 | 2.04 | 7.81x |
| 2281 | 2.03 | 4.59x |

**Feature 4035** (z=2.15, 排他性=16.67x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 76.86 | what is the difference between sodium chloride and lactat... | [24, 11, 18] | 24 |
| 65.99 | what is the rate limiting enzyme of kreb's cycle | [24, 23, 21] | 24 |
| 64.31 | glycogen and amylopectin are long chains of which simple ... | [24, 23, 4] | 24 |
| 55.35 | role of malonyl coa in fatty acid synthesis | [24, 23, 2, 2] | 24 |
| 51.59 | explain the role of glycogenin in glycogen synthesis | [22, 14, 7, 1] | 22 |

**Feature 4697** (z=2.04, 排他性=7.81x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 40.79 | where would a subcutaneous injection be made in the skin | [24, 18, 8, 6] | 24 |
| 29.55 | what are the advantages or disadvantages of using pure to... | [24, 18, 5] | 24 |
| 26.57 | what is the approximate volume of blood in your body | [24, 12, 11, 3] | 24 |
| 20.65 | when did they start vaccinating for whooping cough | [24, 28, 5, 7] | 24 |
| 19.85 | what features of muscle contraction can be determined fro... | [24, 18, 8, 9] | 24 |

**Feature 2281** (z=2.03, 排他性=4.59x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 52.70 | rain sleet or snow that contains a high concentration of ... | [24, 17, 0, 0] | 24 |
| 47.72 | unsaturated fats are comprised of lipids that contain | [24, 17, 9, 5] | 24 |
| 37.40 | what is the structural formula for methylethyl ether | [24, 22, 12, 18] | 24 |
| 36.67 | where is cellulose used in a plant cell | [24, 23, 7, 3] | 24 |
| 33.25 | active transport performs which function in a cell | [24, 25, 7, 0] | 24 |

#### 前缀 25（174 个样本，4 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 2236 | 2.80 | 12.14x |
| 6437 | 2.27 | 74.48x |
| 5581 | 2.13 | 23.29x |
| 2612 | 2.06 | 4.44x |

**Feature 2236** (z=2.80, 排他性=12.14x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 90.84 | what was the elevation of the land where coronado crossed... | [25, 10, 2, 9] | 25 |
| 80.72 | where is mount st. helens located on a map | [25, 20, 5] | 25 |
| 69.63 | what is the big gold dome in jerusalem | [25, 20, 6, 1] | 25 |
| 68.16 | when did the united states acquired puerto rico | [11, 25, 5, 5] | 11 |
| 67.92 | the road that connects the tombs is called | [25, 20, 8, 4] | 25 |

**Feature 6437** (z=2.27, 排他性=74.48x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 41.73 | joyce a portrait of the artist as a young man summary | [25, 0, 17, 1] | 25 |
| 38.54 | where is mount st. helens located on a map | [25, 20, 5] | 25 |
| 35.34 | do you have to read james bond books in order | [25, 0, 15, 1] | 25 |
| 32.49 | justify the title of the novel sense and sensibility | [25, 0, 14, 6] | 25 |
| 31.73 | who is considered the first great modern architect | [25, 19, 13, 1] | 25 |

**Feature 5581** (z=2.13, 排他性=23.29x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 55.27 | who expanded the territory of china during the qing dynasty | [25, 21, 13, 1] | 25 |
| 52.08 | where did the rulers of the qing dynasty originate | [25, 21, 13, 1] | 25 |
| 41.29 | who was the first missionary out of jerusalem | [25, 21, 20, 6] | 25 |
| 40.09 | who built the first temple for god in jerusalem | [25, 21, 20, 0] | 25 |
| 36.40 | explorer who led an early voyage to the coast of newfound... | [25, 11, 15, 1] | 25 |

#### 前缀 28（276 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 7530 | 2.53 | 1.66x |

**Feature 7530** (z=2.53, 排他性=1.66x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 22.27 | what states were most affected by the dust bowl | [28, 12, 14, 0] | 28 |
| 20.98 | when are you considered under the poverty line | [28, 12, 16, 1] | 28 |
| 20.93 | what were the two causes of the dust bowl | [28, 12, 14, 0] | 28 |
| 19.94 | which of the following writers contributed to the harlem ... | [28, 21, 24, 0] | 28 |
| 19.46 | who participated in the columbian exchange and how did it... | [28, 5, 3, 0] | 28 |

#### 前缀 29（149 个样本，1 个路由特征）

| 特征ID | z-score | 排他性 |
|--------|---------|--------|
| 4357 | 2.11 | 6.23x |

**Feature 4357** (z=2.11, 排他性=6.23x) 的 top-5 激活样本：

| 激活值 | Query | DocID | 第一前缀 |
|--------|-------|-------|---------|
| 40.71 | where do characters live in this is us | [10, 24, 0, 0] | 10 |
| 34.60 | who was the girl in the video brenda got a baby | [29, 5, 28] | 29 |
| 24.50 | who wrote the song if i were a boy | [29, 5, 23, 3] | 29 |
| 23.00 | the dj got us falling in love again | [29, 5, 16, 5] | 29 |
| 21.03 | janet jackson some one to call my lover | [29, 9, 28] | 29 |


## 3. 每个前缀的 Top-10 特异特征

按 z-score 降序排列，展示每个前缀最特异的特征。

### 前缀 0（248 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 5018 | 2.35 | 44.8% | 4.7% |
| 2 | 5829 | 2.18 | 44.4% | 7.6% |
| 3 | 6163 | 2.03 | 53.6% | 12.8% |
| 4 | 1975 | 2.02 | 45.6% | 5.8% |
| 5 | 1868 | 2.01 | 44.8% | 7.5% |

### 前缀 1（188 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 2125 | 3.70 | 70.7% | 6.3% |
| 2 | 1717 | 2.37 | 45.7% | 4.7% |
| 3 | 7505 | 2.26 | 33.5% | 2.4% |
| 4 | 7451 | 2.13 | 47.9% | 7.1% |
| 5 | 3569 | 2.05 | 42.6% | 5.0% |

### 前缀 2（248 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 7568 | 3.14 | 83.9% | 12.0% |
| 2 | 3407 | 2.68 | 54.8% | 4.0% |
| 3 | 817 | 2.43 | 60.1% | 7.5% |
| 4 | 2809 | 2.31 | 46.0% | 5.0% |
| 5 | 2069 | 2.28 | 52.0% | 6.1% |
| 6 | 1321 | 2.26 | 48.0% | 4.7% |
| 7 | 5630 | 2.24 | 41.5% | 3.4% |
| 8 | 7969 | 2.17 | 61.7% | 9.2% |
| 9 | 4742 | 2.16 | 57.7% | 11.8% |
| 10 | 4521 | 2.15 | 51.6% | 7.7% |

### 前缀 3（323 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 5672 | 2.01 | 46.1% | 7.1% |

### 前缀 6（259 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 6104 | 2.08 | 36.7% | 5.3% |

### 前缀 8（147 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 7760 | 3.59 | 57.1% | 3.7% |
| 2 | 2680 | 2.50 | 53.7% | 5.9% |
| 3 | 3806 | 2.43 | 44.9% | 5.4% |
| 4 | 7154 | 2.29 | 55.1% | 7.8% |
| 5 | 3002 | 2.24 | 66.7% | 14.5% |
| 6 | 6864 | 2.23 | 65.3% | 11.1% |
| 7 | 2602 | 2.18 | 56.5% | 7.2% |

### 前缀 9（155 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 7771 | 2.99 | 56.1% | 6.2% |
| 2 | 5059 | 2.48 | 34.8% | 2.9% |
| 3 | 1028 | 2.38 | 52.9% | 10.4% |
| 4 | 6966 | 2.11 | 40.0% | 3.8% |

### 前缀 10（286 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 18 | 2.31 | 38.5% | 2.9% |
| 2 | 6796 | 2.27 | 51.0% | 4.9% |
| 3 | 6972 | 2.26 | 36.7% | 2.2% |
| 4 | 2487 | 2.23 | 64.7% | 11.3% |
| 5 | 5103 | 2.18 | 37.8% | 3.2% |
| 6 | 6529 | 2.16 | 64.7% | 10.3% |
| 7 | 435 | 2.02 | 64.0% | 12.3% |

### 前缀 12（184 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 1301 | 2.57 | 78.3% | 17.7% |

### 前缀 15（164 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 3369 | 2.99 | 78.0% | 9.3% |
| 2 | 4329 | 2.62 | 62.2% | 7.1% |
| 3 | 5030 | 2.16 | 56.7% | 6.1% |
| 4 | 5751 | 2.11 | 84.8% | 22.0% |
| 5 | 5065 | 2.09 | 46.3% | 4.3% |
| 6 | 1690 | 2.08 | 43.9% | 4.5% |

### 前缀 17（149 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 4187 | 2.03 | 31.5% | 4.5% |

### 前缀 19（243 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 1264 | 2.05 | 37.9% | 3.3% |

### 前缀 20（161 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 6632 | 2.34 | 47.8% | 8.0% |
| 2 | 371 | 2.24 | 30.4% | 2.3% |

### 前缀 21（217 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 1460 | 2.37 | 47.5% | 5.8% |

### 前缀 22（185 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 7113 | 2.47 | 39.5% | 5.1% |
| 2 | 5047 | 2.09 | 37.8% | 5.0% |

### 前缀 23（234 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 4064 | 2.15 | 30.8% | 2.7% |

### 前缀 24（183 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 4035 | 2.15 | 32.8% | 5.5% |
| 2 | 4697 | 2.04 | 19.1% | 0.9% |
| 3 | 2281 | 2.03 | 29.0% | 2.9% |

### 前缀 25（174 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 2236 | 2.80 | 39.7% | 3.6% |
| 2 | 6437 | 2.27 | 22.4% | 2.0% |
| 3 | 1269 | 2.17 | 43.1% | 4.3% |
| 4 | 5581 | 2.13 | 24.7% | 2.2% |
| 5 | 2612 | 2.06 | 29.9% | 2.9% |

### 前缀 28（276 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 7530 | 2.53 | 60.1% | 7.3% |

### 前缀 29（149 个样本）

| 排名 | 特征ID | z-score | 激活率(该前缀) | 激活率(全局) |
|------|--------|---------|--------------|------------|
| 1 | 4357 | 2.11 | 29.5% | 2.1% |


## 4. 前缀间共享特征分析

有些特征在多个前缀上都有高 z-score，可能是更通用的语义特征。

- 仅在 1 个前缀上显著的特征（独占特征）: **70** 个
- 在 >= 3 个前缀上显著的特征（共享特征）: **0** 个
- 在 >= 5 个前缀上显著的特征（高度共享）: **0** 个

### 独占特征的前缀分布

| 前缀 | 独占特征数 |
|------|-----------|
| 0 | 5 |
| 1 | 5 |
| 2 | 15 |
| 3 | 1 |
| 6 | 1 |
| 8 | 7 |
| 9 | 4 |
| 10 | 7 |
| 12 | 1 |
| 15 | 6 |
| 17 | 1 |
| 19 | 1 |
| 20 | 2 |
| 21 | 1 |
| 22 | 2 |
| 23 | 1 |
| 24 | 3 |
| 25 | 5 |
| 28 | 1 |
| 29 | 1 |

## 5. 总结

### 关键发现

1. **前缀路由特征存在**。共找到 67 个特征，它们对特定前缀高度特异（z > 2.0）且排他性强（排他性 > 1.5x）。
2. 路由特征最多的是前缀 2（15 个），说明该前缀在 SAE 特征空间中有最清晰的 "签名"。
3. 独占特征（仅 1 个前缀显著）有 70 个，说明模型内部确实存在针对特定前缀的离散路由信号。
4. 共享特征（>= 3 个前缀显著）有 0 个，这些可能是更通用的语义编码（如 "is a question about X" 的特征）。

### 含义

- 模型内部确实存储了 DocID 层次结构的离散路由信号，表现为 SAE 特征空间中的前缀特异激活模式。
- 当模型准备生成某个前缀时，对应的路由特征会被激活，引导解码器走向正确的分支。
- 路由特征的存在支持了 "生成式检索模型内部维护了一棵隐式决策树" 的假设。