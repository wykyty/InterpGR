# SAE 特征人工检查表

**SAE**: out/sae_train_8x/layer_12 (d_sae=8192, k=100)
**数据**: dataset/nq320k/dev.json (7830 样本)
**抽取方式**: 从激活频率 >5% 的特征中随机抽 25 个 (seed=42)

**标注指南**: 对每个特征，查看 top-10 激活样本的 query 和 DocID，判断特征类型：

- **语义特征**: top 样本的主题/领域一致（如都关于体育、地理、电影等）
- **路由特征**: top 样本的 DocID 前缀/路径一致（如都映射到某个 doc_id range）
- **其他**: 无明显规律

---

## Feature 97

**激活频率**: 9.7%（759/7830 个样本激活）
**最大激活值**: 63.4774, **平均激活值(激活时)**: 9.0485

**标注**: 语义特征 -- 影视/娱乐/名人

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 5957 | 63.4774 | where was the tv show in the heat of the night filmed | [2, 9, 27] |
| 2 | 7284 | 59.3890 | who plays drew's boyfriend on the night shift | [2, 19, 27] |
| 3 | 7791 | 52.6840 | how many wins does tiger woods have on the pga tour | [13, 9, 29, 0] |
| 4 | 6666 | 46.6637 | can you buy a gun on sunday in wv | [6, 13, 4, 1] |
| 5 | 2325 | 45.5820 | who plays whitey bulger's girlfriend in black mass | [10, 12, 16, 14] |
| 6 | 2980 | 43.1700 | who played in the celebrity all star game 2018 | [5, 8, 29] |
| 7 | 6917 | 41.1648 | short blonde hair girl from orange is the new black | [13, 1, 23, 1] |
| 8 | 2948 | 39.3202 | what season does bart bass die in gossip girl | [21, 27, 29, 3] |
| 9 | 6414 | 39.0398 | when does the new gotham season come out | [2, 13, 29] |
| 10 | 1750 | 37.4997 | where was the original flight of the phoenix filmed | [21, 12, 19, 1] |

## Feature 590

**激活频率**: 8.1%（635/7830 个样本激活）
**最大激活值**: 45.3856, **平均激活值(激活时)**: 6.2548

**标注**: 路由特征 -- DocID [5,3,11,0] = Game of Thrones 相关文档，4/10 条都命中同一 DocID

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 3890 | 45.3856 | when did daylight saving time start in texas | [28, 21, 14] |
| 2 | 3655 | 31.7872 | when game of thrones season 7 episode 8 release date | [5, 3, 11, 0] |
| 3 | 5891 | 26.2905 | when does season 7 game of thrones dvd release | [5, 3, 11, 0] |
| 4 | 4230 | 25.9973 | how many episodes in game if thrones season 7 | [5, 3, 11, 0] |
| 5 | 5577 | 24.3536 | how many episodes in series 7 of game of thrones are there | [5, 3, 11, 0] |
| 6 | 4112 | 22.2211 | what us cities have a population of 1 million | [26, 12, 3, 0] |
| 7 | 1198 | 21.1666 | when did earth's atmosphere change due to living organisms | [28, 18, 7, 1] |
| 8 | 383 | 20.9595 | when is star vs the forces of evil coming back 2018 | [27, 17, 15, 0] |
| 9 | 4786 | 20.0023 | when does the sun come up in the summer | [6, 15, 0, 0] |
| 10 | 7347 | 19.1991 | what type of boundary was the mexico earthquake | [5, 3, 6, 0] |

## Feature 597

**激活频率**: 6.6%（520/7830 个样本激活）
**最大激活值**: 89.2376, **平均激活值(激活时)**: 9.9847

**标注**: 语义特征 -- 音乐/歌曲（top-10 中 6 条关于歌曲/歌手）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6261 | 89.2376 | who recorded i can't help falling in love with you | [19, 9, 9, 1] |
| 2 | 1192 | 72.2143 | burt bacharach raindrops keep falling on my head | [19, 5, 4, 0] |
| 3 | 5586 | 62.7042 | why are there 64 teams in the ncaa tournament | [5, 27, 5, 0] |
| 4 | 7484 | 61.3965 | when does the school year end in america | [6, 15, 9, 0] |
| 5 | 796 | 58.4014 | who has the most 70 point games in nba history | [11, 24, 5, 0] |
| 6 | 2251 | 58.2626 | oldest recording of house of the rising sun | [19, 9, 0, 28] |
| 7 | 5749 | 57.8819 | who sang rain drops keep falling on my head | [19, 5, 4, 0] |
| 8 | 5910 | 57.1278 | where was i can only imagine first sang | [19, 5, 5, 0] |
| 9 | 1211 | 53.6189 | when is i can only imagine coming out | [10, 24, 10, 0] |
| 10 | 4748 | 50.7599 | what are the layers of the earth and its definition | [26, 13, 6, 0] |

## Feature 738

**激活频率**: 7.7%（606/7830 个样本激活）
**最大激活值**: 49.6202, **平均激活值(激活时)**: 7.5711

**标注**: 语义特征 -- 体育（NFL/NBA/网球：Tom Brady, LeBron James, Falcons Super Bowl, Rafael Nadal）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 21 | 49.6202 | where are the giant redwoods located in california | [9, 27, 23, 0] |
| 2 | 1641 | 39.8196 | who was the pitcher who broke his arm | [7, 26, 26, 9] |
| 3 | 6835 | 38.7887 | how long has tom brady been the patriots quarterback | [7, 17, 3, 1] |
| 4 | 120 | 38.2356 | who scored the most points in their nba career | [7, 13, 18, 0] |
| 5 | 3141 | 37.1910 | when did the falcons win the super bowl | [7, 13, 5, 0] |
| 6 | 1625 | 36.6581 | where is jj's husband on criminal minds | [21, 26, 5, 2] |
| 7 | 1020 | 35.1907 | where did rafael nadal win his first tennis title | [7, 8, 15, 3] |
| 8 | 373 | 34.7638 | how many points did lebron james scored in his career | [7, 13, 18, 0] |
| 9 | 5340 | 32.9200 | who played john coffey in the movie the green mile | [21, 18, 3, 8] |
| 10 | 6119 | 32.5847 | where are the washington redskins based out of | [7, 13, 24, 0] |

## Feature 1255

**激活频率**: 5.6%（442/7830 个样本激活）
**最大激活值**: 32.7891, **平均激活值(激活时)**: 8.6427

**标注**: 其他 -- 主题杂乱（yo gabba gabba, 放射性废物, 立克次体, 情歌, 薯蓣, 奥运会）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6343 | 32.7891 | who plays the characters in yo gabba gabba | [23, 14, 19, 3] |
| 2 | 5461 | 31.5778 | high-level radioactive waste requires deep burial because it is explosive | [28, 26, 8, 0] |
| 3 | 658 | 31.0737 | which of the following is caused by a member of the rickettsias | [22, 28, 21] |
| 4 | 4232 | 31.0662 | think i love you from head to toe | [19, 24, 23, 0] |
| 5 | 6447 | 31.0550 | where is the food stored in a yam plant | [26, 28, 12, 4] |
| 6 | 319 | 30.6789 | when are the olympics going to be in canada | [14, 27, 1, 0] |
| 7 | 1984 | 30.0364 | who are the characters in yo gabba gabba | [23, 14, 19, 3] |
| 8 | 5351 | 30.0004 | iss pyaar ko kya naam doon3 full episodes | [2, 28, 2, 2] |
| 9 | 5064 | 29.5987 | what is the full form of ib board | [26, 26, 18, 10] |
| 10 | 1590 | 28.6759 | who wrote cant get you out of my head lyrics | [19, 27, 9, 2] |

## Feature 1621

**激活频率**: 10.5%（819/7830 个样本激活）
**最大激活值**: 74.8184, **平均激活值(激活时)**: 11.1077

**标注**: 语义特征 -- 美式橄榄球/NFL（passing yards, quarterbacks, receiving touchdown, catches, rushing yards）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 700 | 74.8184 | who got the most passing yards in the nfl | [14, 24, 16, 0] |
| 2 | 3656 | 71.3913 | how many quarterbacks have a receiving touchdown in the superbowl | [14, 24, 15, 3] |
| 3 | 293 | 68.6040 | who has the most catches in nfl history | [14, 16, 27] |
| 4 | 6689 | 61.3329 | who has the most receiving yards in the nfl history | [11, 24, 5, 6] |
| 5 | 2280 | 54.1493 | who has the most yards per carry in nfl history | [14, 24, 16, 6] |
| 6 | 5104 | 50.6365 | who has the most rushing yards in a super bowl | [14, 24, 15, 3] |
| 7 | 7696 | 49.7000 | when was the last time an nba backboard broken | [0, 27, 5, 0] |
| 8 | 4077 | 49.3679 | who was the chicago bears quarterback last year | [11, 24, 25, 4] |
| 9 | 283 | 48.8015 | do ghanaians need visa to go to singapore | [14, 14, 19, 1] |
| 10 | 4036 | 47.7027 | when did the black death end in england | [16, 14, 24, 1] |

## Feature 1868

**激活频率**: 7.5%（589/7830 个样本激活）
**最大激活值**: 65.2812, **平均激活值(激活时)**: 10.4145

**标注**: 其他 -- 杂题（腐殖质, 床的流行, 闪存写入次数, 算盘, 圣诞树顶饰）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 1763 | 65.2812 | why top soil has most amount of humus | [0, 25, 25] |
| 2 | 1658 | 58.9898 | when did beds become popular in france and germany | [0, 13, 4, 0] |
| 3 | 4779 | 56.0760 | where did dry as a bone come from | [1, 27, 24, 3] |
| 4 | 7819 | 53.9240 | how many writes does a flash drive have | [0, 2, 18, 3] |
| 5 | 3797 | 51.1334 | how many different types of abacus are there | [0, 17, 22, 7] |
| 6 | 6559 | 48.0017 | do you cut cards to the left or right | [0, 27, 19] |
| 7 | 3801 | 47.6225 | what is the most popular item to put on top of a christmas tree | [0, 10, 6, 2] |
| 8 | 6606 | 43.7578 | when was the abacus invented in ancient china | [0, 17, 22, 7] |
| 9 | 5385 | 40.3082 | what is the difference between italian beef and french dip | [13, 23, 0, 1] |
| 10 | 2096 | 40.2233 | whats the difference between tomato paste and tomato puree | [0, 28, 9, 0] |

## Feature 1993

**激活频率**: 6.9%（539/7830 个样本激活）
**最大激活值**: 113.3120, **平均激活值(激活时)**: 11.7256

**标注**: 语义特征 -- 学术/历史/科学（独立宣言, 磁场, 火山喷发, 格式塔心理学, 季风贸易）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6754 | 113.3120 | when did the continental congress vote to adopt the declaration of independence | [16, 24, 24, 1] |
| 2 | 2223 | 99.0841 | justify the title of the novel sense and sensibility | [25, 0, 14, 6] |
| 3 | 1758 | 87.4627 | what is earth's magnetic field responsible for | [28, 7, 8, 0] |
| 4 | 684 | 79.4232 | when did cat on a hot tin roof take place | [25, 0, 8, 0] |
| 5 | 6587 | 68.8959 | where do the elves go on the boat in lord of the rings | [1, 12, 20, 0] |
| 6 | 2970 | 65.5068 | today gestalt psychology ideas are part of which branch of psychology | [25, 19, 16, 0] |
| 7 | 5490 | 62.4241 | type of volcanic eruption is mostly observed in the philippines | [28, 7, 16] |
| 8 | 5237 | 56.9398 | when was the last time new zealand had an earthquake | [16, 14, 9, 4] |
| 9 | 3400 | 55.2671 | when did canada sign the un declaration of indigenous rights | [16, 15, 9, 2] |
| 10 | 1494 | 52.7868 | what purpose did seasonal monsoon winds have on trade | [28, 7, 9, 3] |

## Feature 2317

**激活频率**: 14.1%（1103/7830 个样本激活）
**最大激活值**: 39.2168, **平均激活值(激活时)**: 7.9746

**标注**: 语义特征 -- 短语/习语起源（great scott, hat trick, bless you, as the crow flies）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6414 | 39.2168 | when does the new gotham season come out | [2, 13, 29] |
| 2 | 7011 | 37.6091 | where did the expression great scott come from | [11, 22, 1, 26] |
| 3 | 4202 | 34.8581 | when did the study of media effects begin | [25, 19, 4, 4] |
| 4 | 2441 | 31.9604 | what is the advantage of genetic recombination as a mode of reproduction in b... | [0, 20, 20, 1] |
| 5 | 4748 | 31.5706 | what are the layers of the earth and its definition | [26, 13, 6, 0] |
| 6 | 1170 | 31.0352 | where does the phrase hat trick come from | [11, 22, 1, 21] |
| 7 | 7416 | 30.2599 | where did bless you when you sneeze come from | [18, 17, 12, 3] |
| 8 | 898 | 28.8663 | how did ww2 end in europe and the pacific | [11, 25, 1, 4] |
| 9 | 2105 | 27.9165 | when does precipitate form in a chemical reaction | [0, 25, 0, 0] |
| 10 | 2906 | 27.8695 | where does the expression as the crow flies come from | [4, 17, 22, 15] |

## Feature 2388

**激活频率**: 6.2%（486/7830 个样本激活）
**最大激活值**: 65.1846, **平均激活值(激活时)**: 10.5075

**标注**: 其他 -- 杂题（摇滚名人堂, 商会目标, 感恩节比赛, AARP政治立场）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 426 | 65.1846 | when was the rock and roll hall of fame built in cleveland | [4, 26, 4, 0] |
| 2 | 5387 | 58.6781 | what are the goals of the us chamber of commerce | [4, 20, 11, 4] |
| 3 | 2465 | 51.5573 | number of employees in the department of health and human services | [26, 19, 3, 4] |
| 4 | 2909 | 50.8215 | who are the dallas cowboys playing on thanksgiving | [11, 24, 15, 1] |
| 5 | 5934 | 49.4507 | where is the second largest mall in america | [14, 25, 11, 4] |
| 6 | 7651 | 48.7443 | where does aarp fall on the political spectrum | [23, 11, 9, 4] |
| 7 | 3058 | 45.1075 | regional health information organizations do all of the following except | [26, 8, 1, 10] |
| 8 | 4506 | 43.2433 | when did the cowboys start playing on thanksgiving day | [11, 24, 15, 1] |
| 9 | 2206 | 41.3239 | where was the dukes of hazzard show filmed | [3, 19, 26, 1] |
| 10 | 1986 | 40.9383 | who won the national championship in volleyball 2017 | [11, 24, 20, 1] |

## Feature 2502

**激活频率**: 6.8%（536/7830 个样本激活）
**最大激活值**: 97.2899, **平均激活值(激活时)**: 12.3368

**标注**: 其他 -- 主题杂乱（金发碧眼起源, 条纹睡衣男孩, 汽车年份, 澳大利亚婚姻法）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 4450 | 97.2899 | where does blonde hair green eyes come from | [0, 5, 26] |
| 2 | 6667 | 74.8616 | when does the boy in the striped pajamas take place | [8, 14, 19, 0] |
| 3 | 5867 | 69.3702 | when does new model year start for cars | [6, 13, 2, 3] |
| 4 | 2946 | 64.5877 | when did the daffodil become the emblem of wales | [16, 18, 0, 10] |
| 5 | 2385 | 62.8514 | marsupials are found in north america and australia | [9, 12, 0] |
| 6 | 7453 | 61.2781 | what is the legal age for marriage in australia | [6, 12, 11, 0] |
| 7 | 27 | 60.4448 | who plays peter in what we do in the shadows | [2, 1, 8, 0] |
| 8 | 4576 | 59.2620 | is aluminium a ferrous or non ferrous metal | [0, 23, 28] |
| 9 | 747 | 52.8574 | where does the movie mothers day take place | [2, 8, 0] |
| 10 | 6040 | 51.1120 | are male and female praying mantis different colors | [22, 7, 0, 7] |

## Feature 2541

**激活频率**: 9.2%（720/7830 个样本激活）
**最大激活值**: 56.0177, **平均激活值(激活时)**: 8.7875

**标注**: 其他 -- 杂题（女王王冠, Mac OS, Riverdale, 本田思域, 权力的游戏）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 7090 | 56.0177 | where did the queen's crown come from | [11, 22, 2, 8] |
| 2 | 1924 | 50.4070 | what is the current mac os operating system | [22, 15, 8, 0] |
| 3 | 6361 | 49.2734 | when does the new season of lost in space come out | [8, 27, 10, 0] |
| 4 | 169 | 41.9857 | does archie end up with betty or veronica in riverdale | [21, 27, 27] |
| 5 | 1038 | 40.4834 | where did the royal family go to school | [4, 16, 0] |
| 6 | 7018 | 40.2708 | what episode of pll does jenna get her sight back | [21, 27, 26, 2] |
| 7 | 7019 | 37.0990 | when did aint get added to the dictionary | [11, 22, 22, 0] |
| 8 | 753 | 35.6127 | is a 2005 honda civic front wheel drive | [0, 17, 21] |
| 9 | 810 | 34.7496 | who plays the dragon queen from game of thrones | [11, 25, 17, 0] |
| 10 | 167 | 33.7135 | where is mom and dad or where are mom and dad | [3, 22, 15, 2] |

## Feature 2596

**激活频率**: 7.0%（551/7830 个样本激活）
**最大激活值**: 48.7183, **平均激活值(激活时)**: 8.4814

**标注**: 语义特征 -- 宗教/教育（Lords Prayer 出现 3 次, 学期时间, 圣经, 澳大利亚大学）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6972 | 48.7183 | when was thine is the kingdom added to lord's prayer | [18, 6, 13, 0] |
| 2 | 4350 | 45.4557 | when does first quarter end in middle school | [6, 25, 24, 1] |
| 3 | 7484 | 44.3952 | when does the school year end in america | [6, 15, 9, 0] |
| 4 | 4760 | 39.2633 | how many us states are commonwealths and which states are they | [18, 23, 20, 1] |
| 5 | 4580 | 37.4328 | where is lord's prayer found in bible | [18, 6, 13, 0] |
| 6 | 44 | 35.6549 | where in the bible can i find the lord's prayer | [18, 6, 13, 0] |
| 7 | 4740 | 32.6201 | the united states was the first country in the world to employ a system of go... | [28, 28, 8, 4] |
| 8 | 985 | 31.0046 | when did university stop being free in australia | [6, 26, 25, 3] |
| 9 | 5867 | 28.5814 | when does new model year start for cars | [6, 13, 2, 3] |
| 10 | 1042 | 27.3732 | who were the first labor unions formed by | [16, 21, 4, 0] |

## Feature 2804

**激活频率**: 18.6%（1456/7830 个样本激活）
**最大激活值**: 87.4549, **平均激活值(激活时)**: 11.5177

**标注**: 语义特征 -- NBA篮球/宗教（NBA 三分线出现 4 次, 方舟/十诫/塔木德）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6470 | 87.4549 | nba record for most double doubles in a season | [6, 9, 20] |
| 2 | 4641 | 77.2147 | what is bound in heaven will be bound on earth | [18, 9, 16, 1] |
| 3 | 7020 | 75.7066 | when did they start 3 pointers in basketball | [6, 9, 1] |
| 4 | 585 | 75.0475 | when did the nba add the three point line | [6, 9, 1] |
| 5 | 606 | 69.1220 | what nba player has scored the most 3 pointers | [11, 24, 5, 2] |
| 6 | 971 | 68.9920 | when did the nba create the 3 point line | [6, 9, 1] |
| 7 | 4803 | 68.2710 | where was the ark of the covenant built | [18, 6, 22, 0] |
| 8 | 2940 | 63.0428 | what written material is included in the talmud | [14, 20, 12, 1] |
| 9 | 270 | 61.5726 | where is the tablet of the ten commandments | [18, 20, 5, 0] |
| 10 | 3884 | 61.3543 | when did the botswana currency first come into circulation | [23, 9, 4, 0] |

## Feature 3333

**激活频率**: 10.3%（806/7830 个样本激活）
**最大激活值**: 60.4539, **平均激活值(激活时)**: 9.6923

**标注**: 语义特征 -- 电影/电视/漫威（Captain America, Walking Dead, Black Panther, Iron Man 2, Harry Potter）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 4073 | 60.4539 | when does the star movie come out in canada | [10, 24, 23, 1] |
| 2 | 5765 | 57.4392 | the first element on the periodic table is | [28, 7, 27] |
| 3 | 466 | 57.0423 | when does clark meet the flash in smallville | [27, 28, 19, 1] |
| 4 | 153 | 55.4984 | what order do the captain america movies go in | [16, 11, 10, 0] |
| 5 | 1876 | 53.9947 | when do the walking dead comics come out | [27, 14, 10, 2] |
| 6 | 2919 | 50.4722 | how long is black panther going to be out | [27, 26, 25, 0] |
| 7 | 329 | 49.1952 | who plays justin hammer in iron man 2 | [7, 11, 14, 0] |
| 8 | 2699 | 45.2061 | what is a another name for the water cycle | [28, 7, 9, 6] |
| 9 | 3986 | 44.5176 | where did the name black death come from | [16, 14, 24, 0] |
| 10 | 7145 | 37.4903 | who is darrell brother in the walking dead | [27, 11, 5, 1] |

## Feature 3399

**激活频率**: 6.5%（510/7830 个样本激活）
**最大激活值**: 94.8858, **平均激活值(激活时)**: 11.9163

**标注**: 语义特征 -- 社交媒体/名人/财富（Instagram 粉丝出现 5 次, 最赚钱的电影/运动员）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 4153 | 94.8858 | who is the most followed user on instagram 2017 | [11, 7, 14, 0] |
| 2 | 7004 | 85.7175 | who has the most followers on the instagram | [11, 7, 14, 0] |
| 3 | 5997 | 80.5524 | who has the most followers in the world on instagram | [11, 7, 14, 0] |
| 4 | 5216 | 77.4229 | who makes the most money in a film | [16, 11, 25, 0] |
| 5 | 7594 | 67.7746 | who founded amazon where is the headquarters of amazon | [14, 7, 17, 1] |
| 6 | 2224 | 61.0071 | top 10 most viewed youtube videos in india | [3, 12, 25, 0] |
| 7 | 4151 | 58.6804 | who has won the cma entertainer of the year the most | [5, 17, 15, 0] |
| 8 | 6759 | 58.3449 | when does copyright start do i have to register the work with the government | [26, 27, 25, 1] |
| 9 | 2166 | 55.8347 | who makes the most money in professional sports | [14, 6, 1, 6] |
| 10 | 660 | 55.4123 | who has most followers on instagram in world | [14, 7, 5, 6, 0] |

## Feature 3675

**激活频率**: 7.2%（567/7830 个样本激活）
**最大激活值**: 40.9288, **平均激活值(激活时)**: 8.3071

**标注**: 其他 -- 杂题（无尾狗, 人权宣言, 爱尔兰联邦, 共和国日嘉宾, H1N1病毒载体）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6994 | 40.9288 | what dog breeds are born without a tail | [11, 22, 15, 17] |
| 2 | 4505 | 39.8303 | who wrote the declaration of man and citizen | [16, 4, 26, 1] |
| 3 | 2590 | 35.0344 | why is ireland not a member of the commonwealth | [16, 0, 15, 0] |
| 4 | 5932 | 34.4241 | who was the chief guest of 2014 republic day | [4, 9, 18, 0] |
| 5 | 2516 | 32.5876 | which animal is the carrier of the h1n1 virus | [24, 15, 11, 1] |
| 6 | 541 | 29.8950 | who were the parties to the atlantic charter in 1941 what were their eight co... | [16, 12, 20, 1] |
| 7 | 783 | 29.2688 | who has been appointed as the election commissioner of india | [14, 9, 4, 2] |
| 8 | 750 | 27.5813 | when did russia join the world economic forum | [16, 15, 2, 6] |
| 9 | 4601 | 27.3795 | which of the follow statements is true of the eastern orthodox churches | [16, 1, 25, 3] |
| 10 | 6861 | 27.1493 | when do new episodes of boruto come out | [27, 12, 14, 1] |

## Feature 3800

**激活频率**: 9.2%（721/7830 个样本激活）
**最大激活值**: 58.0267, **平均激活值(激活时)**: 9.8999

**标注**: 其他 -- 杂题（萨尔瓦多国旗, 格鲁吉亚首都, 可持续建筑, 耶稣会大学）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 2384 | 58.0267 | what do the colors on the el salvador flag mean | [11, 22, 12, 0] |
| 2 | 6126 | 57.4835 | european journal of work and organizational psychology ranking | [11, 10, 25, 5] |
| 3 | 5726 | 51.0896 | where did the last name salgado come from | [11, 23, 1, 16] |
| 4 | 1014 | 47.9427 | capital of georgia the former soviet republic 7 letters | [14, 22, 4, 8] |
| 5 | 4523 | 44.7484 | sustainable architecture and simulation modelling dublin institute of technology | [11, 19, 16, 7] |
| 6 | 2199 | 44.5515 | where does the national security council get their information | [4, 20, 3, 1] |
| 7 | 372 | 42.2112 | sending money home to the native country is an example of | [11, 23, 21, 13] |
| 8 | 5128 | 40.3652 | bosnia and herzegovina croatia macedonia and slovenia all used to be parts of | [14, 3, 12, 2] |
| 9 | 4212 | 37.6411 | who is buried in the tomb of the unknown soldiers | [11, 25, 12, 1] |
| 10 | 5633 | 37.3878 | list of jesuit universities in the united states | [14, 22, 28, 21] |

## Feature 4185

**激活频率**: 7.9%（615/7830 个样本激活）
**最大激活值**: 36.2184, **平均激活值(激活时)**: 5.9860

**标注**: 其他 -- 杂题（互联网, NBA合同, 哈利波特, 民族主义, 安全带法律, 高尔夫术语）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 5141 | 36.2184 | the internet is part of the world wide web | [28, 16, 0, 1] |
| 2 | 7704 | 26.9286 | who has the highest paid contract in the nba | [16, 11, 12, 2] |
| 3 | 4152 | 25.0448 | who played the jewish man in coming to america | [4, 6, 20, 2] |
| 4 | 2659 | 24.7188 | how did nationalism influence the country of india | [16, 26, 12, 2] |
| 5 | 6993 | 23.1092 | the cast of harry potter the goblet of fire | [10, 24, 19, 2] |
| 6 | 3481 | 22.9712 | when did seat belts become law in ontario | [11, 1, 19, 23] |
| 7 | 1914 | 21.4535 | where does the word fore in golf originated | [18, 17, 12, 2] |
| 8 | 5809 | 21.1505 | what is the current rate of interest on ppf | [26, 17, 9, 7] |
| 9 | 783 | 20.5500 | who has been appointed as the election commissioner of india | [14, 9, 4, 2] |
| 10 | 6923 | 20.4226 | who played the wicked witch in wicked on broadway | [3, 10, 15, 11] |

## Feature 4291

**激活频率**: 7.0%（552/7830 个样本激活）
**最大激活值**: 82.9933, **平均激活值(激活时)**: 10.8517

**标注**: 语义特征 -- 时间/日期/政治事件（国情咨文出现 3 次, 五天工作周, 日出时间, 酒类商店关门）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 1360 | 82.9933 | when is the state of the union addressed | [6, 27, 20, 0] |
| 2 | 2163 | 81.9229 | who started the state of the union address | [6, 27, 20, 0] |
| 3 | 1898 | 67.3740 | when did the 5 day work week begin | [6, 15, 1, 0] |
| 4 | 4786 | 62.3851 | when does the sun come up in the summer | [6, 15, 0, 0] |
| 5 | 4327 | 61.6039 | when do liquor stores close on sunday in minnesota | [28, 17, 9, 1] |
| 6 | 3538 | 60.4057 | where does the sun go during the night | [6, 15, 0, 0] |
| 7 | 6065 | 60.3456 | is there a book of james in the catholic bible | [25, 18, 0] |
| 8 | 1019 | 56.4100 | where does the black friday term come from | [6, 15, 1, 2] |
| 9 | 581 | 56.0773 | who sits in front of president during state of the union | [6, 27, 20, 0] |
| 10 | 6484 | 54.2378 | where do you you go to my lovely | [15, 13, 27, 0] |

## Feature 4969

**激活频率**: 11.8%（923/7830 个样本激活）
**最大激活值**: 64.1412, **平均激活值(激活时)**: 8.5419

**标注**: 语义特征 -- 地理/历史（印度河流, 英吉利海峡, 尼罗河支流, 法印战争, 马里帝国）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6628 | 64.1412 | east flowing rivers from north to south in india | [14, 10, 11, 1] |
| 2 | 3808 | 49.6745 | where does the english channel begin and end | [4, 28, 0, 1] |
| 3 | 2952 | 48.4848 | what are the 3 tributaries of the nile river | [11, 10, 4, 22] |
| 4 | 2686 | 44.6425 | what is the average depth of the english channel | [4, 28, 0, 1] |
| 5 | 298 | 43.8417 | when was the last time mount ruapehu erupted | [14, 10, 7, 25] |
| 6 | 240 | 41.2418 | chandra and bhaga river meets at the place | [9, 17, 7, 2] |
| 7 | 2668 | 40.8845 | who laid the foundation for indian national congress | [14, 4, 5, 2] |
| 8 | 6426 | 39.8752 | which best describes timbuktu under the mali empire | [16, 9, 19, 1] |
| 9 | 804 | 39.5897 | who were the major leaders of the french and indian war | [14, 18, 16, 0] |
| 10 | 4305 | 38.6904 | how many circuses are there in the uk | [14, 17, 22] |

## Feature 5086

**激活频率**: 6.9%（543/7830 个样本激活）
**最大激活值**: 59.8757, **平均激活值(激活时)**: 9.1649

**标注**: 其他 -- 杂题（北爱尔兰, 二战荷兰, 黄丝带歌曲, 德国移民, 英国手机号）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 5085 | 59.8757 | map of the six counties of northern ireland | [26, 7, 10, 6] |
| 2 | 1192 | 46.8128 | burt bacharach raindrops keep falling on my head | [19, 5, 4, 0] |
| 3 | 3122 | 44.1840 | when did holland become involved in world war 2 | [16, 25, 29, 0] |
| 4 | 5986 | 40.0525 | song tie a yellow ribbon round the old oak tree | [19, 9, 25] |
| 5 | 7784 | 39.2222 | where did german immigrants settled in the 1800s | [16, 7, 15, 1] |
| 6 | 6225 | 35.6678 | what happened to germany's leader after ww1 | [7, 17, 21, 5] |
| 7 | 4753 | 35.5442 | who sang the theme song from russia with love | [13, 0, 24] |
| 8 | 3240 | 34.9620 | it's a long way to the top if you want to rock & roll ac dc | [19, 5, 19, 5] |
| 9 | 4661 | 34.5901 | how long is a uk mobile phone number | [26, 7, 7, 5] |
| 10 | 2024 | 34.3825 | who sings tie a yellow ribbon around the old oak tree | [19, 9, 25] |

## Feature 6352

**激活频率**: 6.9%（543/7830 个样本激活）
**最大激活值**: 31.5749, **平均激活值(激活时)**: 7.7611

**标注**: 语义特征 -- 电影/娱乐（Robin Hood, El Dorado, Beauty and the Beast, Dazed and Confused）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 6983 | 31.5749 | where was robin hood prince of thieves made | [10, 28, 10] |
| 2 | 6601 | 28.4982 | who played maudie in the movie el dorado | [10, 29, 18, 3] |
| 3 | 5185 | 28.0595 | what happened to the curse of oak island on history channel | [3, 28, 3, 2] |
| 4 | 7365 | 27.6150 | who played marius in the movie les miserables | [10, 17, 6, 1] |
| 5 | 3663 | 26.8070 | who plays heather in beauty and the beast | [21, 28, 11, 3] |
| 6 | 702 | 26.6574 | what is the movie dazed and confused about | [10, 29, 8, 0] |
| 7 | 887 | 26.5377 | when are organisms considered to belong to the same species | [28, 19, 4, 6] |
| 8 | 6069 | 26.3337 | the vta and the substantia nigra are found in the which part of the brain | [24, 29, 8, 2] |
| 9 | 1809 | 25.9841 | which film won the oscar for best animated feature in 2007 | [6, 29, 10, 1] |
| 10 | 6957 | 25.6821 | how many episodes of season 5 of curse of oak island | [3, 28, 3, 2] |

## Feature 7669

**激活频率**: 11.1%（871/7830 个样本激活）
**最大激活值**: 80.4694, **平均激活值(激活时)**: 10.0491

**标注**: 语义特征 -- 历史/军事/太空（Star Trek, NASA, V1/V2火箭, 南北战争将军, 足球名人堂）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 4509 | 80.4694 | what was the first star trek enterprise ship | [23, 23, 9, 1] |
| 2 | 4212 | 73.9739 | who is buried in the tomb of the unknown soldiers | [11, 25, 12, 1] |
| 3 | 6037 | 65.5277 | v1 and v2 rockets in world war 2 | [7, 2, 16, 1] |
| 4 | 5981 | 58.2483 | how many astronauts has nasa sent into space | [26, 4, 7, 7] |
| 5 | 3547 | 57.1732 | who was selected for the 2018 football hall of fame | [23, 11, 13, 1] |
| 6 | 7502 | 48.1643 | where does route 66 start on the west coast | [9, 19, 23, 1] |
| 7 | 4402 | 47.3852 | who was the successful commanding general of the northern forces in the civil... | [7, 0, 13, 2] |
| 8 | 3071 | 45.3145 | who is command sergeant major of the army | [23, 11, 14, 1] |
| 9 | 594 | 44.2962 | name and define the seven gifts of the holy spirit | [11, 22, 10, 3] |
| 10 | 4020 | 44.2169 | who has the biggest airport in the us | [26, 22, 26, 1] |

## Feature 7767

**激活频率**: 6.9%（544/7830 个样本激活）
**最大激活值**: 93.0799, **平均激活值(激活时)**: 10.4956

**标注**: 语义特征 -- 电影/电视/迪士尼（忍者神龟, Big Brother, Lion King, Full House）

| 排名 | 样本ID | 激活值 | Query | DocID (语义ID) |
|------|--------|--------|-------|---------------|
| 1 | 2643 | 93.0799 | what is the names of the teenage mutant ninja turtles | [3, 20, 2, 2] |
| 2 | 2731 | 77.0851 | do they show the jury house on big brother | [3, 8, 11, 0] |
| 3 | 5316 | 72.4264 | where was when we first met netflix filmed | [8, 25, 25, 1] |
| 4 | 2141 | 71.1918 | what is the name of the hyena in lion king | [3, 20, 15, 0] |
| 5 | 308 | 59.1394 | who attended all the three round table conferences | [4, 25, 26, 0] |
| 6 | 4737 | 59.0815 | what kind of bird is in the lion king | [3, 20, 15, 0] |
| 7 | 3325 | 56.9176 | who played the middle sister on full house | [3, 1, 25, 1] |
| 8 | 1165 | 51.5688 | when did the first ninja turtles come out | [3, 20, 4, 0] |
| 9 | 771 | 48.4501 | two atoms of the same element that are covalently bonded | [0, 25, 11, 0] |
| 10 | 7714 | 45.6800 | when does the eclipse end in the us | [11, 9, 9, 0] |

---

## 标注统计

| 类型 | 数量 | 占比 |
|------|------|------|
| 语义特征 | 15 | 60% |
| 路由特征 | 1 | 4% |
| 其他 | 9 | 36% |

## 分析

1. **语义特征占多数**。top 激活样本在主题/领域上高度一致：体育（738, 1621）、影视（3333, 6352, 7767）、音乐（597）、社交媒体（3399）、地理/历史（4969）、宗教（2596）、短语起源（2317）。

2. **路由特征极少（仅 1 个）**。Feature 590 是唯一一个 top 样本 DocID 高度一致的特征——4/10 条映射到 [5,3,11,0]（Game of Thrones）。说明 SAE 特征主要编码输入 query 的语义信息，而非输出 DocID 的路由路径。

3. **其他类** 的特征 top 样本主题杂乱，可能编码更抽象的特征（句式结构、问题类型、token 位置等）。

4. **结论**：Layer 12 的 SAE 特征主要捕捉输入端的语义信息（query 主题），而非输出端的路由信息（DocID 结构）。SAE 在这层学到的更多是理解问题而非生成答案。