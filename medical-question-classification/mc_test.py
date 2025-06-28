# The p-value from the McNemar's test indicates the probability of observing the differences in proportions (discordant pairs) as large as or larger than the one observed, assuming there is no real difference (null hypothesis).
# Test statistic:
# The chi-squared test statistic is calculated as: χ² = (b - c)² / (b + c).
# P-value:
# This statistic is then compared to a chi-squared distribution with 1 degree of freedom to determine the p-value.

# Online:
# https://www.omnicalculator.com/statistics/mcnemars-test

sinabert_pred = {}
parsbert_pred = {}

with open('sinabert_dgs_pred_2025.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        sinabert_pred[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

with open('parsbert_dgs_pred_2025.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        parsbert_pred[line.strip().split('\t')[0]] = line.strip().split('\t')[1]

a, b, c, d = 0, 0, 0, 0

for mid, sina in sinabert_pred.items():
    pars = parsbert_pred[mid]
    if sina == pars:
        if sina == 1:
            a += 1
        else:
            d += 1
    else:
        if sina == 1:
            b += 1
        else:
            c += 1

x = (b - c) * (b - c) / (b + c)

print(x)
