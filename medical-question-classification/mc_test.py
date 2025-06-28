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

mc_nemar = (b-c)*(b-c)/(b+c)

print(mc_nemar)