from sage.knots.link import Link
from PIL import Image
import ast

pd=[[1,27,2,26],[2,10,3,9],[24,3,25,4],[4,17,5,18],[5,23,6,22],[19,7,20,6],[7,19,8,18],[8,23,9,24],[27,11,28,10],[13,1,14,34],[31,14,32,15],[15,32,16,33],[25,16,26,17],[33,30,34,31],[21,20,22,21],[28,11,29,12],[29,13,30,12]]

# pd = ast.literal_eval(pd)
L = Link(pd)

p = L.plot(gap=0.25, thickness=1.5, color='black')
filename = f'ncs19pokepoke.png'
p.save(f"{filename}", dpi=300)

# Resize
# img = Image.open(f"{filename}")
# img_resized = img.resize((700, 700), Image.LANCZOS)
# img_resized.save(f"{filename}")

# import ast
# import json

# unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
# unknots_out  = 'SeeingTheUnknot_unknots.json'

# id = {nc: 1 for nc in range(20, 41)}

# with open(unknots_src, 'r', encoding='utf-8') as f, \
#     open(unknots_out, 'r+', encoding='utf-8') as out:
#     count = 0
#     for line in f:
#         pd = line.strip()
#         pd = ast.literal_eval(pd)
#         pd = ast.literal_eval(pd)
#         nc = len(pd)

#         # if 20 <= nc and nc <= 40:
#         if nc == 32:
#             data = json.load(out)
#             data["unknots"][str(nc)][f"{nc}_{id[nc]}"] = pd
#             id[nc] += 1
#             out.seek(0)
#             json.dump(data, out, indent=4)
#             out.truncate()
#             count += 1
        
#         if count == 2:
#             break










# import ast, json, os

# unknots_src  = 'GoogleDeepmind_hard_unknots.csv'
# unknots_out  = 'SeeingTheUnknot_unknots.json'

# if os.path.exists(unknots_out) and os.path.getsize(unknots_out) > 0:
#     with open(unknots_out, 'r', encoding='utf-8') as out:
#         data = json.load(out)
# else:
#     data = {"unknots": {str(n): {} for n in range(20, 41)}}

# id_counter = {n: len(data["unknots"][str(n)]) + 1 for n in range(20, 41)}

# count = 0
# with open(unknots_src, 'r', encoding='utf-8') as f:
#     for line in f:
#         s = line.strip()
#         if not s:
#             continue
#         try:
#             pd = ast.literal_eval(ast.literal_eval(s))
#             nc = len(pd)
#         except Exception:
#             continue

#         if 20 <= nc and nc <= 40:
#             k = f"{nc}_{id_counter[nc]}"
#             data["unknots"][str(nc)][k] = str(pd).replace(' ', '')
#             id_counter[nc] += 1
#             count += 1
#             print(count)

# with open(unknots_out, 'w', encoding='utf-8') as out:
#     json.dump(data, out, indent=4)
