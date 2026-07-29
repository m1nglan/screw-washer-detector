"""检查量化参数"""
import json
d = json.load(open('output/model.json'))

# 分类器 Conv 层
keys = [k for k in d['configs'].keys() if 'classifier' in k]
print("分类器相关层:")
for key in keys:
    print(f"\n--- {key} ---")
    cfg = d['configs'][key]
    for k, v in cfg.items():
        val = d['values'].get(str(v.get('hash', '')))
        if val:
            print(f"  {k}: scale={val['scale']}, zp={val['zero_point']}")
        else:
            print(f"  {k}: hash={v.get('hash')}")

# 输入量化参数
input_hash = d['configs'][keys[0]]['input']['hash']
inp_val = d['values'].get(str(input_hash))
print(f"\n输入量化: scale={inp_val['scale']}, zp={inp_val['zero_point']}")

