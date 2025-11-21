import json

with open("/home/arthur/stvg_petl/TA-STVG/data/hc-stvg2/annos/test.json", 'r') as file:
    test_anno_ta = json.load(file)

with open("/home/arthur/stvg_petl/TA-STVG/data/hc-stvg2/annos/hcstvg_v2/val.json", 'r') as file:
    test_anno_cg = json.load(file)

for i in test_anno_ta:
    if test_anno_ta[i]['caption'] == "The boy in striped clothes raises his finger forward, then steps back against the door.":
        print(f"Found in TA-STVG test set: {i}")
        target_video = i
        break
    
print(test_anno_ta[target_video])
print(test_anno_cg[target_video])

print(test_anno_ta['88_xH1WLtZ8csM.mp4'])
print(test_anno_cg['88_xH1WLtZ8csM.mp4'])