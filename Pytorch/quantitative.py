import sys
import os
from main_quant import main

PN_score = 0
PP_score = 0
test_length = 25
for id in range(test_length):
    print(id)
    PN_score += main(image_id=id, mode="PN", quant_res=True)
    PP_score += main(image_id=id, mode="PP", quant_res=True)

print('PN_score:' + str(PN_score/test_length))
print('PP_score:' + str(PP_score/test_length))
