import sys
import os
import time
from main_quant import main

t0 = time.time()
RES = False
PN_score = 0
PP_score = 0
test_length = 25
for id in range(test_length):
    print(id)
    PN_score += main(image_id=id, mode="PN", quant_res=RES)
    print('PN_score:' + str(PN_score/(id+1)))

    PP_score += main(image_id=id, mode="PP", quant_res=RES)
    print('PP_score:' + str(PP_score/(id+1)))

    print("Runtime:", time.time() - t0)