import os
import pandas as pd 
import csv

img_list = os.listdir('fashion_data/testK')
from_list_men = ['fashionMENTees_Tanksid0000320301_7additional.jpg','fashionMENTees_Tanksid0000411721_7additional.jpg',
		'fashionMENTees_Tanksid0000595506_4full.jpg']
from_list_women = ['fashionWOMENBlouses_Shirtsid0000644505_3back.jpg', 'fashionWOMENBlouses_Shirtsid0000787301_4full.jpg',
		'fashionWOMENDressesid0000070002_7additional.jpg','fashionWOMENDressesid0000193103_1front.jpg',
		'fashionWOMENDressesid0000229801_1front.jpg', 'fashionWOMENDressesid0000262902_1front.jpg', 'fashionWOMENTees_Tanksid0000561602_2side.jpg']
to_list_men = []
to_list_women = []

output_path	= 'fashion_data/sgp_arb.csv'
if os.path.exists(output_path):
	os.remove(output_path	)

result_file = open(output_path, 'w')
#processed_names = set()
#print >> result_file, 'from: to'
csv_writer = csv.writer(result_file	)

import numpy as np 
csv_writer.writerow(['from','to'])
for i in img_list[:1000]:
	if i.startswith('fashionMEN'):
		tmp = np.load(os.path.join('fashion_data/testK', i))
		if tmp.sum()>6:
			to_list_men.append(i[:-4])
for i in img_list[:1000]:
	if i.startswith('fashionWO'):
		tmp = np.load(os.path.join('fashion_data/testK', i))
		if tmp.sum()>5:
			to_list_women.append(i[:-4])

to_list_women.sort()
to_list_men.sort()

for i in to_list_men:
	for j in from_list_men:
		#print >> result_file, "%s: %s" % (j, i)
		csv_writer.writerow([j,i])
		result_file.flush()
for i in to_list_women:
	for j in from_list_women:
		#print >> result_file, "%s: %s" % (j, i)
		csv_writer.writerow([j,i])
		result_file.flush()
