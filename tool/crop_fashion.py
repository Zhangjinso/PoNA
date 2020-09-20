from PIL import Image
import os

img_dir = '/home/zjs/my/Pose-Transfer/results/sgp_arb/fashion_PATN_1.0/test_latest/images'
save_dir = '/home/zjs/my/Pose-Transfer/results/sgp_arb/fashion_PATN_1.0/images_target'

if not os.path.exists(save_dir):
	os.mkdir(save_dir)

cnt = 0

for item in os.listdir(img_dir):
	if not item.endswith('.jpg') and not item.endswith('.png'):
		continue
	cnt = cnt + 1
	print('%d/%d ...' %(cnt, len(os.listdir(img_dir))))
	img = Image.open(os.path.join(img_dir, item))
	imgcrop = img.crop((176*3, 0, 176*4, 256))
	imgcrop.save(os.path.join(save_dir, item))
