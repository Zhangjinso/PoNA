# PoNA

Code for Our TIP [paper.](https://ieeexplore.ieee.org/document/9222550)





## Data preparation



```bash
#environment setup
conda create -n tip python=3.6
conda activate tip
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install scikit-image pillow pandas tqdm dominate 

```

The process of data preparation is following  [Pose Transfer](https://github.com/tengteng95/Pose-Transfer)



## Testing

Here is the results（**[baidu](https://pan.baidu.com/s/1o8nK0wrrFcvEsUbzHtOHJQ)**  fetch code:abcd) if you are interested in this work. 

Checkpoint of DeepFashion is [here](https://pan.baidu.com/s/1YjkcA-P99cWMMfrB_Vw5Yw ) (fetch code: abcd). You can test by yourself. 

<u>Note that the Checkpoint of Market1501 is lost due to COVID-19 and the architecture is a little difference compared with DeepFashion. We will release it as soon as possible.</u>

```python
#fashion
python test.py --dataroot ./fashion_data/ --name fashion_tip --model PoNA --phase test --dataset_mode keypoint --norm instance --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 18 --no_flip --which_model_netG PoNA --checkpoints_dir ./checkpoints/ --pairLst ./fashion_data/fasion-resize-pairs-test.csv --which_epoch test1 --results_dir ./tip_fashion --display_id 0
```



## Evaluation

```shell
#environment setup
conda create -n test_tip python=3.6
conda activate test_tip
pip install -r requirements_tf.txt
# test script
python tool/getMatric_fashion.py
```

## Citation

If you use this code, please cite our paper.

```b
@ARTICLE{9222550,
  author={K. {Li} and J. {Zhang} and Y. {Liu} and Y. -K. {Lai} and Q. {Dai}},
  journal={IEEE Transactions on Image Processing}, 
  title={PoNA: Pose-Guided Non-Local Attention for Human Pose Transfer}, 
  year={2020},
  volume={29},
  number={},
  pages={9584-9599},
  doi={10.1109/TIP.2020.3029455}}
```

## Acknowledgments

Our code is based on [Pose Transfer](https://github.com/tengteng95/Pose-Transfer).
