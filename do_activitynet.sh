collection=activitynet
visual_feature=i3d
exp_id=run_0
root_path=D:\\PRVR_dataset
device_ids=0
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --device_ids $device_ids