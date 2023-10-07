collection=tvr
visual_feature=i3d_resnet
q_feat_size=768
margin=0.1
exp_id=run_1
root_path=/home/featurize/data
device_ids=0
# training
python method/train.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --exp_id $exp_id \
                    --q_feat_size $q_feat_size --margin $margin --device_ids $device_ids