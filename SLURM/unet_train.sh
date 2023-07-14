cd $HOME/cv_p_13/
source ./env/bin/activate
cd unet/Pytorch-UNet
wandb login b565921774b4adac12eb50d137baff3aa18e514f
python3 train.py --load ./checkpoints/checkpoint_epoch5.pth  --amp --epochs 30 --batch-size 10
deactivate
