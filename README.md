# PatchGAN
Image Decomposition in GAN network(Reference:Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images, CVPR2020)

https://openaccess.thecvf.com/content_CVPR_2020/papers/Zou_Deep_Adversarial_Decomposition_A_Unified_Framework_for_Separating_Superimposed_Images_CVPR_2020_paper.pdf

Requirements:(All network reimplements are same of similar)

* 1.Pytorch 1.3.0
* 2.Torchvision 0.2.0
* 3.Python 3.6.10
* 4.glob
(Dataset)
* 5.PIL
* 6.tqdm(For training)
* 7.Opencv-Python
* 8.tensorboardX

Dataset Modified:

Line 25,26,27

imgpath='/public/zebanghe2/joint/train/mix'

transpath='/public/zebanghe2/joint/train/transmission'

maskpath='/public/zebanghe2/joint/train/sodmask'

Train

python train.py

Epoch Number:Line 72

Batch Size:Line 67

Attention!!!

In my practice, network has a problem of loss Sudden Changing. When the images of two layers are near to black/white, training process will crash and output will change to strange things like texture.

Test
python test.py

If any problem, please ask in issue.

Jerry He

