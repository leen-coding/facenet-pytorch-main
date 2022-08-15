import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet
from utils.dataloader import LFWDataset
from utils.utils_metrics import test

if __name__ == "__main__":
    #--------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #--------------------------------------#
    cuda            = True
    #--------------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet
    #   inception_resnetv1
    #--------------------------------------#
    backbone        = "mobilenet"
    #--------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    #--------------------------------------------------------#
    input_shape     = [160, 160, 3]
    #--------------------------------------#
    #   训练好的权值文件
    #--------------------------------------#
    model_path = "D:\\Files\\arcface-pytorch\\result\\mobilenet+triplet_web_bs192\\ep021-loss0.894-val_loss2.080.pth"
    # model_path      = "model_data/facenet_mobilenet.pth"
    #--------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    #--------------------------------------#
    # blocktest = ["g-g", "m-m", "n-g", "n-m"]
    # for testset in blocktest:
    #
    #     if testset == "g-g":
    #         lfw_dir_path = "D:\\Files\\arcface-pytorch\\ROF\\sunglasses"
    #         lfw_pairs_path = "D:\\Files\\arcface-pytorch\\ROF\\glasses_pairs.txt"
    #     elif testset == "m-m":
    #         lfw_dir_path = "D:\\Files\\arcface-pytorch\\ROF\\masked"
    #         lfw_pairs_path = "D:\\Files\\arcface-pytorch\\ROF\\mask_pairs.txt"
    #     elif testset == "n-g":
    #         lfw_dir_path = "D:\\Files\\arcface-pytorch\\ROF\\combined"
    #         lfw_pairs_path = "D:\\Files\\arcface-pytorch\\ROF\\n-g.txt"
    #     elif testset == "n-m":
    #         lfw_dir_path = "D:\\Files\\arcface-pytorch\\ROF\\combined"
    #         lfw_pairs_path = "D:\\Files\\arcface-pytorch\\ROF\\n-m.txt"
    #     else:
    #         raise ValueError

        # blocktest = ["lfw.3","lfw.5","lfw.7","lfw.9"]
        # for testset in blocktest:
        #
        #     if testset == "lfw.3":
        #         lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-3"
        #         lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb3.txt"
        #     elif testset == "lfw.5":
        #         lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-5"
        #         lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb5.txt"
        #     elif testset == "lfw.7":
        #         lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-7"
        #         lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb7.txt"
        #     elif testset == "lfw.9":
        #         lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-9"
        #         lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb9.txt"
        #     else:
        #         raise ValueError

    blocktest = ["lfw.2","lfw.4","lfw.6","lfw.8"]
    for testset in blocktest:

        if testset == "lfw.2":
            lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-2"
            lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb2.txt"
        elif testset == "lfw.4":
            lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-4"
            lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb4.txt"
        elif testset == "lfw.6":
            lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-6"
            lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb6.txt"
        elif testset == "lfw.8":
            lfw_dir_path    = "D:\\Files\\dataset\\lfws\\lfw-factor-8"
            lfw_pairs_path  = "D:\\Files\\dataset\\lfws\\lfwbb8.txt"
        else:
            raise ValueError

    # testsets = ["rof","mfr2","mlfw","lfw"]
    # for testset in testsets:
    # # testset = "mlfw"
    #     if testset == "rof":
    #         lfw_dir_path    = "D:\\Files\\arcface-pytorch\\ROF\\combined"
    #         lfw_pairs_path  = "D:\\Files\\arcface-pytorch\\ROF\\ROFCombine_m_un_pairs_clean.txt"
    #     elif testset == "mfr2":
    #         lfw_dir_path    = "D:\\Files\\arcface-pytorch\\mfr2"
    #         lfw_pairs_path  = "D:\\Files\\arcface-pytorch\\mfr2_pairs.txt"
    #     elif testset == "mlfw":
    #         lfw_dir_path    = "D:\\Files\\arcface-pytorch\\mlfw_dataset/mlfw_aligned_dir"
    #         lfw_pairs_path  = "D:\\Files\\arcface-pytorch\\mlfw_dataset/mlfw_pairs.txt"
    #     elif testset == "lfw":
    #         lfw_dir_path    = "D:\\Files\\arcface-pytorch\\lfw"
    #         lfw_pairs_path  = "D:\\Files\\arcface-pytorch\\model_data/lfw_pairs_test.txt"
    #     else:
    #         raise ValueError
        # --------------------------------------#
        #   评估的批次大小和记录间隔
        # --------------------------------------#
        batch_size = 64
        log_interval = 1
        # --------------------------------------#
        #   ROC图的保存路径
        # --------------------------------------#
        png_save_path = "model_data/baseline_roc_test{}.png".format(testset)

        test_loader = torch.utils.data.DataLoader(
            LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size,
            shuffle=False, drop_last=False)

        model = Facenet(backbone=backbone, mode="predict")

        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model  = model.eval()
        print("testting {}".format(testset))
        if cuda:
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model = model.cuda()

        test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
