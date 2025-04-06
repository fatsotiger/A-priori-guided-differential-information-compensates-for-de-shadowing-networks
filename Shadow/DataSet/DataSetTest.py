
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os

class DataSet_t(Dataset):
    """

    Dataset for the training of the model.

    """

    def __init__(self, haze_dir):
        """

        :param haze_dir: Address with foggy image
        :param dehaze_dir: Address of fog free image
        """
        # image Set

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

        # haze
        """
        :param haze_list_test: a name list of image , for instence: 1400_1.jpg
        :param root_hazy: Address of image file , for instence: 'E/data/haze/'
        """
        self.haze_list_test = []
        for i in os.listdir(haze_dir):
            self.haze_list_test.append(i)
        self.root_hazy = os.path.join(haze_dir)
        self.file_len = len(self.haze_list_test)



    def __getitem__(self, index):
        """

            If you want to own operate on the image, fill in the module here

        """

        haze = self.transform(
            Image.open(
                self.root_hazy + self.haze_list_test[index]).convert('RGB')
        )

        return haze,self.haze_list_test[index]
    def __len__(self):
        return self.file_len

