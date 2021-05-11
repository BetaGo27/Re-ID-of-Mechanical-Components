import torch

loaddir = '/home/betago/Documents/Thesis/Model/Dataset/cfg2/siam_reid_14.pth'
loaddir2 = '/home/betago/Documents/Thesis/Model/Dataset/25-50/siam_reid_16.pth'
checkpoint = torch.load(loaddir)
checkpoint2 = torch.load(loaddir2)
model1 = checkpoint['model']
model2 = checkpoint2['model']
# if model1.head.fc_embedding.weight == model2.head.fc_embedding.weight:
#     print("True")
# else:
#     print("False")
# print(model1)
# print(model2)
