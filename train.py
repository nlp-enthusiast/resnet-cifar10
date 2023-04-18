import torchvision

import os
import parameters
import function
import torch
import torch.nn as nn
from tqdm import tqdm
from model import Model
import torchvision.transforms as transforms

def main():
    # 选择device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 训练轮数 保存模型
    epochs = parameters.epoch
    save_model = parameters.save_model
    save_path = parameters.save_path

    # myTransforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    # 训练集
    train_dataset = torchvision.datasets.CIFAR10("./data", train=True, transform=torchvision.transforms.Compose(
        [torchvision.transforms.ColorJitter(0.5), torchvision.transforms.RandomHorizontalFlip(),
         torchvision.transforms.ToTensor()]))
    val_dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor())



    train_num = len(train_dataset)
    val_num = len(val_dataset)
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # 批次大小
    batch_size = parameters.batch_size

    # number of workers
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # 创建dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               )

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             )

    model = Model()
    model.to(device)

    #  使用交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.lr, weight_decay=parameters.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=parameters.lr, epochs=parameters.epoch,
                                                steps_per_epoch=len(train_loader))

    best_acc = 0.0
    # 为后面制作表图
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []

    # 开始训练
    for epoch in range(epochs):
        # train
        model.train()
        running_loss_train = 0.0
        train_accurate = 0.0
        train_bar = tqdm(train_loader)
        # 加载一个batch的数据进行训练
        for images, labels in train_bar:
            optimizer.zero_grad()

            outputs = model(images.to(device))
            # 计算损失
            loss = loss_fn(outputs, labels.to(device))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 更新学习率
            sched.step()

            predict = torch.max(outputs, dim=1)[1]
            # 计算准确率
            train_accurate += torch.eq(predict, labels.to(device)).sum().item()
            running_loss_train += loss.item()

        train_accurate = train_accurate / train_num
        running_loss_train = running_loss_train / train_num
        train_acc_list.append(train_accurate)
        train_loss_list.append(running_loss_train)

        print('[epoch %d] train_loss: %.7f  train_accuracy: %.3f' %
              (epoch + 1, running_loss_train, train_accurate))

        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_loader = tqdm(val_loader)
            for val_data in val_loader:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        val_acc_list.append(val_accurate)
        print('[epoch %d] val_accuracy: %.3f' %
              (epoch + 1, val_accurate))

        function.writer_into_excel_onlyval(save_path, train_loss_list, train_acc_list, val_acc_list, "CIFAR10")

        # 选择最best的模型进行保存 评价指标此处是acc
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_model)


if __name__ == '__main__':
    main()
