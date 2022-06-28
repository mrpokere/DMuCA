# 定义普通训练过程
import os
import joblib
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
from utils import count_sliding_window, sliding_window, grouper, camel_to_snake,metrics,display_goundtruth


def train(logger, net, optimizer, criterion, train_loader, epoch, save_epoch, scheduler=None,
          device=torch.device('cpu'), val_loader=None, supervision='full', vis_display=None,
           RUN=None,test_img=None,test_gt=None,hyperparams=None,gt=None):
    # 首先检查损失函数
    
    # for p in range(RUN):
    if criterion is None:
        logger.debug("Missing criterion. You must specify a loss function.")
        raise Exception("Missing criterion. You must specify a loss function.")

    # 定义全局变量
    net.to(device)
    save_epoch = save_epoch if epoch > 20 else 1
    lr_list = []
    losses = np.zeros(1000000)
    avg_losses = np.zeros(10000)
    iter_ = 0
    batch_loss_win, epoch_loss_win, val_win, lr_win = None, None, None, None
    val_accuracies = []
    LEN = len(train_loader)
    test_best = 0.
    avg_loss=0.
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # 因为每轮训练结尾都要进行验证模式，需要重新将模式调整回训练模式
        net.train()
        avg_loss = 0.
        for batch_idx, (data, label) in enumerate(train_loader):
            # 将数据载入GPU
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                loss = criterion(output, label)
            elif supervision == 'semi':
                output = net(data)
                output, rec = output
                loss = criterion[0](output, label) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            losses[iter_] = loss.item()
            # mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])  # 每100个batch计算一次平均损失
            lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            iter_ += 1
            # 释放缓存
            del (data, label, loss, output)
        avg_loss /= len(train_loader)
        avg_losses[e] = avg_loss
        
        
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            
            metric = -val_acc  
        else:
            metric = avg_loss
        # print(results)
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()
        # 在控制台打印信息
        tqdm.write(f"Epoch [{e}/{epoch}    avg_loss:{avg_loss:.5f}, val_acc:{val_acc:.2f}]")
        # 在日志打印信息
        logger.debug(f"Epoch [{e}/{epoch}    avg_loss:{avg_loss:.2f}, val_acc:{val_acc:.2f}]")
        avg_loss=0.
        # print(avg_loss)
        # 保存断点
        #        if e%save_epoch == 0:
        #            save_model(logger, net, camel_to_snake(str(net.__class__.__name__)),train_loader.dataset.dataset_name, epoch=e, metric=abs(metric))
        if e % save_epoch == 0:
            epoch_loss_win = vis_display.line(
                X=np.arange(e),
                Y=avg_losses[:e],
                win=epoch_loss_win,
                opts={'title': "Epoch loss" + str(RUN),
                    'xlabel': "Iterations",
                    'ylabel': "Loss"
                    })
            val_win = vis_display.line(Y=np.array(val_accuracies),
                                    X=np.arange(len(val_accuracies)),
                                    win=val_win,
                                    opts={'title': "Validation accuracy" + str(RUN),
                                            'xlabel': "Epochs",
                                            'ylabel': "Accuracy"
                                            })
    batch_loss_win = vis_display.line(
        X=np.arange(iter_),
        Y=losses[:iter_],
        win=batch_loss_win,
        opts={'title': "Batch loss" + str(RUN),
            'xlabel': "Iterations",
            'ylabel': "Loss"
            })
  
    return net
    


# 普通验证过程
def val(net, data_loader, device='cpu', supervision='full'):
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs,_ = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            # target = target - 1
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    # print(accuracy,total)
    return accuracy/ (total+1e-8)


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

#    probs = np.zeros(img.shape[:2] + (n_classes,))
    probs = np.zeros(img.shape[:2])
    img = np.pad(img, ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)), 'reflect')
    
    iterations = count_sliding_window(img, step=hyperparams['test_stride'], window_size=(patch_size, patch_size))
    
    for batch in tqdm(grouper(batch_size, sliding_window(img, step=1, window_size=(patch_size, patch_size))),
                      total=(iterations//batch_size),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            data = [b[0] for b in batch]

            data = np.copy(data)

            data = data.transpose(0, 3, 1, 2)

            data = torch.from_numpy(data)

            indices = [b[1:] for b in batch]

            data = data.to(device)
            data = data.type(torch.cuda.FloatTensor)
            # print(data.shape)
            output = net(data)
            # print(output.shape)
            _, output = torch.max(output, dim=1)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')
            if center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x, y] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


# 保存模型
def save_model(logger, model, model_name, dataset_name, metric):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('run')+' '+"{:.2f}".format(metric)
        tqdm.write("Saving neural network weights in {}".format(filename))
        logger.debug("-----Saving neural network weights in {}-----".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
    else:
        filename = str('run')
        tqdm.write("Saving model params in {}".format(filename))
        logger.debug("-----Saving model params in {}-----".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')
