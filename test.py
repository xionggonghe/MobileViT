from ViTmodule.mobilevit_block import *
import argparse

def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    # total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} training parameters.')
    return total_params


from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    model = MobileViTBlockv2(opts=None, in_channels=3, attn_unit_dim=32)
    print(model)
    cnn_paras_count(model)
    images = torch.rand([8, 3, 64, 64])
    # writer = SummaryWriter('runs/experiment_1')
    # writer.add_graph(model, images)
    # writer.close()

    y = model(images)
    print(y)





