import time

from torch.utils.tensorboard import SummaryWriter

from src_ele.dir_operation import check_and_make_dir, traverseFolder, traverseFolder_folder


def start_logging(path_log='logs_train'):
    # 添加tensorboard记录器
    check_and_make_dir(path_log)
    logging_list = traverseFolder_folder(path_log)
    dir_path = ''
    if logging_list == []:
        dir_path = path_log+'/1_logging_{}'.format(time.strftime("%m-%d__%H-%M-%S", time.localtime()))
        # check_and_make_dir(dir_path)
    else:
        numer_logging = int(logging_list[-1].split('\\')[1].split('_')[0]) + 1
        # print(numer_logging)
        dir_path = path_log+'/{}_logging_{}'.format(numer_logging, time.strftime("%m-%d__%H-%M-%S", time.localtime()))
        # check_and_make_dir(dir_path)
    writer = SummaryWriter(dir_path)
    return writer

# def logging_data(writer):
#     writer.add_scalar('rnet_loss', loss.item(), total_train_step)
#     writer.add_scalar('max loss', max_loss, epoch_times + last_epoch)
#     writer.add_scalar('mean loss', total_loss / train_batch_num, epoch_times + last_epoch)
#     writer.add_images('output pic:{}'.format(total_train_step), pngs_org*256)
#
# def end_logging(writer):
#     writer.close()
#     pass
# start_logging()