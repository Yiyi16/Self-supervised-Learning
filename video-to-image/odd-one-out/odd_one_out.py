import argparse
import numpy
import os
import shutil
import time
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from odd_one_out_func import *

# Load all model arch available on Pytorch
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/zhangyy/imgdataset/vocdata/', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--outf', default='./voc-test/b_1_u_v_a/',
                    help='folder to output model checkpoints')
parser.add_argument('--evalf', default="/eval" ,help='path to evaluate sample')
parser.add_argument('--arch', '-a', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')#default 0.9
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--train', action='store_true',
                    help='train the model')
parser.add_argument('--test', action='store_true',
                    help='test a [pre]trained model on new images')
parser.add_argument('-t', '--fine-tuning', action='store_true',
                    help='transfer learning + fine tuning - train only the last FC layer.')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--lam', default=0, type=float)
parser.add_argument('--smoothlam', default=0,type=float)
parser.add_argument('--eqlam', default=0, type=float)
parser.add_argument('--proxyloss', default=0, type=float)

best_prec1 = torch.FloatTensor([0])
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic=True
#torch.backends.cudnn.benchmark = True

def get_images_name(folder):
        """Create a generator to list images name at evaluation time"""
        onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        for f in onlyfiles:
            yield f

def pil_loader(path):
    """Load images from /eval/ subfolder and resized it as squared"""
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            sqrWidth = numpy.ceil(numpy.sqrt(img.size[0]*img.size[1])).astype(int)
            return img.resize((sqrWidth, sqrWidth))

def main():
    ### proxy task ###
    data_path = '/home/zhangyy/UCF101/pyflow/flow_result/'
    save_model_path = './proxy_result/uf_voc_alexnet'
    CNN_embed_dim = 256
    dropout_p = 0.3
    alex_size = 224
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 512
    RNN_fc_dim = 128

    k = 16
    batch_size = 8
    learning_rate = 1e-4
    log_interval = 10
    begin_frame, end_frame, skip_frame = 4,18,3
  
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    params = {'batch_size':batch_size, 'shuffle':True, 'num_workers':0, 'pin_memory':True} if use_cuda else {}

    action_names = np.load('/home/zhangyy/UCF101/pyflow/label.npy')
    le = LabelEncoder()
    le.fit(action_names)
    list(le.classes_)

    action_category = le.transform(action_names).reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(action_category)

    actions = []
    fnames = os.listdir(data_path)[:1000]

    all_names = []
    for f in fnames:
        loc1 = f.find('v_')
        loc2 = f.find('_g')
        actions.append(f[(loc1 + 2): loc2])
        name = os.path.join(data_path, f)
        all_names.append(name+'__1')
        all_names.append(name+'__2')
        all_names.append(name+'__3')

    all_X_list = all_names
    all_y_list = labels2cat(le, action_names)

    proxy_transform = transforms.Compose([transforms.Resize([224,224]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    selected_frames1 = np.arange(begin_frame, end_frame, skip_frame).tolist()
    selected_frames2 = np.arange(begin_frame, end_frame, skip_frame).tolist()
    selected_frames3 = np.arange(begin_frame, end_frame, skip_frame).tolist()

    proxy_train_set = Dataset_CRNN(all_X_list, all_y_list, selected_frames1, selected_frames2, selected_frames3, transform=proxy_transform)
    proxy_train_loader = data.DataLoader(proxy_train_set, **params)
    
    proxymodel = proxyEncoder()
    global args, best_loss, cuda, labels, best_prec1
    args = parser.parse_args()
    best_loss = 99
   

    try:
        os.makedirs(args.outf)
    # os.makedirs(opt.outf+"/model")
    except OSError:
        pass
    # can we use CUDA?
    cuda = True #torch.cuda.is_available()
    print ("=> using cuda: {cuda}".format(cuda=cuda))
    args.distributed = args.world_size > 1
    print ("=> distributed training: {dist}".format(dist=args.distributed))

    ############ DATA PREPROCESSING ############
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'ttrain')
    testdir = os.path.join(args.data, 'val')
    # Normalize on RGB Value
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Size on model
    if args.arch.startswith('inception'):
        size = (299, 299)
    else:
        size = (224, 256)

    # Train -> Preprocessing -> Tensor
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(size[0]), #224 , 299
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    #print (train_dataset.classes)
    # Get number of labels
    labels = len(train_dataset.classes)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # Pin memory
    if cuda:
        pin_memory = True
    else:
        pin_memory = False

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=pin_memory, sampler=None)

        
    if args.test:
        # Testing -> Preprocessing -> Tensor
        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(testdir, transforms.Compose([
                transforms.Resize(size[1]), # 256
                transforms.CenterCrop(size[0]), # 224 , 299
                transforms.ToTensor(),
                normalize,
            ]), loader=pil_loader),
            batch_size=16, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)

    
    ############ BUILD MODEL ############
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Create model from scratch or use a pretrained one
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
        # print(model)
        # quit()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](num_classes=labels)
        # print(model)

    # Freeze model, train only the last FC layer for the transfered task
    if args.fine_tuning:
        print("=> transfer-learning mode + fine-tuning (train only the last FC layer)")
        # Freeze Previous Layers(now we are using them as features extractor)
        for param in model.parameters():
            param.requires_grad = False
        # Fine Tuning the last Layer For the new task
        # RESNET
        if args.arch == 'resnet18':
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, labels)
            parameters = model.fc.parameters()
            # print(model)
            # quit()
        # ALEXNET & VGG
        elif args.arch == 'alexnet' or args.arch == 'vgg16':
            model.classifier._modules['6'] = nn.Linear(4096, labels)
            parameters = model.classifier._modules['6'].parameters()
            # print(model)
            # quit()
        elif args.arch == 'densenet121': # DENSENET
            model.classifier = nn.Linear(1024, labels)
            parameters = model.classifier.parameters()
            # print(model)
            # quit()
        # INCEPTION
        elif args.arch == 'inception_v3':
            # Auxiliary Fc layer
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs, labels)
            # parameters = model.AuxLogits.fc.parameters()
            # print (parameters)
            # Last layer
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, labels)
            parameters = model.fc.parameters()
            # print(model)
            # quit()
        else:
            print("Error: Fine-tuning is not supported on this architecture.")
            exit(-1)
    else:
        #parameters = list(model.parameters())+list(cnn_encoder.parameters())+list(rnn_decoder.parameters())
        parameters = list(model.parameters())+list(proxymodel.parameters())

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if cuda:
       criterion.cuda()

    # Set SGD + Momentum
    #optimizer = torch.optim.SGD(parameters, args.lr,momentum = args.momentum,
    #                            weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.1,0.999), weight_decay=0)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if cuda:
                checkpoint = torch.load(args.resume)
            else:
                # Load GPU model on CPU
                checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
            args.start_epoch = checkpoint['epoch']
            #best_prec1 = checkpoint['best_prec1']
            #best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Load model on GPU or CPU
    if cuda:
        proxymodel.cuda()
        model.cuda()
    else:
        model.cpu()
    ############ TRAIN/EVAL/TEST ############
    #cudnn.benchmark = True

    # Evaluate?
    if args.evaluate:
        print("=> evaluating...")
        validate(val_loader, model, criterion)
        return

    # Training
    if args.train:
        print("=> training...")
        proxyiter = iter(proxy_train_loader)
        trainiter = iter(train_loader) 
        names = get_images_name(os.path.join(testdir, 'images'))
        end=time.time()
       
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            #adjust_learning_rate(optimizer, epoch)
            try:
                px, py = next(proxyiter)
                px, py = px.cuda(), py.cuda()
                #px1, px2, px3, py = next(proxyiter)
                #px1, py2, px3, py = px1.cuda(),px2.cuda(),px3.cuda(),py.cuda()
            except StopIteration:
                proxyiter = iter(proxy_train_loader)
                px, py = next(proxyiter)
                px, py = px.cuda(), py.cuda()
                #px1, px2, px3, py = next(proxyiter)
                #px1, px2, px3, py = px1.cuda(),px2.cuda(),px3.cuda(),py.cuda()
            try:
                X, y = next(trainiter)
                X, y = X.cuda(), y.cuda()
            except StopIteration:
                trainiter = iter(train_loader)
                X,y = next(trainiter)
                X, y = X.cuda(), y.cuda()
            # Train for one epoch
            
            loss1 = train(px, py, X, y, proxymodel, model, criterion, optimizer, epoch, end)
            prec1 = torch.Tensor([0])
            #if loss1 < 0.9:
                #prec1 = validate(val_loader, model, criterion)
                #prec1 = test(test_loader, model, names, train_dataset.classes)
            # Remember best prec@1 and save checkpoint
            if cuda:
                loss1 = loss1.cpu() # Load on CPU if CUDA
                prec1 = prec1.cpu()
            # Get bool not ByteTensor
            is_best = bool(loss1 < best_loss)
            #print(best_prec1.numpy())
            #is_best = bool(prec1.numpy() > best_prec1.numpy())
            # Get greater Tensor
            best_loss = torch.tensor(min(loss1, best_loss))
            #best_prec1 = torch.FloatTensor(max(prec1.numpy(), best_prec1.numpy()))
            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    #'best_prec1': best_prec1,
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),},is_best)
    '''        
            names = get_images_name(os.path.join(testdir,'images'))
            checkpoint = torch.load(os.path.join(args.outf, 'model_best.pth.tar'))
            model.load_state_dict(checkpoint['state_dict'])
            test(test_loader, model, names, train_dataset.classes)
    '''
    # Testing
    if args.test:
        print('=> testing...')
        names = get_images_name(os.path.join(testdir, 'images'))
        checkpoint = torch.load(os.path.join(args.outf,'model_best.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        iteration = checkpoint['epoch']
        print(iteration)
        test(test_loader, model, names, train_dataset.classes)
        return

def train(px, py, input, target, proxymodel, model, criterion, optimizer, epoch, end):
    """Train the model on Training Set"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    #cnn_encoder, rnn_decoder = proxy_model
    #cnn_encoder.train()
    #rnn_decoder.train()
    proxymodel.train()
    model.train()

   
    correct, total = 0,0
    #print(px.shape)
    px_var = torch.autograd.Variable(px) 
    
    #px2_var = torch.autograd.Variable(px2)
    #px3_var = torch.autograd.Variable(px3)
    py_var = torch.autograd.Variable(py)
    #print(px.shape)
    pout = proxymodel(px_var)
    #print(pout)
    outt = F.sigmoid(pout)
    
    #print(py_var)
    #pout = pout.view(pout.shape[0],-1)
    #print(pout.shape)
    print(py_var)
    print(outt)
    #ploss = criterion(pout, py_var)
    loss_P1= torch.nn.CrossEntropyLoss(reduce=True,size_average=False)
   
  
    #print("py!!!!!!!!!!!!!!:",py_var)
    #print(outt)
    ploss = loss_P1(outt, py_var)
    #print("ploss!!!!!!!!!!!!:",ploss)

    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)
    #print(input.shape)
    output = model(input_var)
    
    loss = criterion(output, target_var)
    #print("loss!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:",loss)
    loss.add_(args.proxyloss * ploss)
    
    prec1 = accuracy(output.data, target, topk=(1,))
    top1.update(prec1[0], input.size(0))
    losses.update(loss.data[0], input.size(0))
    
   
    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

        # measure elapsed time
    batch_time.update(time.time() -end)
   
        
        # Info log every args.print_freq
    if epoch % args.print_freq == 0:
        print('Iteration: [{0}]\t'
              'Time {batch_time.val:.3f}\t'
              'Loss {loss.val:.4f}\t'
              'Proxy {proxy}\t'
              'Train Acc {top1_avg}'.format(
               epoch, batch_time=batch_time,
               loss=losses, proxy=ploss,
               top1_avg=numpy.asscalar(top1.avg.cpu().numpy())))

    return losses.avg


def test(test_loader, model, names, classes):
    """Test the model on the Evaluation Folder

    Args:
        - classes: is a list with the class name
        - names: is a generator to retrieve the filename that is classified
    """
    # switch to evaluate mode
    #model = torch.load(os.path.join(args.outf, 'model_best.pth.tar'))
    model.eval()
    top1 = AverageMeter()
    # Evaluate all the validation set
    correct, total  = 0,0
    for i, (input, target) in enumerate(test_loader):
        if cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        prec1 = accuracy(output.data, target, topk=(1,))
        top1.update(prec1[0], input.size(0))
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target_var).sum().item() 
        '''
        # Take last layer output
        if isinstance(output, tuple):
            output = output[len(output)-1]

        # print (output.data.max(1, keepdim=True)[1])
        lab = classes[numpy.asscalar(output.data.max(1, keepdim=True)[1].cpu().numpy())]
        print ("Images: " + next(names) + ", Classified as: " + lab)
        '''
    print(' * test acc {acc}'.format(acc=numpy.asscalar(top1.avg.cpu().numpy())))
   # print(' ** acc {acc}'.format(acc=100.*correct/total))
    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(args.outf, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.outf, filename), os.path.join(args.outf,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # pretrained -t 150
    lr = args.lr * (0.1 ** (epoch // 1800))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
