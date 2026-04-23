from argparse import ArgumentParser
from utils.data import IRSTDDataSet
import torch.utils.data as Data
import torch
from model.backbone import MultiAreaNet
from torch.optim import Adagrad
import time
import os
from tqdm import tqdm
from loss import AverageMeter,SLSIoULoss,Dice

from utils.criterion import *
from PIL import Image
from utils.lr_strategy import *
from model.Sobel_detection import *
import cv2
from model.backbone import MMER

def Args():
    parser=ArgumentParser()
    
    parser.add_argument('--data-dir',type=str,default='./datasets/IRSTD-1k/')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--warm-epoch',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=800)
    parser.add_argument('--lr',type=float,default=0.05)
    parser.add_argument('--start-epoch',type=int,default=0)
    
    parser.add_argument('--crop-size',type=int,default=256)
    parser.add_argument('--base-size',type=int,default=256)

    parser.add_argument('--weight-path',type=str,default='./weight/MultiAreaNet_weight.pkl') 
    parser.add_argument('--if-checkpoint',type=bool,default=False)
    parser.add_argument('--mode',type=str,default='train')
    args=parser.parse_args()
    return args
        
     

class Trainer(object):
    def __init__(self,args):
        self.args=args
        self.mode=args.mode
        self.lr=args.lr
        trainSet=IRSTDDataSet(self.args,'train')
        testSet=IRSTDDataSet(self.args,'test')

        self.trainLoader=Data.DataLoader(trainSet,self.args.batch_size,shuffle=True,drop_last=True)
        self.testLoader=Data.DataLoader(testSet,batch_size=1,drop_last=False)

        
        self.device=torch.device('cuda')
       
        
        self.model=MMER(1)
        
        self.model.to(self.device)
        self.optimizer=Adagrad(filter(lambda param:param.requires_grad,self.model.parameters()),lr=self.args.lr)
        self.loss_computation=SLSIoULoss()
        self.edge_loss_1=nn.BCEWithLogitsLoss()
        
        self.down = nn.MaxPool2d(2, 2)
        self.PD_FA=PD_FA(1,10,self.args.base_size)
        self.mIOU=mIoU(1)
        self.ROC=ROCMetric(1,10)
        self.best_IOU=0

        # self.loss_computation=torch.nn.CrossEntropyLoss()
        if args.mode=='train':
            if args.if_checkpoint:
                checkpoint=torch.load('./weight/MultiAreaNet-2025-05-12-18-30-14/checkpoint.pkl')
                self.model.load_state_dict(checkpoint['net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.start_epoch = checkpoint['epoch'] + 1
                self.best_IOU = checkpoint['iou']
            
            self.save_folder = 'weight/MultiAreaNet_max0.5-%s'%(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
        
        elif args.mode=='test':
            weight=torch.load(args.weight_path)
            # self.model.load_state_dict(weight['state_dict'])
            self.model.load_state_dict(torch.load('./weight/MultiAreaNet-2025-05-12-18-30-14/weight.pkl'))

    def train(self,epoch):
        self.model.train()
        process=tqdm(self.trainLoader)
        
        tag=False
        edge_tag=False
        losses=AverageMeter()
        for _,(img,label) in enumerate(process):
            img=img.to(self.device)
            label=label.to(self.device)
            
            if epoch>self.args.warm_epoch:
                tag=True
            if epoch>self.args.warm_epoch*10:
                edge_tag=True
            masks,pred,edge_out=self.model(img,tag,edge_tag) 
            
            if edge_tag:  
                edge_gt=[]
                for m in range(label.size(0)):
                    temp=label[m]
                    edge_gt.append(temp)
                edge_gt=torch.cat([edge_gt[0],edge_gt[1],edge_gt[2],edge_gt[3]],dim=0)
                edge_gt=edge_gt.unsqueeze(1)
                edge_loss=5*self.edge_loss_1(edge_out,edge_gt)+Dice(edge_out,edge_gt)
            else:
                edge_loss=0.0
                
            loss=0
         
            loss+=self.loss_computation(pred,label,self.args.warm_epoch,epoch)
            for j in range(len(masks)):
                if j>0:
                    label=self.down(label)
                loss+=self.loss_computation(masks[j],label,self.args.warm_epoch,epoch)
            
            loss/=(len(masks)+1)
            if edge_tag:
                loss+=0.05*min(float(epoch/self.args.epochs),0.5)*edge_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(),pred.size(0))
            process.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
        
        
        # adjust_learning_rate(self.optimizer, epoch, self.args.epochs, self.args.lr,
        #                      self.args.warm_epoch, 1e-6)   
            
            
            
    def test(self,epoch):
        
        self.model.eval()
        self.mIOU.reset()
        self.PD_FA.reset()
        process=tqdm(self.testLoader)
        tag=False
        edge_tag=False
        with torch.no_grad():
            for batch_idx,(img,label) in enumerate(process):
                img=img.to(self.device)
                label=label.to(self.device)
                if epoch>self.args.warm_epoch:
                    tag=True
                if epoch>self.args.warm_epoch*2:
                    edge_tag=True
                loss=0
                _,predicted,edge_out=self.model(img,tag,edge_tag)
                self.mIOU.update(predicted,label)
                self.PD_FA.update(predicted,label)
                self.ROC.update(predicted,label)
                _, mean_IoU = self.mIOU.get()
                pred_img=torch.sigmoid(predicted)
                pred_img=pred_img.cpu().numpy()
                for z in range(pred_img.shape[0]):
                    img_array=pred_img[z,0]
                    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
                    img_array = (img_array * 255).astype(np.uint8)
                    
                    filename = f"epoch{epoch}_batch{batch_idx}_img{z}.png"
                    my_save_path = os.path.join('./my_pics', filename)
                    
                    cv2.imwrite(my_save_path, img_array)
                
                
                process.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA,PD=self.PD_FA.get(len(self.testLoader))
            _, mean_IoU = self.mIOU.get()

            
            if self.mode=='train':
                
                
                
                if mean_IoU>self.best_IOU:
                    self.best_IOU=mean_IoU

                    torch.save(self.model.state_dict(), self.save_folder+'/weight.pkl')
                    with open(os.path.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_IOU, PD[0], FA[0] * 1000000))
                all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch}
                torch.save(all_states, self.save_folder+'/checkpoint.pkl')
                
                
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')
                
                


if __name__=='__main__':
    args=Args()

    trainer=Trainer(args)
    if trainer.mode=='train':
        for epoch in range(args.start_epoch+1,args.epochs+1):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
      


    
