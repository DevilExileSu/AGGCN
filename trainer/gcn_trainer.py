import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from trainer.trainer import Trainer
from utils.metrics import report
from utils import scorer

class GCNTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, cfg, logger, data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model=model, optimizer=optimizer, criterion=criterion, cfg=cfg, logger=logger)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.do_validation = self.valid_data_loader is not None
        self.device = 'cuda:0' if cfg['cuda'] else 'cpu'
        self.log_step = cfg['log_step']
        self.conv_l2 = cfg['conv_l2']
        self.pooling_l2 = cfg['pooling_l2']
        self.max_grad_norm = cfg['max_grad_norm']

    def _train_epoch(self, epoch):
        def get_lr(optimizer):
            return [param['lr'] for param in optimizer.param_groups]
        self.model.train()
        total_loss = 0
        for idx, inputs in enumerate(self.data_loader):
            org_idx = inputs[-2]
            inputs = [item.to(self.device) for item in inputs if not isinstance(item, list)]
            self.optimizer.zero_grad()
            outputs, pooling_out = self.model(inputs[:-1])
            loss = self.criterion(outputs, inputs[-1])
            # if self.conv_l2 > 0:
            #     loss += self.model.conv_l2() * self.conv_l2
            # if self.pooling_l2 > 0:
            #     loss += self.pooling_l2 * (pooling_out ** 2).sum(1).mean()
            loss_val = loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss_val

            if idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {}, {}/{} ({:.0f}%), Loss: {:.6f}'.format(epoch, 
                        idx, 
                        len(self.data_loader), 
                        idx * 100 / len(self.data_loader), 
                        loss.item()
                        ))

        self.logger.info('Train Epoch: {}, total Loss: {:.6f}, mean Loss: {:.6f}'.format(
                epoch,
                total_loss, 
                total_loss / len(self.data_loader)
                ))
        
        if self.do_validation:
            self.logger.debug("start validation")
            val_loss, f1 = self._valid_epoch()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)
        self.logger.info('Train Epoch: {}, current lr is : {}'.format(epoch, get_lr(self.optimizer)))
        return val_loss, f1

    def _valid_epoch(self):
        self.model.eval()
        val_loss = 0
        preds = []
        labels = []
        with torch.no_grad():
            for idx, inputs in enumerate(self.valid_data_loader):
                org_idx = inputs[-2]
                inputs = [item.to(self.device) for item in inputs if not isinstance(item, list)]
                outputs, pooling_out = self.model(inputs[:-1])
                val_loss += self.criterion(outputs, inputs[-1])
                pred = F.softmax(outputs, 1).data.cpu().numpy().tolist()
                pred = np.argmax(pred, axis=1)
                preds.extend(pred)
                labels += inputs[-1].tolist()

        preds = [self.data_loader.id2label[pred] for pred in preds]
        labels = [self.data_loader.id2label[label] for label in labels]

        valid_p, valid_r, valid_f1 = scorer.score(labels, preds, verbose=True)
        self.logger.info(' validation precision is : {:.3f}, validation recall is : {:.3f}, validation f1_macro is : {:.3f}, bset scores is : {}'.format(valid_p, valid_r, valid_f1, self.best_score))
        return val_loss/len(self.valid_data_loader), valid_f1





        