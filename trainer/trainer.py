'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 12:04:35
LastEditors: Gary Liu
LastEditTime: 2021-12-08 19:31:48
'''
import torch
import numpy as np
from base import BaseTrainer
import datetime
from numpy import inf
from utils import inf_loop, MetricTracker, get_concept_embedding, build_kb, build_vocab, debug_print_dims
from logger import TensorboardWriter
from pymagnitude import Magnitude

class MEmoRTrainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

            self.optimizer.zero_grad()
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

            output = self.model(U_v, U_a, U_t, U_p, M_v, M_a,M_t, seq_lengths, target_loc, seg_len, n_c)
            assert output.shape[0] == target.shape[0]
            target = target.squeeze(1)
            loss = self.criterion(output, target)
            loss.backward()

            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c)
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class MEmoRTrainer_KE(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, config, data_loader,
                 valid_data_loader=None, len_epoch=None):
        # super().__init__(model, criterion, metric_ftns, config)

        # 不继承父类，直接重写

        self.config = config
        self.logger = config.get_logger(
            'trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        model.initialize(config, self.device)

        ##############
        # 传入这些矩阵的时候，因为一些权重没有放在GPU上，需要额外to一下
        self.data_loader = data_loader
        concept2id_v, concept2id_a, concept2id_t = self.data_loader.get_concept2ids()
        vectors = Magnitude(config["knowledge"]["embedding_file"])

        embedding_concept_v = get_concept_embedding(concept2id_v, config, vectors)
        embedding_concept_a = get_concept_embedding(concept2id_a, config, vectors)
        embedding_concept_t = get_concept_embedding(concept2id_t, config, vectors)
        edge_matrix_v, affectiveness_v = build_kb(concept2id_v, config, "visual")
        edge_matrix_a, affectiveness_a = build_kb(concept2id_a, config, "audio")
        edge_matrix_t, affectiveness_t = build_kb(concept2id_t, config, "text")

        self.embedding_concept_v = torch.from_numpy(embedding_concept_v).to(self.device)
        self.embedding_concept_a = torch.from_numpy(embedding_concept_a).to(self.device)
        self.embedding_concept_t = torch.from_numpy(embedding_concept_t).to(self.device)
        self.edge_matrix_v, self.affectiveness_v = edge_matrix_v.to(self.device), affectiveness_v.to(self.device)
        self.edge_matrix_a, self.affectiveness_a = edge_matrix_a.to(self.device), affectiveness_a.to(self.device)
        self.edge_matrix_t, self.affectiveness_t = edge_matrix_t.to(self.device), affectiveness_t.to(self.device)

        model.g_att_v.init_params(self.edge_matrix_v, self.affectiveness_v, self.embedding_concept_v, self.device)
        model.g_att_a.init_params(self.edge_matrix_a, self.affectiveness_a, self.embedding_concept_a, self.device)
        model.g_att_t.init_params(self.edge_matrix_t, self.affectiveness_t, self.embedding_concept_t, self.device)
        ##############

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self.resume = True
            self._resume_checkpoint(config.resume)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            ###########################
            # 这里的接口也调整一下
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl, C_al, C_tl, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
            ###########################

            self.optimizer.zero_grad()
            # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

            ###########################
            # 这里的接口也调整一下
            output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl.tolist(), C_al.tolist(), C_tl.tolist(), target_loc, seq_lengths, seg_len, n_c)
            ###########################
            assert output.shape[0] == target.shape[0]
            target = target.squeeze(1)
            loss = self.criterion(output, target)
            loss.backward()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):

                ###########################
                # 这里的接口也调整一下
                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl, C_al, C_tl, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
                ###########################

                # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                ###########################
                # 这里的接口也调整一下
                output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl.tolist(), C_al.tolist(), C_tl.tolist(), target_loc, seq_lengths, seg_len, n_c)
                ###########################
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class MEmoRTrainer_KE_ALL(BaseTrainer):
    """训练过程管理器实例
    """    
    def __init__(self, model, criterion, metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None):        
        # super().__init__(model, criterion, metric_ftns, config)

        # 不继承父类，直接重写

        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])

        ##############
        self.data_loader = data_loader
        # 因为所有的标签混合在一起了，所以只有一个词汇表
        self.concept2id = self.data_loader.get_concept2ids()
        # config['vocab_size'] = len(self.concept2id)

        model.initialize(config, self.device, len(self.concept2id))

        embedding_concept = get_concept_embedding(self.concept2id, config)
        edge_matrix, affectiveness = build_kb(self.concept2id, config, "all")
        model.g_att.init_params(edge_matrix, affectiveness, embedding_concept, self.device)
        ##############

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        self.resume = False
        self.best_epoch_resume = None

        if config.resume is not None:
            self.resume = True
            self._resume_checkpoint(config.resume)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            ###########################
            # 这里的接口也调整一下
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
            ###########################

            self.optimizer.zero_grad()
            # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            ###########################
            # 这里的接口也调整一下
            output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            ###########################
            assert output.shape[0] == target.shape[0]
            target = target.squeeze(1)
            loss = self.criterion(output, target)
            loss.backward()
            
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):

                ###########################
                # 这里的接口也调整一下
                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]
                ###########################

                # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                ###########################
                # 这里的接口也调整一下
                output = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                ###########################
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class KEERTrainer_KE_CL(BaseTrainer):
    """训练过程管理器实例
    """    
    def __init__(self, model, criterion, metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None, criterion_cl=None):        
        # super().__init__(model, criterion, metric_ftns, config)

        # 不继承父类，直接重写

        # 模型、数据、优化实例的创建部分，不变化
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.data_loader = data_loader
        self.concept2id = self.data_loader.get_concept2ids()
        self.useful_ids = self.data_loader.get_useful_ids()
        # print(self.useful_ids)
        model.initialize(config, self.device, len(self.concept2id))

        embedding_concept = get_concept_embedding(self.concept2id, config)
        edge_matrix, affectiveness = build_kb(self.concept2id, config, "all")
        model.g_att.init_params(edge_matrix, affectiveness, embedding_concept, self.device)

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.criterion_cl = criterion_cl
        self.temperature = config['nce_temperature']
        self.loss_fusion = config['loss_fusion']
        self.margin = config['margin']
        self.metric_ftns = metric_ftns


        #######################
        # 因为我们要把指示表示变成输入，因此知识产生部分不动
        # for name, parameter in model.named_parameters():
        #     if name.startswith('g_att'):
        #         print('!!! Caution! Setting', name, 'with NO Gradient')
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True

        for name, parameter in model.named_parameters():
            if name.startswith('memory'):
                print('!!! Caution! Setting', name, 'with NO Gradient')
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
            
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        #######################

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 优化策略和配置，不动
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # 日志配置，不动
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        ###############################################################
        # 这里是模型继续进行训练的部分，因此对比学习现有的实现方案直接从这里开始
        # 记在模型后，进入到对比学习优化模式
        # 所以在这个阶段训练时，不需要开启resume
        self.resume = False
        self.best_epoch_resume = None

        if config.resume is not None:
            self.resume = True
            self._resume_checkpoint(config.resume, cl=True)
        else:
            # raise RuntimeError("WRONG CONFIGURATION FOR CONTRAST LEARNING SETTINGS!")
            pass
        ################################################################


        # 有关于验证、模型选择和优化配置的部分，不动
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', 'loss_base', 'loss_cl', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train() # 原来这部分不会影响梯度计算呀
        
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):

            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

            # if torch.any(concepts_length <= 0):
            #     print(concepts_length)
            # 初始训练
            self.optimizer.zero_grad()
            # print("POS")
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            # output, score_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            output, knowledge_batch_list, feature_batch_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, score_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, score_batch = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, local_nodes_batch_list, kecrs_batch_list, feature_batch_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)

            assert output.shape[0] == target.shape[0]
            
            loss_base = self.criterion(output, target.squeeze(1))

            if (loss_base != loss_base).any():
                raise ValueError("NaN base loss!")

            # target_onehot = torch.zeros_like(output).scatter_(1, target, 1)
            # grad_local_node_list = torch.autograd.grad((output * target_onehot).sum(), local_nodes_batch_list, create_graph=True)
           
            # ### 根据梯度情况计算正负mask
            # mask_pos_list, mask_neg_list = [], []
            # for index, grad_local_node in enumerate(grad_local_node_list):
            #     # 正负样本：取top 50（113XX个单词）
            #     if grad_local_node is None:
            #         print(grad_local_node_list[index])
            #         print(concepts_length[index])
            #     grad_local_node = grad_local_node.sum(dim=-1) # seq_len, vocab_size
            #     _, index_asc = torch.sort(grad_local_node) # seq_len, vocab_size
            #     index_pos = index_asc[:, -10:] # 最后面50个是梯度最大的
            #     index_neg = index_asc[:, 10:] # 最前面50个是梯度最小的

            #     mask_pos = torch.zeros_like(grad_local_node).scatter_(1, index_pos, 1) # seq_len, vocab_size
            #     mask_neg = torch.zeros_like(grad_local_node).scatter_(1, index_neg, 1) # seq_len, vocab_size

            #     mask_pos_list.append(mask_pos)
            #     mask_neg_list.append(mask_neg)

            # # 正负knowledge表示获得
            # _, kecrs_pos_batch_list, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, mask_pos_list)
            # _, kecrs_neg_batch_list, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, mask_neg_list)

            # # kecrs_pos_batch_list, kecrs_neg_batch_list = None, None

            # # 对比损失计算
            # loss_cl = self.criterion_cl(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, self.temperature)
        
           
            # 第一个方案，计算bilinear相似度的loss
            # loss_cl = self.criterion_cl(score_list, self.margin)
            # loss_cl = self.criterion_cl(score_list, self.temperature)
            # loss_cl = self.criterion_cl(score_batch, self.temperature)
            loss_cl = self.criterion_cl(knowledge_batch_list, feature_batch_list, self.temperature)
            # loss_cl = self.criterion_cl(score, score_neg, k_index_neg_list, self.temperature)
            loss = loss_base + self.loss_fusion * loss_cl
            # loss = loss_base + loss_cl
 

            loss.backward()

            if (loss != loss).any():
                raise ValueError("NaN loss!")

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_base', loss_base.item())
            self.train_metrics.update('loss_cl', loss_cl.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Base Loss {:.6f} CL Loss {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_base.item(),
                    loss_cl.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        # 优化效率调整部分
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        # 因为整体上来讲不影响验证集测试，因此这部分不做变动
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):

                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

                # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                # output, _, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                output, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                # self.valid_metrics.update('loss_base', loss_base.item())
                # self.valid_metrics.update('loss_cl', loss_cl.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class KEERTrainer_KE_CL_CK(BaseTrainer):
    """训练过程管理器实例
    """    
    def __init__(self, model, criterion, metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None, criterion_cl=None):        
        # super().__init__(model, criterion, metric_ftns, config)

        # 不继承父类，直接重写

        # 模型、数据、优化实例的创建部分，不变化
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.data_loader = data_loader
        self.concept2id = self.data_loader.get_concept2ids()
        self.useful_ids = self.data_loader.get_useful_ids()
        # print(self.useful_ids)
        model.initialize(config, self.device, len(self.concept2id))

        embedding_concept = get_concept_embedding(self.concept2id, config)
        edge_matrix, affectiveness = build_kb(self.concept2id, config, "all")
        model.g_att.init_params(edge_matrix, affectiveness, embedding_concept, self.device)

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.criterion_cl = criterion_cl
        self.temperature = config['nce_temperature']
        self.loss_fusion = config['loss_fusion']
        self.margin = config['margin']
        self.metric_ftns = metric_ftns


        #######################
        # 因为我们要把指示表示变成输入，因此知识产生部分不动
        # for name, parameter in model.named_parameters():
        #     if name.startswith('g_att'):
        #         print('!!! Caution! Setting', name, 'with NO Gradient')
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True

        for name, parameter in model.named_parameters():
            if name.startswith('memory'):
                print('!!! Caution! Setting', name, 'with NO Gradient')
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
            
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        #######################

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 优化策略和配置，不动
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # 日志配置，不动
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        ###############################################################
        # 这里是模型继续进行训练的部分，因此对比学习现有的实现方案直接从这里开始
        # 记在模型后，进入到对比学习优化模式
        # 所以在这个阶段训练时，不需要开启resume
        self.resume = False
        self.best_epoch_resume = None

        if config.resume is not None:
            self.resume = True
            self._resume_checkpoint(config.resume, cl=True)
        else:
            # raise RuntimeError("WRONG CONFIGURATION FOR CONTRAST LEARNING SETTINGS!")
            pass
        ################################################################


        # 有关于验证、模型选择和优化配置的部分，不动
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', 'loss_base', 'loss_cl', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train() # 原来这部分不会影响梯度计算呀
        
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):

            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

            # if torch.any(concepts_length <= 0):
            #     print(concepts_length)
            # 初始训练
            self.optimizer.zero_grad()
            # print("POS")
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            # output, score_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, kecrs_batch_list, feature_batch_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, score_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output, score_batch = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            output, kecrs_batch_list, feature_batch_list, local_nodes_batch_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)

            assert output.shape[0] == target.shape[0]
            
            loss_base = self.criterion(output, target.squeeze(1))

            if (loss_base != loss_base).any():
                raise ValueError("NaN base loss!")

            target_onehot = torch.zeros_like(output).scatter_(1, target, 1)
            grad_local_node_list = torch.autograd.grad((output * target_onehot).sum(), local_nodes_batch_list, create_graph=True)
           
            ### 根据梯度情况计算正负mask
            mask_pos_list, mask_neg_list = [], []
            for index, grad_local_node in enumerate(grad_local_node_list):
                # 正负样本：取top 50（113XX个单词）
                if grad_local_node is None:
                    print(grad_local_node_list[index])
                    print(concepts_length[index])
                grad_local_node = grad_local_node.sum(dim=-1) # seq_len, vocab_size
                _, index_asc = torch.sort(grad_local_node) # seq_len, vocab_size
                index_pos = index_asc[:, -50:] # 最后面50个是梯度最大的
                # index_neg = index_asc[:, 10:] # 最前面50个是梯度最小的
                index_neg = index_asc[:, :50] # 最前面50个是梯度最小的

                mask_pos = torch.zeros_like(grad_local_node).scatter_(1, index_pos, 1) # seq_len, vocab_size
                mask_neg = torch.zeros_like(grad_local_node).scatter_(1, index_neg, 1) # seq_len, vocab_size

                mask_pos_list.append(mask_pos)
                mask_neg_list.append(mask_neg)

            # 正负knowledge表示获得
            # kecrs_pos_batch_list, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, mask_pos_list)
            kecrs_neg_batch_list, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, mask_neg_list)

            # kecrs_pos_batch_list, kecrs_neg_batch_list = None, None

            # 对比损失计算
            kecrs_pos_batch_list = kecrs_batch_list
            loss_cl = self.criterion_cl(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, self.temperature)
        
           
            # 第一个方案，计算bilinear相似度的loss
            # loss_cl = self.criterion_cl(score_list, self.margin)
            # loss_cl = self.criterion_cl(score_list, self.temperature)
            # loss_cl = self.criterion_cl(score_batch, self.temperature)
            # loss_cl = self.criterion_cl(kecrs_batch_list, feature_batch_list, self.temperature)
            # loss_cl = self.criterion_cl(score, score_neg, k_index_neg_list, self.temperature)
            loss = loss_base + self.loss_fusion * loss_cl
            # loss = loss_base + loss_cl
 

            loss.backward()

            if (loss != loss).any():
                raise ValueError("NaN loss!")

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_base', loss_base.item())
            self.train_metrics.update('loss_cl', loss_cl.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Base Loss {:.6f} CL Loss {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_base.item(),
                    loss_cl.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        # 优化效率调整部分
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        # 因为整体上来讲不影响验证集测试，因此这部分不做变动
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):

                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

                # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                output, _, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                # output, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                # self.valid_metrics.update('loss_base', loss_base.item())
                # self.valid_metrics.update('loss_cl', loss_cl.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)


class KEERTrainer_KE_CL_2(BaseTrainer):
    """训练过程管理器实例
    """    
    def __init__(self, model, criterion, metric_ftns, config, data_loader, valid_data_loader=None, len_epoch=None, criterion_cl=None):        
        # super().__init__(model, criterion, metric_ftns, config)

        # 不继承父类，直接重写

        # 模型、数据、优化实例的创建部分，不变化
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.device, device_ids = self._prepare_device(config['n_gpu'])

        self.data_loader = data_loader
        self.concept2id = self.data_loader.get_concept2ids()
        self.useful_ids = self.data_loader.get_useful_ids()
        # print(self.useful_ids)
        model.initialize(config, self.device, len(self.concept2id))

        embedding_concept = get_concept_embedding(self.concept2id, config)
        edge_matrix, affectiveness = build_kb(self.concept2id, config, "all")
        model.g_att.init_params(edge_matrix, affectiveness, embedding_concept, self.device)

        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.criterion_cl = criterion_cl
        self.temperature = config['nce_temperature']
        self.loss_fusion = config['loss_fusion']
        self.margin = config['margin']
        self.metric_ftns = metric_ftns


        #######################
        # 因为我们要把指示表示变成输入，因此知识产生部分不动
        # for name, parameter in model.named_parameters():
        #     if name.startswith('g_att'):
        #         print('!!! Caution! Setting', name, 'with NO Gradient')
        #         parameter.requires_grad = False
        #     else:
        #         parameter.requires_grad = True

        for name, parameter in model.named_parameters():
            if name.startswith('memory'):
                print('!!! Caution! Setting', name, 'with NO Gradient')
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True
            
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        #######################

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')

        # 优化策略和配置，不动
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # 日志配置，不动
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        ###############################################################
        # 这里是模型继续进行训练的部分，因此对比学习现有的实现方案直接从这里开始
        # 记在模型后，进入到对比学习优化模式
        # 所以在这个阶段训练时，不需要开启resume
        self.resume = False
        self.best_epoch_resume = None

        if config.resume is not None:
            self.resume = True
            self._resume_checkpoint(config.resume, cl=True)
        else:
            # raise RuntimeError("WRONG CONFIGURATION FOR CONTRAST LEARNING SETTINGS!")
            pass
        ################################################################


        # 有关于验证、模型选择和优化配置的部分，不动
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        self.log_step = 200
        self.train_metrics = MetricTracker('loss', 'loss_base', 'loss_cl', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train() # 原来这部分不会影响梯度计算呀
        
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):

            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

            # 初始训练
            self.optimizer.zero_grad()
            # print("POS")
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            output, kecrs_batch_list, score_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)

            assert output.shape[0] == target.shape[0]
            # target = target.squeeze(1)
            
            loss_base = self.criterion(output, target.squeeze(1))
            # loss_base.backward(retain_graph=True)

            if (loss_base != loss_base).any():
                raise ValueError("NaN base loss!")

            # C_pos = torch.zeros_like(C)
            # C_neg = torch.zeros_like(C) # batch, seglen, concept_length (padded)

            # for i in range(C.size(0)):
            #     for j in range(seg_len[i]):
            #         if concepts_length[i][j].item() > 0:
            #             C_pos[i, j, :concepts_length[i][j]] = C[i, j, :concepts_length[i][j]]
            #             C_pos[i, j, concepts_length[i][j]: concepts_length[i][j] + 1] = torch.from_numpy(np.random.choice(self.useful_ids, 1))

            #             C_neg[i, j, :concepts_length[i, j]] = torch.from_numpy(np.random.choice(self.useful_ids, concepts_length[i, j].item()))
            #         else:
            #             C_pos[i, j, :concepts_length[i][j]] = C[i, j, :concepts_length[i][j]]
                        
            #             C_neg[i, j, :1] = torch.from_numpy(np.random.choice(self.useful_ids, 1))
            # output_pos, _, feature_pos = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_pos, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            # output_neg, _, feature_neg = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_neg, concepts_length, target_loc, seq_lengths, seg_len, n_c)
            
            # 计算梯度
            target_onehot = torch.zeros_like(output).scatter_(1, target, 1)
            kecrs_grads = torch.autograd.grad((output * target_onehot).sum(), kecrs_batch_list, create_graph=True) # tuple: batch, seg_len, dim_k

            # 负样本mask生成，这里的mask是指每个batch中每个seg的mask，根据梯度判断，面向最大梯度的调整
            # k_mask_neg_list = list()
            k_index_neg_list = list()
            for kecrs_grad in kecrs_grads:
                k_grad = kecrs_grad.sum(1) # seg_len
                k_grad_values, k_grad_indices = k_grad.sort(0) # seg_len
                
                if k_grad_values.size(0) < 2:
                    # 就一个片段
                    # k_mask_n = torch.zeros_like(k_grad_values, dtype=torch.int8).to(self.device)
                    # k_mask_neg_list.append(k_mask_n)
                    k_index_neg_list.append(0)
                else:
                    # k_mask_n = torch.ones_like(k_grad_values, dtype=torch.int8).to(self.device)
                    # k_mask_n[k_grad_indices[-1]] = 0
                    # k_mask_neg_list.append(k_mask_n)
                    k_index_neg_list.append(k_grad_indices[-1])

            # 针对负样本，使用随机抽取的概念替换上述负样本行
            C_neg = torch.zeros_like(C)
            for i in range(C.size(0)):
                for j in range(seg_len[i]):
                    if j == k_index_neg_list[i]:
                        if concepts_length[i][j] > 0:
                            C_neg[i, j, :concepts_length[i, j]] = torch.from_numpy(np.random.choice(self.useful_ids, concepts_length[i, j].item()))
                        else:
                            # 如果原始的概念长度为0，则不需要替换
                            C_neg[i, j, :concepts_length[i, j]] = C[i, j, :concepts_length[i, j]]
                    else:
                        C_neg[i, j, :concepts_length[i, j]] = C[i, j, :concepts_length[i, j]]
                    
            C_neg.to(C.device)
            # print("NEG")
            score_neg_list = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_neg, concepts_length, target_loc, seq_lengths, seg_len, n_c, contrastive_flag=True)

            loss_cl = self.criterion_cl(score_list, score_neg_list, k_index_neg_list, self.margin)
            # loss_cl = self.criterion_cl(score, score_neg, k_index_neg_list, self.temperature)
            # loss = loss_base + self.loss_fusion * loss_cl
            loss = loss_base + 10 * loss_cl
 
            # 生成正负mask

            # k_mask_pos_list = list()
            # k_mask_neg_list = list()
            
            # for kecrs_grad in kecrs_grads:
            #     k_grad = kecrs_grad.sum(1) # seg_len
            #     k_grad_values, k_grad_indices = k_grad.sort(0) # seg_len
            #     # print(k_grad_values, k_grad_indices)

            #     if k_grad_values.size(0) < 2:
            #         # 只有一个片段，正样本保持不变（全1），负样本扣掉（全0）
            #         k_mask_p = torch.ones_like(k_grad_values, dtype=torch.int8).to(self.device)
            #         k_mask_pos_list.append(k_mask_p)

            #         k_mask_n = torch.zeros_like(k_grad_values, dtype=torch.int8).to(self.device)
            #         k_mask_neg_list.append(k_mask_n)
            #     else:
            #         # 多个片段，正样本抹掉最小，负样本抹掉最大
            #         k_mask_p = torch.ones_like(k_grad_values, dtype=torch.int8).to(self.device)
            #         k_mask_p[k_grad_indices[0]] = 0
            #         k_mask_pos_list.append(k_mask_p)

            #         k_mask_n = torch.ones_like(k_grad_values, dtype=torch.int8).to(self.device)
            #         k_mask_n[k_grad_indices[-1]] = 0
            #         k_mask_neg_list.append(k_mask_n)

            # # 正样本训练
            # # self.optimizer.zero_grad()
            # # print("POS")
            # output_pos, _, feature_pos = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, k_mask_pos_list)

            # # 负样本训练
            # # self.optimizer.zero_grad()
            # # print("NEG")
            # output_neg, _, feature_neg = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c, k_mask_neg_list)

            # # loss_cl = self.criterion_cl(feature, feature_pos, feature_neg, self.margin)
            # loss_cl = self.criterion_cl(target, output_pos, output_neg, self.margin)
            # # loss_cl = self.criterion_cl(feature, feature_pos, feature_neg, self.temperature)

            # loss = loss_base + self.loss_fusion * loss_cl
            # print(loss_base, loss_cl, loss)


            loss.backward()

            if (loss != loss).any():
                raise ValueError("NaN loss!")

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            self.train_metrics.update('loss_base', loss_base.item())
            self.train_metrics.update('loss_cl', loss_cl.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f} Base Loss {:.6f} CL Loss {:.6f} Time:{}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_base.item(),
                    loss_cl.item(),
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        # 优化效率调整部分
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """

        # 因为整体上来讲不影响验证集测试，因此这部分不做变动
        self.model.eval()
        self.valid_metrics.reset()
        outputs, targets = [], []
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):

                target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, umask, seg_len, n_c = [d.to(self.device) for d in data]

                # 这里计算的是各个序列的长度，因为1是连续的，0是后pad的，所以找到最后一个1就可以得到长度了
                seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

                output, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                output, _, _ = self.model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concepts_length, target_loc, seq_lengths, seg_len, n_c)
                target = target.squeeze(1)
                loss = self.criterion(output, target)

                outputs.append(output.detach())
                targets.append(target.detach())

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                # self.valid_metrics.update('loss_base', loss_base.item())
                # self.valid_metrics.update('loss_cl', loss_cl.item())

            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(outputs, targets))

        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

