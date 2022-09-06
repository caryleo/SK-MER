import argparse
import torch
from tqdm import tqdm
import model.metric as module_metric
import model.loss as module_loss
from parse_config import ConfigParser
from utils import create_model, create_dataloader, get_concept_embedding, build_kb
from pymagnitude import Magnitude

from torch.utils.data import dataloader

def main(config):
    logger = config.get_logger('test')
    ##############
    data_loader = create_dataloader(config)
    ###############
    model = create_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.initialize(config, device)

    ##############
    # 传入这些矩阵的时候，因为一些权重没有放在GPU上，需要额外to一下
    concept2id_v, concept2id_a, concept2id_t = data_loader.get_concept2ids()
    vectors = Magnitude(config["knowledge"]["embedding_file"])

    embedding_concept_v = get_concept_embedding(concept2id_v, config, vectors)
    embedding_concept_a = get_concept_embedding(concept2id_a, config, vectors)
    embedding_concept_t = get_concept_embedding(concept2id_t, config, vectors)
    edge_matrix_v, affectiveness_v = build_kb(concept2id_v, config, "visual")
    edge_matrix_a, affectiveness_a = build_kb(concept2id_a, config, "audio")
    edge_matrix_t, affectiveness_t = build_kb(concept2id_t, config, "text")

    embedding_concept_v = torch.from_numpy(embedding_concept_v).to(device)
    embedding_concept_a = torch.from_numpy(embedding_concept_a).to(device)
    embedding_concept_t = torch.from_numpy(embedding_concept_t).to(device)
    edge_matrix_v, affectiveness_v = edge_matrix_v.to(device), affectiveness_v.to(device)
    edge_matrix_a, affectiveness_a = edge_matrix_a.to(device), affectiveness_a.to(device)
    edge_matrix_t, affectiveness_t = edge_matrix_t.to(device), affectiveness_t.to(device)

    model.g_att_v.init_params(edge_matrix_v, affectiveness_v, embedding_concept_v, device)
    model.g_att_a.init_params(edge_matrix_a, affectiveness_a, embedding_concept_a, device)
    model.g_att_t.init_params(edge_matrix_t, affectiveness_t, embedding_concept_t, device)
    ####################
    
    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()


    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    loss_fn = getattr(module_loss, config['loss'])
    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, target_loc, umask, seg_len, n_c = [d.to(device) for d in data]
            # seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
            # output = model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, seq_lengths, target_loc, seg_len, n_c)
            target, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl, C_al, C_tl, target_loc, umask, seg_len, n_c = [d.to(device) for d in data]
            seq_lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
            output = model(U_v, U_a, U_t, U_p, M_v, M_a, M_t, C_v, C_a, C_t, C_vl.tolist(), C_al.tolist(), C_tl.tolist(), target_loc, seq_lengths, seg_len, n_c)
            target = target.squeeze(1)
            loss = loss_fn(output, target)
            batch_size = U_v.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
