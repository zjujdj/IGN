import pickle
import os
import pandas as pd
import re
from rdkit import Chem
from prody import *
from graph_constructor import GraphDatasetIGN, collate_fn_ign
import argparse
from utils import *
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model_v2 import IGN
from sklearn.metrics import mean_absolute_error, mean_squared_error


# def pocket_truncate(protein_file, ligand_file, pocket_out_file, complex_out_file, distance=5, sanitize=True):
#     ligand = Chem.MolFromMolFile(ligand_file, sanitize=sanitize)
#     structure = parsePDB(protein_file)
#     if ligand and structure:
#         protein = structure.select('protein')  # remove water and other useless
#         selected = protein.select('same residue as within %s of ligand' % distance, ligand=ligand.GetConformer().GetPositions())
#         writePDB(pocket_out_file, selected)  # contain H
#         # to prepare inputs for the model
#         pocket = Chem.MolFromPDBFile(pocket_out_file, sanitize=sanitize)  # not contain H
#         if pocket:
#             Chem.MolToPDBFile(pocket, pocket_out_file)  # not contain H
#             with open(complex_out_file, 'wb') as f:
#                 pickle.dump([ligand, pocket], f)
#         else:
#             print('pocket file read error for %s' % pocket_out_file)
#     elif ligand is None and structure is not None:
#         print('only ligand file read error for %s' % ligand_file)
#     elif structure is None and ligand is not None:
#         print('only protein file read error for %s' % protein_file)
#     else:
#         print('both protein file and ligand file read error for %s' % protein_file)

# pdb_home_path = '/home/dejun/work_spaces/py/new/dataset/pdbbind2016_all'
# path_marker = '/'
# targets = []
# sets = []
# refined_targets = os.listdir(pdb_home_path + path_marker + 'refined-set')
# for target in refined_targets:
#     if len(target) == 4:
#         targets.append(target)
#         sets.append('refined-set')
#
# general_targets = os.listdir(pdb_home_path + path_marker + 'general-set')
# for target in general_targets:
#     if len(target) == 4:
#         targets.append(target)
#         sets.append('general-set')
# ligand_files = []
# protein_files = []
#
# complex_dir = '/home/dejun/work_spaces/py/new/dataset/hxp/pdb2016_glide_best_pose_complex'
# pocket_dir = '/home/dejun/work_spaces/py/new/dataset/hxp/pdb2016_glide_best_pose_pocket'
# path_marker = '/'
# complex_files = []
# pocket_files = []
# for i, target in enumerate(targets):
#     file = pdb_home_path + path_marker + sets[i] + path_marker + '%s' % targets[i] + path_marker + '%s_pose_gen' % targets[i] + path_marker + 'SP.csv'
#     if os.path.exists(file):
#         files = os.listdir(pdb_home_path + path_marker + sets[i] + path_marker + '%s' % targets[i] + path_marker + '%s_pose_gen' % targets[i])
#         for file in files:
#             if re.search('pose0', file):
#                 ligand_files.append(pdb_home_path + path_marker + sets[i] + path_marker + '%s' % targets[i] + path_marker + '%s_pose_gen' % targets[i] + path_marker + file)
#                 protein_files.append(pdb_home_path + path_marker + sets[i] + path_marker + '%s' % targets[i] + path_marker + '%s_prot' % targets[i] + path_marker + '%s_p.pdb' % targets[i])
#                 complex_files.append(complex_dir + path_marker + target)
#                 pocket_files.append(pocket_dir + path_marker + target + '_pkt.pdb')
# print(len(protein_files))
# print(protein_files[0:10])
#
# import multiprocessing
# if __name__ == '__main__':
#     limit = None
#     pool = multiprocessing.Pool(24)
#     pool.starmap(pocket_truncate, zip(protein_files[:limit], ligand_files[:limit], pocket_files[:limit], complex_files[:limit]))
#     pool.close()
#     pool.join()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, bg3, Ys, _ = batch
        bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
        outputs, weights = model(bg, bg3)
        loss = loss_fn(outputs, Ys)
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validation_dataloader, device):
    true = []
    pred = []
    key = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validation_dataloader):
            # DTIModel.zero_grad()
            bg, bg3, Ys, keys = batch
            bg, bg3, Ys = bg.to(device), bg3.to(device), Ys.to(device)
            outputs, weights = model(bg, bg3)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs.data.cpu().numpy())
            key.append(keys)
    return true, pred, key, weights.data.cpu().numpy()


test_scrpits = False

# home_path = r'G:\02ReData201904034\hxp'
# data_split_file = r'G:\05Coding\GNN_DTI-master\dpi_v4\dataset_5A\PDB2016All_Splits_new_1.pkl'
# pdb_info_file = r'G:\02ReData201904034\dpi_3D\dataset\PDB2016ALL.csv'
# complex_path = r'G:\02ReData201904034\hxp\pdb2016_glide_best_pose_complex'
# path_marker = '\\'

home_path = '/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/hxp'
data_split_file = '/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/hxp/PDB2016All_Splits_new_1.pkl'
pdb_info_file = '/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/hxp/PDB2016ALL.csv'
complex_path = '/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/hxp/pdb2016_glide_best_pose_complex'
path_marker = '/'

with open(data_split_file, 'rb') as f:
    PDB2016All_Splits = pickle.load(f)
train_labels = PDB2016All_Splits['train_labels']
train_dirs = PDB2016All_Splits['train_dirs']
train_keys = PDB2016All_Splits['train_keys']
valid_labels = PDB2016All_Splits['valid_labels']
valid_dirs = PDB2016All_Splits['valid_dirs']
valid_keys = PDB2016All_Splits['valid_keys']
test_labels = PDB2016All_Splits['test_labels']
test_dirs = PDB2016All_Splits['test_dirs']
test_keys = PDB2016All_Splits['test_keys']
train_sets = [dir.split('\\')[-2] for dir in train_dirs]
valid_sets = [dir.split('\\')[-2] for dir in valid_dirs]
test_sets = [dir.split('\\')[-2] for dir in test_dirs]

train_labels_new, train_keys_new = [], []
valid_labels_new, valid_keys_new = [], []
test_labels_new, test_keys_new = [], []

for key, label in zip(train_keys, train_labels):
    if os.path.exists(complex_path + path_marker + key):
        train_keys_new.append(key)
        train_labels_new.append(label)
for key, label in zip(valid_keys, valid_labels):
    if os.path.exists(complex_path + path_marker + key):
        valid_keys_new.append(key)
        valid_labels_new.append(label)
for key, label in zip(test_keys, test_labels):
    if os.path.exists(complex_path + path_marker + key):
        test_keys_new.append(key)
        test_labels_new.append(label)

pdb_info = pd.read_csv(pdb_info_file)
outputs = [(key, label) for (key, label) in zip(train_keys_new, train_labels_new) if
           pdb_info[pdb_info['PDB'] == key]['type'].values[0] != 'IC50']
train_keys_new, train_labels_new = map(list, zip(*outputs))
outputs = [(key, label) for (key, label) in zip(valid_keys_new, valid_labels_new) if
           pdb_info[pdb_info['PDB'] == key]['type'].values[0] != 'IC50']
valid_keys_new, valid_labels_new = map(list, zip(*outputs))
outputs = [(key, label) for (key, label) in zip(test_keys_new, test_labels_new) if
           pdb_info[pdb_info['PDB'] == key]['type'].values[0] != 'IC50']
test_keys_new, test_labels_new = map(list, zip(*outputs))
train_dirs_new = [complex_path + path_marker + key for key in train_keys_new]
valid_dirs_new = [complex_path + path_marker + key for key in valid_keys_new]
test_dirs_new = [complex_path + path_marker + key for key in test_keys_new]
print('train data:', len(train_keys_new))
print('valid data:', len(valid_keys_new))
print('test data:', len(test_keys_new))

# # 口袋截断(外部预测集)
# # 4csj
# sdfs = os.listdir(home_path + path_marker + 'sdfs_4csj')
# ligand_files = [home_path + path_marker + 'sdfs_4csj' + path_marker + _ for _ in sdfs]
# protein_files = [home_path + path_marker + '4csj_rep.pdb' for _ in sdfs]
# pocket_out_files = [home_path + path_marker + '4csj_pocket' + path_marker + _ + '_pkt.pdb' for _ in sdfs]
# complex_out_files = [home_path + path_marker + '4csj_complex' + path_marker + _ for _ in sdfs]
# pool = mp.Pool(8)
# pool.starmap(pocket_truncate, zip(protein_files, ligand_files, pocket_out_files, complex_out_files))
# pool.close()
# pool.join()
#
# # 5g5w
# sdfs = os.listdir(home_path + path_marker + 'sdfs_5g5w')
# ligand_files = [home_path + path_marker + 'sdfs_5g5w' + path_marker + _ for _ in sdfs]
# protein_files = [home_path + path_marker + 'prep_5g5w_rep.pdb' for _ in sdfs]
# pocket_out_files = [home_path + path_marker + '5g5w_pocket' + path_marker + _ + '_pkt.pdb' for _ in sdfs]
# complex_out_files = [home_path + path_marker + '5g5w_complex' + path_marker + _ for _ in sdfs]
# pool = mp.Pool(8)
# pool.starmap(pocket_truncate, zip(protein_files, ligand_files, pocket_out_files, complex_out_files))
# pool.close()
# pool.join()

_4csj_keys = os.listdir(home_path + path_marker + '4csj_complex')
_4csj_labels = [0 for _ in _4csj_keys]
_4csj_dirs = [home_path + path_marker + '4csj_complex' + path_marker + _ for _ in _4csj_keys]

_5g5w_keys = os.listdir(home_path + path_marker + '5g5w_complex')
_5g5w_labels = [0 for _ in _5g5w_keys]
_5g5w_dirs = [home_path + path_marker + '5g5w_complex' + path_marker + _ for _ in _5g5w_keys]

print('4csj data', len(_4csj_keys))
print('5g5w data', len(_5g5w_keys))
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.5, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=5000, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=50, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=0.00, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=5, help="the number of independent runs")
    argparser.add_argument('--node_feat_size', type=int, default=54 + 40)  # both acsf feature and basic atom feature
    argparser.add_argument('--edge_feat_size_2d', type=int, default=12)
    argparser.add_argument('--edge_feat_size_3d', type=int, default=21)
    argparser.add_argument('--graph_feat_size', type=int, default=256)
    argparser.add_argument('--num_layers', type=int, default=3, help='the number of intra-molecular layers')
    argparser.add_argument('--outdim_g3', type=int, default=200, help='the output dim of inter-molecular layers')
    argparser.add_argument('--d_FC_layer', type=int, default=200, help='the hidden layer size of task networks')
    argparser.add_argument('--n_FC_layer', type=int, default=2, help='the number of hidden layers of task networks')
    argparser.add_argument('--dropout', type=float, default=0.25, help='dropout ratio')
    argparser.add_argument('--n_tasks', type=int, default=1)
    argparser.add_argument('--num_workers', type=int, default=0,
                           help='number of workers for loading data in Dataloader')
    argparser.add_argument('--num_process', type=int, default=10,
                           help='number of process for generating graphs')
    argparser.add_argument('--dic_path_suffix', type=str, default='0')

    # paras acsf setting
    argparser.add_argument('--EtaR', type=float, default=4.00, help='EtaR')
    argparser.add_argument('--ShfR', type=float, default=3.17, help='ShfR')
    argparser.add_argument('--Zeta', type=float, default=8.00, help='Zeta')
    argparser.add_argument('--ShtZ', type=float, default=3.14, help='ShtZ')
    args = argparser.parse_args()
    print(args)
    lr, epochs, batch_size, num_workers = args.lr, args.epochs, args.batch_size, args.num_workers
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions

    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks= args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks
    dic_path_suffix = args.dic_path_suffix
    num_process = args.num_process

    # paras for acsf setting
    EtaR, ShfR, Zeta, ShtZ = args.EtaR, args.ShfR, args.Zeta, args.ShtZ

    if test_scrpits:
        epochs = 5
        repetitions = 3

        train_keys_new = train_keys_new[:100]
        train_labels_new = train_labels_new[:100]
        train_dirs_new = train_dirs_new[:100]

        valid_keys_new = valid_keys_new[:100]
        valid_labels_new = valid_labels_new[:100]
        valid_dirs_new = valid_dirs_new[:100]

        test_keys_new = test_keys_new[:100]
        test_labels_new = test_labels_new[:100]
        test_dirs_new = test_dirs_new[:100]

        _4csj_keys = _4csj_keys[:100]
        _4csj_labels = _4csj_labels[:100]
        _4csj_dirs = _4csj_dirs[:100]

        _5g5w_keys = _5g5w_keys[:100]
        _5g5w_labels = _5g5w_labels[:100]
        _5g5w_dirs = _5g5w_dirs[:100]

    # generating the graph objective using multi process
    train_dataset = GraphDatasetIGN(keys=train_keys_new, labels=train_labels_new, data_dirs=train_dirs_new,
                                    graph_ls_file=home_path + path_marker + 'train_data_best_pose.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')
    valid_dataset = GraphDatasetIGN(keys=valid_keys_new, labels=valid_labels_new, data_dirs=valid_dirs_new,
                                    graph_ls_file=home_path + path_marker + 'valid_data_best_pose.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')
    test_dataset = GraphDatasetIGN(keys=test_keys_new, labels=test_labels_new, data_dirs=test_dirs_new,
                                   graph_ls_file=home_path + path_marker + 'test_data_best_pose.pkl',
                                   graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                   dis_threshold=8.00, path_marker='/')

    _4csj_dataset = GraphDatasetIGN(keys=_4csj_keys, labels=_4csj_labels, data_dirs=_4csj_dirs,
                                    graph_ls_file=home_path + path_marker + '4csj.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')
    _5g5w_dataset = GraphDatasetIGN(keys=_5g5w_keys, labels=_5g5w_labels, data_dirs=_5g5w_dirs,
                                    graph_ls_file=home_path + path_marker + '5g5w.pkl',
                                    graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                    dis_threshold=8.00, path_marker='/')
    stat_res = []
    print('the number of train data:', len(train_dataset))
    print('the number of valid data:', len(valid_dataset))
    print('the number of test data:', len(test_dataset))
    print('the number of 4csj data:', len(_4csj_dataset))
    print('the number of 5g5w data:', len(_5g5w_dataset))
    for repetition_th in range(repetitions):
        dt = datetime.datetime.now()
        filename = home_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        set_random_seed(repetition_th)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)

        # model
        DTIModel = IGN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d,
                       num_layers=num_layers,
                       graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                       d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout,
                       n_tasks=n_tasks)
        print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
        if repetition_th == 0:
            print(DTIModel)
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
        DTIModel.to(device)
        optimizer = torch.optim.Adam(DTIModel.parameters(), lr=lr, weight_decay=l2)

        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance, filename=filename)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(DTIModel, loss_fn, train_dataloader, optimizer, device)

            # validation
            train_true, train_pred, _, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
            valid_true, valid_pred, _, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))
            early_stop = stopper.step(valid_rmse, DTIModel)
            end = time.time()
            if early_stop:
                break
            print(
                "epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (
                    epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(DTIModel)
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn_ign)
        _4csj_dataloader = DataLoaderX(_4csj_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)
        _5g5w_dataloader = DataLoaderX(_5g5w_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                       collate_fn=collate_fn_ign)

        train_true, train_pred, tr_keys, _ = run_a_eval_epoch(DTIModel, train_dataloader, device)
        valid_true, valid_pred, val_keys, _ = run_a_eval_epoch(DTIModel, valid_dataloader, device)
        test_true, test_pred, te_keys, _ = run_a_eval_epoch(DTIModel, test_dataloader, device)
        _5g5w_true, _5g5w_pred, _5g5w_keys, _ = run_a_eval_epoch(DTIModel, _5g5w_dataloader, device)
        _4csj_true, _4csj_pred, _4csj_keys, _ = run_a_eval_epoch(DTIModel, _4csj_dataloader, device)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()
        tr_keys = np.concatenate(np.array(tr_keys), 0).flatten()

        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        val_keys = np.concatenate(np.array(val_keys), 0).flatten()

        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        te_keys = np.concatenate(np.array(te_keys), 0).flatten()

        _4csj_true = np.concatenate(np.array(_4csj_true), 0).flatten()
        _4csj_pred = np.concatenate(np.array(_4csj_pred), 0).flatten()
        _4csj_keys = np.concatenate(np.array(_4csj_keys), 0).flatten()

        _5g5w_true = np.concatenate(np.array(_5g5w_true), 0).flatten()
        _5g5w_pred = np.concatenate(np.array(_5g5w_pred), 0).flatten()
        _5g5w_keys = np.concatenate(np.array(_5g5w_keys), 0).flatten()

        pd_tr = pd.DataFrame({'keys': tr_keys, 'train_true': train_true, 'train_pred': train_pred})
        pd_va = pd.DataFrame({'keys': val_keys, 'valid_true': valid_true, 'valid_pred': valid_pred})
        pd_te = pd.DataFrame({'keys': te_keys, 'test_true': test_true, 'test_pred': test_pred})
        pd_4csj = pd.DataFrame({'keys': _4csj_keys, 'test_true': _4csj_true, 'test_pred': _4csj_pred})
        pd_5g5w = pd.DataFrame({'keys': _5g5w_keys, 'test_true': _5g5w_true, 'test_pred': _5g5w_pred})

        pd_4csj.sort_values(by=['test_pred'], inplace=True, ascending=False)
        pd_4csj.reset_index(inplace=True, drop=True)
        print('the rank of K292-1866 in 4csj is:', pd_4csj[pd_4csj['keys']=='K292-1866.sdf'].index)

        pd_5g5w.sort_values(by=['test_pred'], inplace=True, ascending=False)
        pd_5g5w.reset_index(inplace=True, drop=True)
        print('the rank of K292-1866 in 5g5w is:', pd_5g5w[pd_5g5w['keys']=='K292-1866.sdf'].index)

        pd_tr.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_tr.csv'.
                     format(dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_va.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_va.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_te.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_te.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_4csj.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_4csj.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
        pd_5g5w.to_csv(home_path + path_marker + 'stats/{}_{:02d}_{:02d}_{:02d}_{:d}_5g5w.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)

        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
                                                    r2_score(train_true, train_pred), \
                                                    mean_absolute_error(train_true, train_pred), \
                                                    pearsonr(train_true, train_pred)
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(mean_squared_error(valid_true, valid_pred)), \
                                                    r2_score(valid_true, valid_pred), \
                                                    mean_absolute_error(valid_true, valid_pred), \
                                                    pearsonr(valid_true, valid_pred)
        test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
                                                r2_score(test_true, test_pred), \
                                                mean_absolute_error(test_true, test_pred), \
                                                pearsonr(test_true, test_pred)

        print('***best %s model***' % repetition_th)
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
            test_rmse, test_r2, test_mae, test_rp[0]))
