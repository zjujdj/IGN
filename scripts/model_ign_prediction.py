from graph_constructor import *
import argparse
from utils import *
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from model_v2 import IGN
from sklearn.metrics import mean_absolute_error, mean_squared_error


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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


model_files = ['2021-07-10_21_59_17_51270.pth', '2021-07-10_23_20_19_61978.pth', '2021-07-11_01_12_59_947126.pth',
               '2021-07-11_03_20_09_152747.pth',  '2021-07-11_04_43_30_960805.pth']
path_marker = '/'


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=500, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=10, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=70, help="early stopping patience")
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
    argparser.add_argument('--num_process', type=int, default=8,
                           help='number of process for generating graphs')
    argparser.add_argument('--dic_path_suffix', type=str, default='0')

    # paras acsf setting
    argparser.add_argument('--EtaR', type=float, default=4.00, help='EtaR')
    argparser.add_argument('--ShfR', type=float, default=3.17, help='ShfR')
    argparser.add_argument('--Zeta', type=float, default=8.00, help='Zeta')
    argparser.add_argument('--ShtZ', type=float, default=3.14, help='ShtZ')

    # prediction data file
    argparser.add_argument('--test_file_path', type=str, default='/mnt/g/02ReData201904034/hxp/github_project/input_data/user2', help='the testing data path')

    args = argparser.parse_args()


    lr, epochs, batch_size, num_workers = args.lr, args.epochs, args.batch_size, args.num_workers
    tolerance, patience, l2, repetitions = args.tolerance, args.patience, args.l2, args.repetitions

    # paras for model
    node_feat_size, edge_feat_size_2d, edge_feat_size_3d = args.node_feat_size, args.edge_feat_size_2d, args.edge_feat_size_3d
    graph_feat_size, num_layers = args.graph_feat_size, args.num_layers
    outdim_g3, d_FC_layer, n_FC_layer, dropout, n_tasks = args.outdim_g3, args.d_FC_layer, args.n_FC_layer, args.dropout, args.n_tasks
    dic_path_suffix = args.dic_path_suffix
    num_process = args.num_process

    # paras for acsf setting
    EtaR, ShfR, Zeta, ShtZ = args.EtaR, args.ShfR, args.Zeta, args.ShtZ

    home_path = args.test_file_path
    zip_files = os.listdir(home_path)  # only one zip file in the user dir
    for file in zip_files:
        if file.endswith('.zip'):
            zip_file = file
    assert zip_file.endswith('.zip'), 'the test file should end with .zip'
    cmdline = 'cd %s && unzip -o %s' % (home_path, zip_file)
    os.system(cmdline)

    mol_files = os.listdir(home_path + path_marker + zip_file[:-4])

    mode = 'mode2'
    for mol_file in mol_files:
        if mol_file.endswith('.sdf') or mol_file.endswith('.pdb'):
            mode = 'mode1'

    # one pdb file and one sdf file
    if mode == 'mode1':
        for file in mol_files:
            if file.endswith('.sdf'):
                sdf_file = file
            else:
                pdb_file = file
        sdfs = Chem.SDMolSupplier(home_path + path_marker + zip_file[:-4] + path_marker + sdf_file)
        sdfs = [_ for _ in sdfs]
        # 写出单个文件
        cmdline = 'cd %s && mkdir -p %s %s %s' % (home_path, 'sdfs', 'pockets', 'complexes')
        os.system(cmdline)
        for sdf in sdfs:
            if sdf:
                sdf_name = sdf.GetProp('_Name')  # 获取配体文件的文件名
                Chem.MolToMolFile(sdf, home_path + path_marker + 'sdfs' + path_marker + '%s.sdf' % (pdb_file[:-4] + '_' + sdf_name))
        ligand_files = os.listdir(home_path + path_marker + 'sdfs')
        ligand_dirs = [home_path + path_marker + 'sdfs' + path_marker + file for file in ligand_files]
        protein_dirs = [home_path + path_marker + zip_file[:-4] + path_marker + pdb_file for _ in ligand_files]
        pocket_out_dirs = [home_path + path_marker + 'pockets' + path_marker + file.replace('.sdf', '_pkt.pdb') for file in ligand_files]
        complex_out_dirs = [home_path + path_marker + 'complexes' + path_marker + file[:-4] for file in ligand_files]

    # multiple pdb files
    if mode == 'mode2':
        ligand_dirs, protein_dirs, pocket_out_dirs,  complex_out_dirs = [], [], [], []
        targets = mol_files  # 每个靶点一个dir
        for target in targets:
            files = os.listdir(home_path + path_marker + zip_file[:-4] + path_marker + target)
            for file in files:
                if file.endswith('.sdf'):
                    sdf_file = file
                elif file.endswith('.pdb'):
                    pdb_file = file
            sdfs = Chem.SDMolSupplier(home_path + path_marker + zip_file[:-4] + path_marker + target + path_marker + sdf_file)
            sdfs = [_ for _ in sdfs]

            # 写出单个sdf文件
            cmdline = 'cd %s && mkdir -p %s' % (home_path + path_marker + zip_file[:-4] + path_marker + target, 'sdfs')
            os.system(cmdline)
            for sdf in sdfs:
                if sdf:
                    sdf_name = sdf.GetProp('_Name')  # 获取配体文件的文件名
                    Chem.MolToMolFile(sdf, home_path + path_marker + zip_file[:-4] + path_marker + target + path_marker + 'sdfs' + path_marker +'%s.sdf' % (pdb_file[:-4] + '_' + sdf_name))
            cur_ligands = os.listdir(home_path + path_marker + zip_file[:-4] + path_marker + target + path_marker + 'sdfs')
            ligand_dirs.extend([home_path + path_marker + zip_file[:-4] + path_marker + target + path_marker + 'sdfs' + path_marker + file for file in cur_ligands])
            protein_dirs.extend([home_path + path_marker + zip_file[:-4] + path_marker + target + path_marker + pdb_file for _ in cur_ligands])
            pocket_out_dirs.extend([home_path + path_marker + 'pockets' + path_marker + file.replace('.sdf', '_pkt.pdb') for file in cur_ligands])
            complex_out_dirs.extend([home_path + path_marker + 'complexes' + path_marker + file[:-4] for file in cur_ligands])

        cmdline = 'cd %s && mkdir -p %s %s' % (home_path, 'pockets', 'complexes')
        os.system(cmdline)
    pool = mp.Pool(num_process)
    pool.starmap(pocket_truncate, zip(protein_dirs, ligand_dirs, pocket_out_dirs, complex_out_dirs))
    pool.close()
    pool.join()

    test_keys = os.listdir(home_path + path_marker + 'complexes')
    test_labels = [0 for _ in test_keys]
    test_dirs = [home_path + path_marker + 'complexes' + path_marker + key for key in test_keys]

    # generating the graph objective using multi process
    test_dataset = GraphDatasetIGN(keys=test_keys, labels=test_labels, data_dirs=test_dirs,
                                   graph_ls_file=home_path + path_marker + 'test_data_ign.bin',
                                   graph_dic_path=home_path + path_marker + 'tmpfiles', num_process=num_process,
                                   dis_threshold=8.00, path_marker=path_marker)
    print('the number of test data:', len(test_dataset))
    total_pred = 0
    for repetition_th in range(repetitions):
        dt = datetime.datetime.now()
        set_random_seed(repetition_th)
        # model
        DTIModel = IGN(node_feat_size=node_feat_size, edge_feat_size=edge_feat_size_3d, num_layers=num_layers,
                       graph_feat_size=graph_feat_size, outdim_g3=outdim_g3,
                       d_FC_layer=d_FC_layer, n_FC_layer=n_FC_layer, dropout=dropout,
                       n_tasks=n_tasks)
        print('number of parameters : ', sum(p.numel() for p in DTIModel.parameters() if p.requires_grad))
        # 加载训练好的模型
        DTIModel.load_state_dict(torch.load(f='../model_save' + path_marker + model_files[repetition_th], map_location='cpu')['model_state_dict'])
        device = torch.device("cuda:%s" % args.gpuid if torch.cuda.is_available() else "cpu")
        DTIModel.to(device)

        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=False, num_workers=num_workers,
                                      collate_fn=collate_fn_ign)
        test_true, test_pred, te_keys, _ = run_a_eval_epoch(DTIModel, test_dataloader, device)

        # metrics
        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        te_keys = np.concatenate(np.array(te_keys), 0).flatten()
        total_pred = total_pred + test_pred

    pd_te = pd.DataFrame({'keys': te_keys, 'test_pred': total_pred/repetitions})
    pd_te.to_csv(home_path + path_marker + 'prediction.csv', index=False)

    # 最后删除用户的无用文件
    files = os.listdir(home_path)
    del_files = []
    for file in files:
        if file.endswith('.csv') or file.endswith('.zip'):
            pass
        else:
            del_files.append(file)
    cmdline = 'cd %s && rm -rf %s' % (home_path, ' '.join(del_files))
    os.system(cmdline)