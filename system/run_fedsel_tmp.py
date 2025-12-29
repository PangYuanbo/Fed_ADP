import os, torch
from types import SimpleNamespace
from fedsel_lab.servers.server_fedsel import ServerFedSel
from flcore.trainmodel.models import LocalModel, FedAvgCNN

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = SimpleNamespace()
args.device = device
args.device_id = '0'
args.dataset = 'cifar-10-normal'
args.alpha = 0.5
args.num_clients = 11
args.join_ratio = 1.0
args.random_join_ratio = False
args.global_rounds = 50
args.eval_gap = 1
args.batch_size = 32
args.local_learning_rate = 0.01
args.local_steps = 1
args.num_classes = 10
args.lamda = 0.0

args.difference_privacy = True
args.epsilon = 1.0
args.delta = 1e-6
args.clip_value = 0.005
args.layer_noise_mode = 'global'
args.global_noise = False
args.threshold_high = 0.6
args.threshold_low = 0.4

args.fedsel_mechanism = os.environ.get('FEDSEL_MECHANISM', 'PE')
args.fedsel_k = 100
args.fedsel_beta = 0.9
fedsel_mu_env = os.environ.get('FEDSEL_MU')
args.fedsel_mu = float(fedsel_mu_env) if fedsel_mu_env is not None else 0.1

args.enable_selective_upload = False
args.gradient_keep_ratio = 0.3
args.noise_multiplier = 1.0

args.enable_mia = False

feature_extractor = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
classifier = torch.nn.Linear(512, args.num_classes)
backbone = torch.nn.Sequential(
    feature_extractor.conv1,
    feature_extractor.conv2,
    torch.nn.Flatten(),
    feature_extractor.fc1,
)
args.model = LocalModel(backbone, classifier).to(device)

print(f'Using device: {device}')
print(f'Dataset: {args.dataset}, clients: {args.num_clients}, rounds: {args.global_rounds}')

server = ServerFedSel(args, times=0)
server.train(args)
