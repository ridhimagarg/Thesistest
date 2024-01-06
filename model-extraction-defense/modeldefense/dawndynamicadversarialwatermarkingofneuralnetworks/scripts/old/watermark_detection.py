import detection
import seaborn as sns
import seaborn as sns

import detection

sns.set_style("whitegrid")

model = "mnist"  # cifar10 (and rn34), mnist
if model == "cifar10":
    ground_path = "attacker_cifar_to_cifar_ws250_base_ground_truth.pkl"
    watermark_path = "attacker_cifar_to_cifar_ws250_base_watermark.pkl"
    legitimate_path = "victim_cifar_base_ground_truth.pkl"
elif model == "cifar10rn34":
    ground_path = "attacker_cifar_to_cifar_ws250_rn34_ground_truth.pkl"
    watermark_path = "attacker_cifar_to_cifar_ws250_rn34_watermark.pkl"
    legitimate_path = "victim_cifar_rn34_ground_truth.pkl"
elif model == "mnist":
    ground_path = "attacker_mnist_to_mnist_ws250_l5_ground_truth.pkl"
    watermark_path = "attacker_mnist_to_mnist_ws250_l5_watermark.pkl"
    legitimate_path = "victim_mnist_l5_ground_truth.pkl"
else:
    raise Exception("Mistakes were made.")

batched = True

root = "data/detection/"


watermark = detection.load_file(root + watermark_path)
ground = detection.load_file(root + ground_path)
legitimate = detection.load_file(root + legitimate_path)

if batched:
    ground = [img for batch in ground for img in batch]
    watermark = [img for batch in watermark for img in batch]
    legitimate = [img for batch in legitimate for img in batch]