import os

import torch
import torch.nn as nn
import torch.utils.data as data
from datasets import load_dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.utils import save_image
from tqdm import tqdm

import torchutils

def parse_args():
    parser = torchutils.ArgumentParser("Simple training loop.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pretrained training configuration.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["imagenet-1k"],
        help="Dataset to train on. Must be one of: `imagenet-1k`.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet-18"],
        help="Pretrained model name. Must be one of: `resnet-18`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for training (default: 1)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker threads to use in data loading (default: 0)",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        default=False,
        help="Whether to pin memory for data loading (default: False)",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="bullfrog",
        help="Class name (bullfrog).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(".", "output"),
        help="Directory to save final output (default: ./output).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.07,
        help="Perturbation budget (default: 0.07)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Perturbation per step (default: 0.01)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Maximum number of samples to perturb (default: 100)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum number of steps to optimize for (default: 50)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    args = parser.parse_args()    
    return args

def main():
    args = parse_args()
    pad_length = len(str(args.num_samples))

    torchutils.set_seed(args.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Training to fool {args.model}")
    if args.model == "resnet-18":
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.eval()

    model = model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)

    print(f"Training on {args.dataset}")
    if args.dataset == "imagenet-1k":
        hf_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
        filtered_hf_dataset = hf_dataset.filter(
            lambda example: args.class_name in hf_dataset.features["label"].int2str(example["label"])
        )
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        def transform_example(example):
            example["image"] = transform(example["image"])
            return example

        filtered_hf_dataset = filtered_hf_dataset.map(transform_example)
        dataloader = data.DataLoader(
            filtered_hf_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

    criterion = nn.CrossEntropyLoss()

    num_evaluated = 0
    num_correct_on_clean_images = 0
    num_correct_on_perturbed_images = 0
    num_attack_success = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, args.model, exist_ok=True)

    pbar = tqdm(total=args.num_samples, position=0)
    for batch in dataloader:
        remaining = args.num_samples - num_evaluated
        if remaining <= 0:
            break

        original_images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            clean_outputs = model(normalize(original_images))
            clean_predictions = clean_outputs.argmax(dim=1)

        clean_correct = clean_predictions == labels
        num_correct_on_clean_images += clean_correct.sum().item()

        images = original_images.clone()

        for _ in tqdm(
            range(args.max_steps),
            position=1,
            leave=False,
        ):
            images = images.detach().requires_grad_(True)

            outputs = model(normalize(images))
            cost = criterion(outputs, labels)

            model.zero_grad(set_to_none=True)
            cost.backward()

            images = images + args.alpha * images.grad.sign()

            images = torch.max(
                torch.min(
                    images,
                    original_images + args.epsilon,
                ),
                original_images - args.epsilon,
            )

            images = images.clamp(0., 1.).detach()

        with torch.no_grad():
            adv_outputs = model(normalize(images))
            adv_predictions = adv_outputs.argmax(dim=1)

        adv_correct = adv_predictions == labels

        num_correct_on_perturbed_images += adv_correct.sum().item()

        attack_success = clean_correct & (~adv_correct)
        num_attack_success += attack_success.sum().item()

        num_evaluated += images.shape[0]

        pbar.update(images.shape[0])

        save_image(
            images.detach().cpu(),
            os.path.join(
                args.output_dir,
                args.model,
                f"{pbar.n:0{pad_length}d}.png"
            )
        )

    pbar.close()

    original_accuracy = 100.0 * num_correct_on_clean_images / num_evaluated
    accuracy_on_perturbed_images = 100.0 * num_correct_on_perturbed_images / num_evaluated

    if num_correct_on_clean_images > 0:
        attack_success_rate = 100.0 * num_attack_success / num_correct_on_clean_images
    else:
        attack_success_rate = 0.0

    print("\n***** Running training *****")
    print(f"  Samples Evaluated = {num_evaluated}")
    print(f"  Original Accuracy = {original_accuracy:.2f}%")
    print(f"  Accuracy on Perturbed Images = {accuracy_on_perturbed_images:.2f}%")
    print(f"  Attack Success Rate = {attack_success_rate:.2f}%")
    print(f"  Accuracy Drop = {original_accuracy - accuracy_on_perturbed_images:.2f}")

if __name__ == "__main__":
    main()
