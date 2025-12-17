import argparse
import os
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from dataset import FpcDataset
from model.simulator import Simulator
from utils.noise import get_velocity_noise
from utils.utils import NodeType

import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default="data")
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--noise_std', type=float, default=2e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    parser.add_argument('--log_dir', type=str, default="runs")
    return parser.parse_args()


def train_one_epoch(model: Simulator, dataloader, optimizer, transformer, device, noise_std):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for graph in tqdm.tqdm(dataloader, disable=(dist.get_rank() != 0)):
        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.x[:, 0]
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)

        mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
        errors = ((predicted_acc - target_acc) ** 2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    total_loss_tensor = torch.tensor(total_loss, device=device)
    num_batches_tensor = torch.tensor(num_batches, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / num_batches_tensor.item()
    return avg_loss


@torch.no_grad()
def evaluate(model: Simulator, dataloader, transformer, device):
    model.eval()
    local_losses = []

    for graph in dataloader:
        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.x[:, 0]
        predicted_velocity = model(graph, None)

        mask = torch.logical_or(node_type == NodeType.NORMAL, node_type == NodeType.OUTFLOW)
        errors = ((predicted_velocity - graph.y) ** 2)[mask]
        loss = torch.sqrt(torch.mean(errors))
        local_losses.append(loss.item())

    world_size = dist.get_world_size()
    gathered_losses = [None] * world_size
    dist.all_gather_object(gathered_losses, local_losses)

    return np.mean(gathered_losses)


def main_worker(local_rank, world_size, args):

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    dist.init_process_group(backend='nccl', world_size=world_size, rank=local_rank)

    if local_rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

    transformer = T.Compose([
        T.FaceToEdge(),
        T.Cartesian(norm=False),
        T.Distance(norm=False)
    ])


    train_dataset = FpcDataset(data_root=args.dataset_dir, split='train')
    valid_dataset = FpcDataset(data_root=args.dataset_dir, split='valid')

    #With DistributedSampler
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        sampler=valid_sampler,
        num_workers=2,
        pin_memory=True
    )

    simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
    simulator = simulator.to(device)
    simulator = DDP(simulator, device_ids=[local_rank])

    optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

    writer = None
    if local_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_loss = float('inf')
    best_epoch = -1

    for epoch in range(1, args.num_epochs + 1):
        train_sampler.set_epoch(epoch)

        train_loss = train_one_epoch(simulator, train_loader, optimizer, transformer, device, args.noise_std)
        valid_loss = evaluate(simulator, valid_loader, transformer, device)

        if local_rank == 0:
            print(f"Epoch {epoch}/{args.num_epochs} Train Loss: {train_loss:.2e} Valid Loss: {valid_loss:.2e}")
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/valid', valid_loss, epoch)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': simulator.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_loss': valid_loss,
                }, checkpoint_path)
                print(f"  -> New best model saved at epoch {epoch} with valid loss {valid_loss:.2e}")

    if local_rank == 0:
        writer.close()
        print(f"\nTraining finished. Best model at epoch {best_epoch} with validation loss {best_valid_loss:.2e}")

    dist.destroy_process_group()


def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1 and local_rank == 0:
        raise NotImplementedError("Single-GPU mode not implemented in this DDP script.")
    main_worker(local_rank, world_size=world_size, args=args)


if __name__ == '__main__':
    args = parse_args()
    main(args)