import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def best_pos_distance(query, pos_vecs):
    num_pos = pos_vecs.shape[0]
    query_copies = query.repeat(int(num_pos), 1)  # 重复增加列数，以进行矩阵运算
    diff = ((pos_vecs - query_copies) ** 2).sum(1)

    min_pos, _ = diff.min(0)  # easiest positive
    max_pos, _ = diff.max(0)  # hardest positive
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=False, ignore_zero_loss=False):
    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    negative = ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)  # query与所有neg的距离之和

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(0)[0]
    else:
        triplet_loss = loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss
    return triplet_loss


def quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg, m1, m2, use_min=False, lazy=False, ignore_zero_loss=False):

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos

    num_neg = neg_vecs.shape[0]
    query_copies = q_vec.repeat(int(num_neg), 1)
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    loss = m1 + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)
    loss = loss.clamp(min=0.0)
    if lazy:
        triplet_loss = loss.max(0)[0]
    else:
        triplet_loss = loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss

    other_neg_copies = other_neg.repeat(int(num_neg), 1)
    second_loss = m2 + positive - ((neg_vecs - other_neg_copies) ** 2).sum(1).unsqueeze(1)
    second_loss = second_loss.clamp(min=0.0)
    if lazy:
        second_loss = second_loss.max(0)[0]
    else:
        second_loss = second_loss.sum(0) / num_neg
    if ignore_zero_loss:
        hard_second = torch.gt(second_loss, 1e-16).float()
        num_hard_second = torch.sum(hard_second)
        second_loss = second_loss.sum() / (num_hard_second + 1e-16)
    else:
        second_loss = second_loss

    total_loss = triplet_loss + second_loss

    return total_loss


# class StructureAware:
#     def __init__(self):
#         # self.orig_weight = 100
#         # self.weight = 100
#         self.margin = 0.0002
#
#         # # Constraint relaxation
#         # gamma = 10
#         # lin = torch.linspace(0, 1, 20)
#         # exponential_factor = gamma*(lin - 0.5)
#         # self.weight_factors = 1 / (1 + exponential_factor.exp())
#
#     # def adjust_weight(self, epoch):
#     #     if configs.train.loss.incremental.adjust_weight:
#     #         self.weight = self.orig_weight * self.weight_factors[epoch - 1]
#     #     else:
#     #         pass
#
#     def __call__(self, old_rep, new_rep):
#         with torch.no_grad():
#             old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1)  # B x D x D
#             norm_old_vec = F.normalize(old_vec, p=2, dim=2)
#             old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1, 2)).view(-1)
#
#         new_vec = new_rep.unsqueeze(0) - new_rep.unsqueeze(1)
#         norm_new_vec = F.normalize(new_vec, p=2, dim=2)
#         new_angles = torch.bmm(norm_new_vec, norm_new_vec.transpose(1,2)).view(-1)
#
#         loss_incremental = F.smooth_l1_loss(new_angles, old_angles, reduction='none')
#         loss_incremental = F.relu(loss_incremental - self.margin)
#
#         # Remove 0 terms from loss which emerge due to margin
#         # Only do if there are any terms where inc. loss is not zero
#         if torch.any(loss_incremental > 0):
#             loss_incremental = loss_incremental[loss_incremental > 0]
#
#         # loss_incremental = self.weight * loss_incremental.mean()
#         loss_incremental = loss_incremental.mean()
#         return loss_incremental


def structure_aware_loss(old_rep, new_rep):
    with torch.no_grad():
        old_vec = old_rep.unsqueeze(0) - old_rep.unsqueeze(1)  # B x D x D
        norm_old_vec = F.normalize(old_vec, p=2, dim=2)
        old_angles = torch.bmm(norm_old_vec, norm_old_vec.transpose(1, 2)).view(-1)

    new_vec = new_rep.unsqueeze(0) - new_rep.unsqueeze(1)
    norm_new_vec = F.normalize(new_vec, p=2, dim=2)
    new_angles = torch.bmm(norm_new_vec, norm_new_vec.transpose(1, 2)).view(-1)

    loss_incremental = F.smooth_l1_loss(new_angles, old_angles, reduction='none')
    loss_incremental = F.relu(loss_incremental - 0.0002)

    # Remove 0 terms from loss which emerge due to margin
    # Only do if there are any terms where inc. loss is not zero
    if torch.any(loss_incremental > 0):
        loss_incremental = loss_incremental[loss_incremental > 0]

    # loss_incremental = self.weight * loss_incremental.mean()
    loss_incremental = loss_incremental.mean()
    return loss_incremental
