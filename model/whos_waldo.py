from collections import defaultdict
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from .ot import optimal_transport_dist
from .uniter_model import (UniterPreTrainedModel, UniterModel)
from toolz.sandbox import unzip
from data import whos_waldo_ot_collate
from data.loader import move_to_cuda

nullid_file = './storage/nullid.npz'
with np.load(nullid_file) as null_load:
    NULLID = null_load['nullid']

with open('./dataset_meta/blurry_bbs.json', 'r', encoding='utf-8') as file:
    blurry_bbs = json.load(file)


def _compute_pad(lens, max_len):
    pad = torch.zeros(len(lens), max_len, dtype=torch.uint8)
    for i, l in enumerate(lens):
        pad.data[i, l:].fill_(1)
    return pad


class WhosWaldo(UniterPreTrainedModel):
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim)
        self.uniter = UniterModel(config, img_dim)

    def contrastive_help(self, inputs):
        # print(">> inside forward")
        (input_ids, img_feats, img_pos_feats, attn_masks, targets, id, img_neg_id, iden2token_pos, gt, num_bb, conf
         ) = map(list, unzip(inputs))
        # print('input_ids: ', input_ids)
        # print('img_feats: ', img_feats)
        # print('img_pos_feats: ', img_pos_feats)
        # print('attn_masks: ', attn_masks)
        # print('id: ', id)
        # print('img_neg_id: ', img_neg_id)
        # print('iden2token_pos: ', iden2token_pos)
        # print('gt: ', gt)
        # print('num_bb: ', num_bb)
        #
        # print("zipping: ")

        input_ids = [x.cpu() for x in input_ids]
        img_feats = [x.cpu() for x in img_feats]
        conf = [x.cpu() for x in conf]
        img_pos_feats = [x.cpu() for x in img_pos_feats]
        attn_masks = [x.cpu() for x in attn_masks]
        targets = [x.cpu() for x in targets]

        b_size = len(input_ids)

        ls = []

        for i in range(b_size):
            c_input_ids = [x.detach().clone() for x in input_ids]
            c_conf = [conf[i].detach().clone() for _ in range(b_size)]
            c_img_feats = [img_feats[i].detach().clone() for _ in range(b_size)]
            c_img_pos_feats = [img_pos_feats[i].detach().clone() for _ in range(b_size)]
            c_attn_masks = [torch.ones(len(x) + num_bb[i], dtype=torch.long) for x in input_ids]
            res = whos_waldo_ot_collate(zip(c_input_ids, c_img_feats, c_img_pos_feats, c_attn_masks, targets, id, img_neg_id, iden2token_pos, gt, num_bb, c_conf))
            ls.append(move_to_cuda(res))

        return ls

    def forward(self, batch, task, null_id=False):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        img_pos_feat = batch['img_pos_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']
        targets = batch['targets']
        ot_inputs = batch['ot_inputs']
        ids = batch['id']
        iden2token_pos = batch['iden2token_pos']
        gt = batch['gt']
        num_bbs = batch['num_bbs']
        conf = batch['conf']

        ls = self.contrastive_help(batch['ori_inputs'])
        # return

        if task == 'matching':
            return self.forward_matching(ls)
        elif task == 'gt':
            # self.forward_matching(ls)
            return self.forward_gt(input_ids, position_ids, img_feat, img_pos_feat,
                                   attention_mask, gather_index,
                                   ot_inputs, ids, iden2token_pos, gt, num_bbs, conf, null_id)
        else:
            raise NotImplementedError('Undefined task for WhosWaldo model')

    def forward_matching(self, ls):
        """
        for 1-1 pairs
        """

        row_sm = []
        col_sm = []
        for batch in ls:
            input_ids = batch['input_ids']
            position_ids = batch['position_ids']
            img_feat = batch['img_feat']
            img_pos_feat = batch['img_pos_feat']
            attention_mask = batch['attn_masks']
            gather_index = batch['gather_index']
            targets = batch['targets']
            ot_inputs = batch['ot_inputs']
            ids = batch['id']
            iden2token_pos = batch['iden2token_pos']
            gt = batch['gt']
            num_bbs = batch['num_bbs']

            T, sim, _ = self.forward_ot(
                input_ids, position_ids, img_feat, img_pos_feat, attention_mask,
                gather_index, ot_inputs, iden2token_pos, use_null_id=False
            )

            # if (not sum([len(x) for x in iden2token_pos]) != 8):
            #     print('sim', sim.shape)
            #     print('iden2token_pos l', [len(x) for x in iden2token_pos])
            #     print('num_bbs ', num_bbs)
            #     print()

            #     for i in range (sim.shape[0]):
            #         print("> new")
            #         print(num_bbs[i], iden2token_pos[i])
            #         print(sim[i])
            #         print(torch.where(sim[i] >= 1.0, 0.0, sim[i]))
            #         print('> next')

            #[[[1 2]
            #  [3 4]
            #  [5 6]]] 

            # paddings are 1.0
            tr = torch.where(sim >= 1.0, 0.0, sim).max(dim=2)[0]
            r_sim_mean = tr.sum(dim=1)/(1e-9 + (tr != 0).sum(dim=1))
            rsm = F.softmax(r_sim_mean, dim=0)
            row_sm.append(rsm)

            tc = torch.where(sim >= 1.0, 0.0, sim).max(dim=1)[0]
            c_sim_mean = tc.sum(dim=1)/(1e-9 + (tc != 0).sum(dim=1))
            csm = F.softmax(c_sim_mean, dim=0)
            col_sm.append(csm)

            # break

        # print("row_sm:", row_sm)
        r_stack = torch.stack(row_sm)
        c_stack = torch.stack(col_sm)
        # print("m_stack:", m_stack)
        eye = torch.eye(len(ls)).cuda()

        r_target_pred = torch.argmax(r_stack, dim=1)  # [B]
        c_target_pred = torch.argmax(c_stack, dim=1)  # [B]
        

        prediction = torch.argmax(eye, dim=1)  # [all_querys]

        # print("target_pred:", target_pred)
        # print("prediction:", prediction)

        r_matching_loss = F.cross_entropy(r_stack, prediction, reduction='none')
        c_matching_loss = F.cross_entropy(c_stack, prediction, reduction='none')
        
        agreement_loss = F.mse_loss(
            torch.diag_embed(r_stack, offset=0, dim1=-2, dim2=-1), 
            torch.diag_embed(c_stack, offset=0, dim1=-2, dim2=-1)
        )

        matching_loss = r_matching_loss + c_matching_loss + 0.15*agreement_loss
        matching_scores = torch.eq(r_target_pred, prediction).sum().cpu().numpy()
        # matching_loss = F.binary_cross_entropy(sigmoid_mean, targets.float(), reduction='none')
        # matching_scores = np.sum(np.where(sigmoid_mean.cpu().detach().numpy() < 0.5, 0, 1) == targets.cpu().detach().numpy())
        
        # print("matching_loss:", matching_loss)
        # print("matching_scores:", matching_scores)

        # print("")

        return matching_loss, matching_scores, (r_matching_loss, c_matching_loss, agreement_loss)

    def forward_ot(self, input_ids, position_ids, img_feat, img_pos_feat,
                   attention_mask, gather_index, ot_inputs, iden2token_pos, use_null_id=False):
        """
        compute similarity matrices
        """

        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attention_mask, gather_index,
                                      output_all_encoded_layers=False)

        ot_scatter = ot_inputs['ot_scatter']

        b = sequence_output.size(0)
        tl = input_ids.size(1)
        il = img_feat.size(1)
        max_l = max(ot_inputs['scatter_max'] + 1, tl + il)

        ot_scatter = ot_scatter.unsqueeze(-1).expand_as(sequence_output)
        ctx_emb = torch.zeros(b, max_l, self.config.hidden_size,
                              dtype=sequence_output.dtype,
                              device=sequence_output.device
                              ).scatter_(dim=1, index=ot_scatter,
                                         src=sequence_output)
        txt_emb = ctx_emb[:, :tl, :]
        img_emb = ctx_emb[:, tl:tl + il, :]

        img_pad = ot_inputs['img_pad']

        # trim txt_emb & txt_pad to only include [NAME] relevant tokens
        batch_size, max_text_len, emb_size = txt_emb.shape
        filtered_matrices = []
        txt_lens = []
        for ex in range(batch_size):
            iden2token_pos_ex = iden2token_pos[ex]
            if use_null_id:
                txt_lens.append(len(iden2token_pos_ex)+1)
            else:
                txt_lens.append(len(iden2token_pos_ex))  # number of identities

            mat_ex = txt_emb[ex]
            filtered_rows = []

            for identity_num, positions in iden2token_pos_ex.items():
                identity_embeddings = []
                for pos in positions:
                    identity_embeddings.append(mat_ex[pos+1])  # +1 as the [CLS] token is the first token
                arr = torch.stack(identity_embeddings, dim=0)
                mean_embedding = torch.mean(arr, axis=0)
                filtered_rows.append(mean_embedding)
            if use_null_id:
                filtered_rows.append(torch.tensor(data=NULLID, requires_grad=True).half().cuda())

            filtered_rows = torch.stack(filtered_rows, dim=0)
            filtered_matrices.append(filtered_rows)

        filtered_matrices = pad_sequence(filtered_matrices, batch_first=True)
        txt_emb = filtered_matrices
        max_tl = max(txt_lens)

        txt_pad = _compute_pad(txt_lens, max_tl).cuda()

        # NOTE: run in fp32 for stability
        T, _, C = optimal_transport_dist(txt_emb.float(), img_emb.float(),
                                         txt_pad, img_pad)

        sim = 1-C
        # sigmoid_sim = torch.sigmoid(sim)
        T = torch.transpose(T, 1, 2)
        # return T, sim, sigmoid_sim
        return T, sim, None

    def forward_gt(self, input_ids, position_ids, img_feat, img_pos_feat,
                   attention_mask, gather_index, ot_inputs,
                   ids, iden2token_pos, gt, num_bbs, conf, use_null_id=False):

        T, sim, _ = self.forward_ot(input_ids, position_ids, img_feat, img_pos_feat,
                                          attention_mask, gather_index, ot_inputs, iden2token_pos, use_null_id)

        gt_id_targets = []
        gt_id_results = []
        gt_face_targets = []
        gt_face_results = []

        null_id_cnt = 0
        null_id_correct = 0
        null_id_pairs_cnt = 0
        null_id_pairs_correct = 0
        null_id_ce_loss = 0

        for batch_idx in range(len(ids)):
            id = ids[batch_idx]
            gt_ex = gt[batch_idx]
            gt_rev_ex = {v: k for k, v in gt_ex.items()}
            box_cnt = num_bbs[batch_idx]
            iden2token_pos_ex = iden2token_pos[batch_idx]

            # print('conf: ', conf[batch_idx])

            # print('position_ids', position_ids)
            # print('id', id)
            # print('gt_ex', gt_ex)
            # print('gt_rev_ex', gt_rev_ex)
            # print('box_cnt', box_cnt)
            # print('iden2token_pos_ex', iden2token_pos_ex)

            # sigmoid_idx = sigmoid_sim[batch_idx]
            sim_idx = sim[batch_idx]
            # print('sim_idx', sim_idx)

            # each row (identity)
            for identity_idx in gt_ex.keys():
                id_row = sim_idx[int(identity_idx)][:box_cnt]
                id_row_sm = F.softmax(id_row)
                gt_id_results.append(id_row_sm)
                gt_id_targets.append(gt_ex[identity_idx])

            # print('gt_id_results', gt_id_results)
            # print('gt_id_targets', gt_id_targets)

            # each column (person detection)
            num_ids = len(iden2token_pos_ex)
            sim_idx_T = torch.transpose(sim_idx, 0, 1)
            for face_idx in gt_rev_ex.keys():
                face_col = sim_idx_T[int(face_idx)][:num_ids]
                face_col_sm = F.softmax(face_col)
                gt_face_results.append(face_col_sm)
                gt_face_targets.append(int(gt_rev_ex[face_idx]))

            # # null identity
            # if use_null_id and num_ids > box_cnt and id in blurry_bbs.keys():
            #     null_id_cnt += 1
            #     blur_list = blurry_bbs[id]

            #     gt_boxes = [int(i) for i in gt_rev_ex.keys()]  # boxes for which we have gt
            #     null_id_row = sigmoid_idx[num_ids][:box_cnt]
            #     null_id_res = []
            #     nullid_targets = []

            #     for i in range(box_cnt):
            #         if i in gt_boxes:  # has ground truth, not a null person!
            #             nullid_targets.append(0.0)
            #             null_id_res.append(null_id_row[i])
            #             null_id_pairs_cnt += 1
            #         elif i in blur_list: # does not have ground truth and is blurry, consider as null person
            #             nullid_targets.append(1.0)
            #             null_id_res.append(null_id_row[i])
            #             null_id_pairs_cnt += 1

            #     null_id_res = torch.tensor(null_id_res)
            #     nullid_targets = torch.tensor(nullid_targets)
            #     example_loss = F.binary_cross_entropy(null_id_res, nullid_targets, reduction='mean')
            #     # average score for example
            #     example_scores = np.mean(np.where(null_id_res.numpy() < 0.5, 0, 1) == nullid_targets.numpy())
            #     # total score for this example
            #     example_total_scores = np.sum(np.where(null_id_res.numpy() < 0.5, 0, 1) == nullid_targets.numpy())

            #     null_id_ce_loss += example_loss
            #     null_id_correct += example_scores
            #     null_id_pairs_correct += example_total_scores

        # no example in the entire batch has ground truth
        if len(gt_id_results) == 0 or len(gt_face_results) == 0:
            return None, None, null_id_cnt, T, sim, None

        gt_id_results = pad_sequence(gt_id_results, batch_first=True, padding_value=0.0)
        gt_id_targets = torch.tensor(gt_id_targets).cuda()
        gt_id_loss = F.cross_entropy(gt_id_results, gt_id_targets, reduction='none')
        gt_id_scores = (gt_id_results.max(dim=-1)[1] == gt_id_targets).sum().item()

        gt_face_results = pad_sequence(gt_face_results, batch_first=True, padding_value=0.0)
        gt_face_targets = torch.tensor(gt_face_targets).cuda()
        gt_face_loss = F.cross_entropy(gt_face_results, gt_face_targets, reduction='none')
        gt_face_scores = (gt_face_results.max(dim=-1)[1] == gt_face_targets).sum().item()

        gt_losses = {
            'gt_row_loss': gt_id_loss,
            'gt_col_loss': gt_face_loss,
            'gt_null_id_loss': None
        }
        gt_scores = {
            'gt_row_scores': gt_id_scores,
            'gt_col_scores': gt_face_scores,
            'gt_null_id_scores': None,
            'gt_id_res': gt_id_results.max(dim=-1)[1] == gt_id_targets,
            'gt_id_res_max': gt_id_results.max(dim=-1)[1],
            'conf': conf,
        }

        # if null_id_cnt != 0:
        #     gt_losses['gt_null_id_loss'] = null_id_ce_loss / null_id_cnt
        #     gt_scores['gt_null_id_scores'] = float(null_id_correct) / float(null_id_cnt)
        #     gt_scores['gt_null_id_gt_pairs_correct'] = null_id_pairs_correct
        #     gt_scores['gt_null_id_gt_pairs_total'] = null_id_pairs_cnt
        return gt_losses, gt_scores, null_id_cnt, T, sim, None
