import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################
# ScoreFusion
# -------------------------------------------------------------
# Input:
#   q_emb      : [1024] (question embedding from text encoder)
# Output:
#   w          : [2]    (weights for mpnet & gnn scores)
###############################################################
class ScoreFusion(nn.Module):
    def __init__(self, q_dim=1024, hidden_dim=256):
        super().__init__()

        # attention-like layer
        self.attn = nn.Sequential(
            nn.Linear(q_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)   # output: [w_mp, w_gnn]
        )

    def forward(self, q_emb):
        # q_emb: [1024]
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)      # → [1, 1024]

        w = self.attn(q_emb)               # → [1, 2]
        w = F.softmax(w, dim=-1)           # weight sum = 1
        return w.squeeze(0)                # → [2]


###############################################################
# Helper: match score dict keys to tensor in same order
###############################################################
def dict_to_tensor(score_dict, entity_list, device):
    """
    score_dict = {entity_string: score}
    entity_list = [e1, e2, ...] (target entities)
    Return: tensor [len(entity_list)]
    """
    scores = []
    for ent in entity_list:
        if ent in score_dict:
            scores.append(score_dict[ent])
        else:
            scores.append(0.0)
    return torch.tensor(scores, dtype=torch.float32, device=device)


###############################################################
# fuse_and_train
# -------------------------------------------------------------
# 학습 모드: 
#   1) q_text → q_emb
#   2) mp_scores / gnn_scores → tensor로 변환
#   3) w = fusion(q_emb)
#   4) fused_score = w[0]*mp + w[1]*gnn
#   5) gold_entities로 BCE loss 계산
###############################################################
def fuse_and_train(
    q_text,
    mp_scores,
    gnn_scores,
    gold_entities,
    text_encoder,
    fusion_model,
    optimizer,
    criterion,
    train_mode,
    device="cuda:0"
):
    """
    gold_entities: set([...])
    mp_scores: {entity: score}
    gnn_scores: {entity: score}
    """

    # -----------------------------
    # 1) Candidate entity union
    # -----------------------------
    all_entities = list(set(mp_scores.keys()) | set(gnn_scores.keys()))
    if len(all_entities) == 0:
        return {}, None

    # -----------------------------
    # 2) Score tensors
    # -----------------------------
    mp_vec = dict_to_tensor(mp_scores, all_entities, device)      # [N]
    gnn_vec = dict_to_tensor(gnn_scores, all_entities, device)    # [N]

    # -----------------------------
    # 3) gold label tensor
    # -----------------------------
    gold = torch.zeros(len(all_entities), device=device)
    for i, ent in enumerate(all_entities):
        if ent in gold_entities:
            gold[i] = 1.0

    # -----------------------------
    # 4) question embedding
    # -----------------------------
    with torch.no_grad():
        q_emb = text_encoder(q_text).to(device)   # [1, 1024] or [1024]
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)

    # -----------------------------
    # 5) fusion weight
    # -----------------------------
    w = fusion_model(q_emb)      # [2]
    w_mp, w_gnn = w[0], w[1]

    # -----------------------------
    # 6) fused score
    # -----------------------------
    fused = w_mp * mp_vec + w_gnn * gnn_vec       # [N]

    # inference-only mode
    if not train_mode:
        fused_dict = {ent: fused[i].item() for i, ent in enumerate(all_entities)}
        return fused_dict, None

    # -----------------------------
    # 7) loss 계산
    # -----------------------------
    loss = criterion(fused, gold)

    # -----------------------------
    # 8) backward + update
    # -----------------------------
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -----------------------------
    # 9) return fused score dict + loss value
    # -----------------------------
    fused_dict = {ent: fused[i].item() for i, ent in enumerate(all_entities)}

    return fused_dict, loss.item()


