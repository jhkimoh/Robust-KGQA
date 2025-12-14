import torch
import torch.nn.functional as F
from .gnn_utils import *
#! model GNNRetriever 
def test(model, kg_data, question_dataset, device='cuda:0'):
    #! kg_data가 어떻게 생긴건지?
    entity2prob = {}
    graph = [kg_data]
    batch = question_dataset[0]
    ent2id = kg_data['ent2id']
    id2ent = {v : k for k, v in ent2id.items()}

    model.eval()

    entities_weight = None
    batch = {k : v.to(device) for k, v in batch.items()}
    ent_pred, _ = model(graph[0], batch, entities_weight=entities_weight) #! tensor([1,1998])
    target_entities_mask = batch["supporting_entities_masks"].unsqueeze(0)
    target_pred_score = torch.where(target_entities_mask == 0, torch.tensor(-float('inf'), device=device), ent_pred)

    probs = F.softmax(target_pred_score, dim=1)
    scores_np = probs.detach().cpu().numpy()[0]
    candidate_list = []
    for i, score in enumerate(scores_np):
        candidate_list.append((score, 0 + i))

    sorted_candidates = sorted(candidate_list, key=lambda x: x[0], reverse=True)
    entity2prob = {id2ent[ent] : prob for prob, ent in sorted_candidates}
    index_list = [idx for _, idx in sorted_candidates]
    selected_target = [id2ent[idx] for idx in index_list]

    return selected_target, entity2prob
            
