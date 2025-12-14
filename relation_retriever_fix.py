import json
import re
import time
from llm import *
from utils import *
from prompts import *
import argparse
import sys
# import requests
# import atexit
# def send_notification(message):
#     try:
#         requests.post("https://ntfy.sh/gold", data=message.encode(encoding='utf-8'))
#     except Exception as e:
#         print(f"알림 전송 실패: {e}")

# def on_program_exit():
#     print("프로그램 종료! 알림 전송 중...")
#     send_notification("Program Finished (Exit)")

def extract_json_from_response(response_text):
    text = response_text.strip()
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    if match:
        try:
            # ,] 를 ] 로, ,} 를 } 로 치환하는 간단한 정규식
            fixed_str = re.sub(r',\s*\]', ']', match.group(0))
            fixed_str = re.sub(r',\s*\}', '}', fixed_str)
            return json.loads(fixed_str)
        except:
            pass

    print(f"Warning: JSON Parsing Failed completely.\nRaw Output: {text[:100]}...") 
    return []


def run_pruning_experiment(model, question, topic, candidates):
    result = {
        "final_selection": [],
        "reasoning_log": [],
        "method": "LLM"
    }

    if len(candidates) > 3:
        formatted_candidates = "\n".join([f"- {rel}" for rel in candidates])
        prompt = NEW_PRUNING_PROMPT.format(question=question, topic_entity=topic, candidate_relations=formatted_candidates) 
        response_text = model.llm_call(prompt, max_new_token=1024, task='relation', printing=True)
        parsed_data = extract_json_from_response(response_text)
        result["reasoning_log"] = parsed_data
        retrieved_rel = []
        if parsed_data:
            filtered_list = [
                item['relation'] for item in parsed_data 
                if isinstance(item, dict) and item.get('score', 0) >= 7
                and item.get('relation') in candidates
            ]
            retrieved_rel = filtered_list

        retriever_top_k = candidates[:2]
        final_set = set(retrieved_rel) | set(retriever_top_k)
        retrieved_rel = list(final_set)
        result['final_selection'] = retrieved_rel
    elif len(candidates) > 0:
        result["final_selection"] = candidates
        result["method"] = "Pass_Through"
    else:
        result["final_selection"] = ["None"]
        result["method"] = "No_Candidates"

    return result


if __name__ == '__main__':
    # atexit.register(on_program_exit)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rel_ranker_path", type=str, default = '/home/jhkim/kgqa/ProgRAG/data/ckpt/Rel_Retriever') #!#!
    parser.add_argument("--golden_path", type=str, default = '/home/jhkim/kgqa/webqsp_test_goldenpath.jsonl') 
    parser.add_argument("--dataset", type=str, default='webqsp')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--topk", type=int, default=15)
    # LLM related
    parser.add_argument("--is_GPT", action='store_true', default = False)
    parser.add_argument("--llm_model_path", type=str, default='google/gemma-2-9b-it')
    parser.add_argument("--is_8bit", action='store_true', default=False)
    parser.add_argument("--do_uncertainty", action='store_true', default=False)
    args = parser.parse_args()
    model = LLM(args=args)
    #! tail_graph relation_graph
    graph = []
    file_path = f'data/{args.dataset}/total_graph_{args.dataset}.jsonl'
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            graph.append(data)
    total_graph = graph[0]
    tail_graph = get_tail_graph(total_graph)
    relation_graph = get_relation_graph(total_graph)
    pruner = RelationRetriever(ckpt_path = args.rel_ranker_path, device = args.device, tail_graph = tail_graph, rel_graph = relation_graph, topk = args.topk)

    # 1. 실패한 14개 ID 리스트업
    if args.dataset == 'webqsp':
        target_ids = [
            "WebQTest-257", "WebQTest-1597", "WebQTest-504", "WebQTest-114",
            "WebQTest-683", "WebQTest-1510", "WebQTest-284", "WebQTest-1232",
            "WebQTest-1421", "WebQTest-555", "WebQTest-134", "WebQTest-189",
            "WebQTest-1050", "WebQTest-560"
        ]
    else:
        sys.exit("not webqsp dataset!!! sorry... not prepared T_T\n")

    # 2. 데이터 로드 (WebQTest_processed.jsonl 파일 경로 수정 필요)
    data_map = {}
    with open(args.golden_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] in target_ids:
                data_map[item['id']] = item

    # 4. 실험 루프 및 정밀 로깅
    analysis_results = []
    total_hit = []
    for q_id in target_ids:
        if q_id not in data_map:
            continue
            
        data = data_map[q_id]
        question = data['question']
        visited_ent = data['topic']
        topic_ent = data['topic']
        golden_rel = set()
        for path in data['golden_path']:
            golden_rel.add(path[1])
        golden_rel = list(golden_rel)
        cand_rel, cand_path = pruner.pruning(topic_ent, question, visited_ent)
        if not cand_rel:
            print(f"Warning: {q_id}에 대한 Candidate Relation 정보가 없습니다.")
            continue
        if len(topic_ent) > 0:
            topic_ent_str = ", ".join(topic_ent) # "Thomas Paine"
        else:
            topic_ent_str = "Unknown Entity"
        result_log = run_pruning_experiment(model, question, topic_ent_str, cand_rel)
        hit =  exact_match(result_log['final_selection'], golden_rel)
        total_hit.append(hit)
        # (3) 결과 저장
        analysis_entry = {
            "id": q_id,
            "question": question,
            "input_candidates": cand_rel,
            "llm_reasoning": result_log['reasoning_log'],
            "final_selection": result_log['final_selection'],
            "method": result_log['method'],
            "golden_relations": golden_rel,
            "EM":hit
        }
        analysis_results.append(analysis_entry)
        
        print(f"Processed {q_id}: Selected {len(result_log['final_selection'])} relations ({result_log['method']})")

    # 7. 파일 저장
    output_filename = 'pruner_failure_analysis_v3.json'
    with open(output_filename, 'w') as f:
        json.dump(analysis_results, f, indent=4)
    
    print("EM:", sum(total_hit)/len(total_hit))
    print(f"=== 분석 완료: {output_filename} 저장됨 ===")