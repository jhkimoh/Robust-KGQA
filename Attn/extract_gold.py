import json
import os
from tqdm import tqdm


INPUT_PATH = ""
OUTPUT_PATH = ""


# ============================================================
# ★ gold_entities 추출 함수
# golden_path 내에서 index 0,2,4,... 이 entity임
# ============================================================
def extract_gold_entities(sample):
    """
    sample['golden_path']는 다음과 같은 형태:
        [
          ['John Noble', 'film.actor.film', 'm.03l6qx7', 'film.performance.character', 'Denethor II'],
          ['John Noble', 'film.actor.film', 'm.0528y98', 'film.performance.character', 'Denethor II']
        ]
    여기서 entity는 index 0,2,4,... 의 요소.
    """

    # golden_path가 없는 경우 → tail entity만 gold로 사용
    if "golden_path" not in sample or sample["golden_path"] is None:
        if "a_entity" in sample:
            return set(sample["a_entity"])
        return set()

    gold_entities = set()

    for path in sample["golden_path"]:
        # 예: path = [e1, r1, e2, r2, e3]
        for i, item in enumerate(path):
            if i % 2 == 0:  # entity index
                gold_entities.add(item)

    return gold_entities


# ============================================================
# ★ JSONL 처리 함수
# ============================================================
def process_jsonl(input_path, output_path):

    if os.path.exists(output_path):
        print(f"[Warning] Output file already exists → overwrite: {output_path}")

    total_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
    print(f"Total samples: {total_lines}")

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in tqdm(f_in, total=total_lines, desc="Extracting gold entities"):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)

            # gold entity 추출
            gold_entities = list(extract_gold_entities(sample))

            # sample에 추가
            sample["gold_entities"] = gold_entities

            # 저장
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[Done] Saved processed dataset → {output_path}")


# ============================================================
# ★ 메인 실행부
# ============================================================
if __name__ == "__main__":
    process_jsonl(INPUT_PATH, OUTPUT_PATH)
