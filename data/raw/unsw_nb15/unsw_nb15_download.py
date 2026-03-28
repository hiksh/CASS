"""
unsw_nb15_download.py
---------------------
UNSW-NB15 데이터셋을 Kaggle에서 다운로드하여
training-flow.csv / test-flow.csv 를 생성합니다.

실행 위치: data/raw/unsw_nb15/
실행 방법: python unsw_nb15_download.py

Kaggle 원본 파일:
  UNSW_NB15_training-set.csv  →  training-flow.csv
  UNSW_NB15_testing-set.csv   →  test-flow.csv

Kaggle 원본 컬럼 (45개):
  id, dur, proto, service, state, spkts, dpkts, sbytes, dbytes, rate,
  sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit,
  swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean,
  trans_depth, response_body_len, ct_srv_src, ct_state_ttl, ct_dst_ltm,
  ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login,
  ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports,
  attack_cat, label

출력 컬럼 (45개):
  피처 42개 + attack_name, attack_flag, attack_step
"""

import sys
import os
import shutil
import logging
import argparse
import kagglehub

# ── Kaggle 설정 ───────────────────────────────────────────────────────────────
KAGGLE_DATASET = "mrwellsdavid/unsw-nb15"

# 다운로드할 파일 → 출력 파일명 매핑
FILE_MAP = {
    "UNSW_NB15_training-set.csv": "training-flow.csv",
    "UNSW_NB15_testing-set.csv":  "test-flow.csv",
}

# ── 컬럼 설정 ─────────────────────────────────────────────────────────────────

# 원본에서 제거할 컬럼 (행 식별자만 제거, 네트워크 피처는 유지)
DROP_COLS = {
    "id",
}

# 원본 컬럼명 → 출력 컬럼명 변경
RENAME = {
    "proto":      "protocol",
    "attack_cat": "attack_name",
    "label":      "attack_flag",
}

# 출력 피처 컬럼 순서 (Kaggle 원본 컬럼 순서 유지, id만 제외)
FEATURE_COLS = [
    "dur", "protocol", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl",
    "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
    "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

# attack_cat (소문자 정규화 후) → attack_step 매핑
STEP_MAP = {
    "normal":         "benign",
    "analysis":       "lateral-movement",
    "backdoor":       "installation",
    "dos":            "action",
    "exploits":       "infection",
    "fuzzers":        "infection",
    "generic":        "infection",
    "reconnaissance": "reconnaissance",
    "shellcode":      "action",
    "worms":          "action",
}


# ── 다운로드 ──────────────────────────────────────────────────────────────────

def download(tdir: str) -> dict:
    """
    Kaggle에서 데이터셋을 다운로드하고
    필요한 파일만 tdir로 이동합니다.

    Returns:
        {원본 파일명: 로컬 경로} 딕셔너리
    """
    logging.info(f"Kaggle 데이터셋 다운로드: {KAGGLE_DATASET}")
    path = kagglehub.dataset_download(KAGGLE_DATASET, force_download=True)

    moved = {}
    for src_name, dst_name in FILE_MAP.items():
        src = os.path.join(path, src_name)
        dst = os.path.join(tdir, src_name)

        if not os.path.exists(src):
            logging.error(f"파일 없음: {src_name}")
            sys.exit(1)

        if os.path.exists(dst):
            os.remove(dst)
        shutil.move(src, dst)
        moved[src_name] = dst
        logging.info(f"  이동 완료: {src_name}")

    return moved


# ── 변환 ──────────────────────────────────────────────────────────────────────

def convert(src_path: str, dst_path: str) -> None:
    """
    원본 CSV 한 파일을 읽어 컬럼 선택/변환/라벨링 후 저장합니다.

    Args:
        src_path: 원본 CSV 경로 (Kaggle 다운로드 파일)
        dst_path: 출력 CSV 경로 (training-flow.csv 또는 test-flow.csv)
    """
    out_cols = FEATURE_COLS + ["attack_name", "attack_flag", "attack_step"]
    stats = {"written": 0, "skipped": 0}

    with open(src_path, "r", encoding="utf-8") as f_in, \
         open(dst_path, "w", encoding="utf-8") as f_out:

        # 헤더 파싱
        raw_header = f_in.readline().strip().split(",")
        col_index = {c.strip().lower(): i for i, c in enumerate(raw_header)}

        # 원본 컬럼명(소문자) → 출력 컬럼명의 역방향 맵
        reverse_rename = {v: k for k, v in RENAME.items()}

        def get_idx(out_col):
            """출력 컬럼명으로 원본 CSV 인덱스를 반환합니다."""
            raw_col = reverse_rename.get(out_col, out_col)
            return col_index.get(raw_col.lower())

        # 피처 인덱스 사전 계산 (반복 호출 방지)
        feat_indices = []
        for col in FEATURE_COLS:
            idx = get_idx(col)
            if idx is None:
                logging.error(f"원본 파일에 컬럼 없음: '{col}' (원본명: '{reverse_rename.get(col, col)}')")
                sys.exit(1)
            feat_indices.append(idx)

        ac_idx = col_index.get("attack_cat")
        if ac_idx is None:
            logging.error("원본 파일에 'attack_cat' 컬럼 없음")
            sys.exit(1)

        # 출력 헤더 쓰기
        f_out.write(",".join(out_cols) + "\n")

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")

            # attack_cat 정규화
            attack_cat_raw = parts[ac_idx].strip()
            attack_cat = attack_cat_raw.lower()
            if not attack_cat:
                attack_cat = "normal"
                attack_cat_raw = "Normal"

            attack_step = STEP_MAP.get(attack_cat)
            if attack_step is None:
                logging.warning(f"알 수 없는 attack_cat: '{attack_cat_raw}' — 건너뜀")
                stats["skipped"] += 1
                continue

            attack_flag = "0" if attack_cat == "normal" else "1"

            # 피처 값 추출
            feat_vals = [parts[i].strip() for i in feat_indices]

            f_out.write(",".join(feat_vals + [attack_cat_raw, attack_flag, attack_step]) + "\n")
            stats["written"] += 1

    logging.info(
        f"  저장: {os.path.basename(dst_path)} — {stats['written']:,}행 "
        f"(건너뜀: {stats['skipped']:,}행)"
    )


# ── 정리 ──────────────────────────────────────────────────────────────────────

def finalize(moved: dict) -> None:
    """변환에 사용한 원본 Kaggle 파일을 제거합니다."""
    for src_path in moved.values():
        if os.path.exists(src_path):
            os.remove(src_path)
            logging.info(f"  삭제: {os.path.basename(src_path)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def command_line_args():
    parser = argparse.ArgumentParser(
        description="UNSW-NB15 Kaggle 다운로드 및 전처리",
    )
    parser.add_argument(
        "-t", "--target",
        help="출력 디렉토리 (기본: 현재 디렉토리)",
        type=str, default=".",
    )
    parser.add_argument(
        "-l", "--log",
        help="로그 레벨 (DEBUG/INFO/WARNING/ERROR)",
        default="INFO", type=str,
    )
    return parser.parse_args()


def main():
    args = command_line_args()
    logging.basicConfig(level=args.log, format="%(levelname)s  %(message)s")

    tdir = os.path.abspath(args.target)
    if not os.path.exists(tdir):
        logging.error(f"대상 디렉토리가 없습니다: {tdir}")
        sys.exit(1)

    # 1. 다운로드
    moved = download(tdir)

    # 2. 변환
    for src_name, dst_name in FILE_MAP.items():
        src_path = moved[src_name]
        dst_path = os.path.join(tdir, dst_name)
        logging.info(f"변환 중: {src_name} → {dst_name}")
        convert(src_path, dst_path)

    # 3. 원본 파일 정리
    logging.info("원본 파일 정리 중 ...")
    finalize(moved)

    logging.info("완료.")


if __name__ == "__main__":
    main()
