import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import create_engine, text

from backend.config import SQLALCHEMY_DATABASE_URI

# ==============================
# 0. 재현성(결정성) 고정
# ==============================
SEED = 42

def _seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

_seed_everything(SEED)

# ==============================
# 1. 하이퍼파라미터 설정
# ==============================
EMBED_DIM = 32
N_EPOCHS = 5
BATCH_SIZE = 1024
LR = 1e-3
NEGATIVE_RATIO = 4


# ==============================
# 2. DB에서 인터랙션 + 전체 아이템 로딩
# ==============================
def load_interactions_and_items():
    engine = create_engine(SQLALCHEMY_DATABASE_URI)

    items_sql = """
        SELECT rcp_sno AS item_id
        FROM recipes
        ORDER BY rcp_sno ASC
    """
    all_items_df = pd.read_sql(items_sql, engine)
    if all_items_df.empty:
        raise RuntimeError("recipes 테이블에 레시피가 없습니다. 먼저 CSV 적재를 확인하세요.")
    all_item_ids = all_items_df["item_id"].astype(int).tolist()

    inter_sql = """
        SELECT
            user_id,
            rcp_sno AS item_id,
            preference_type
        FROM user_references
        ORDER BY user_id ASC, rcp_sno ASC
    """
    df = pd.read_sql(inter_sql, engine)

    if df.empty:
        raise RuntimeError("USER_REFERENCES 테이블에 데이터가 없습니다.")

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    item_set = set(all_item_ids)
    before = len(df)
    df = df[df["item_id"].isin(item_set)].copy()
    after = len(df)
    if after == 0:
        raise RuntimeError("USER_REFERENCES의 item_id가 recipes에 존재하지 않습니다.")
    if after != before:
        print(f"recipes에 없는 item_id {before - after}건을 학습에서 제외했습니다.")

    df["weight"] = df["preference_type"].apply(lambda t: 2 if t == "LIKE" else 1)

    return df, all_item_ids, engine


# ==============================
# 3. ID → index 매핑 생성
# ==============================
def build_id_mappings(df: pd.DataFrame, all_item_ids: List[int]):
    unique_users = sorted(df["user_id"].unique().tolist())
    all_item_ids = sorted([int(x) for x in all_item_ids])

    user2idx: Dict[int, int] = {u: i for i, u in enumerate(unique_users)}
    item2idx: Dict[int, int] = {it: i for i, it in enumerate(all_item_ids)}

    idx2user: Dict[int, int] = {i: u for u, i in user2idx.items()}
    idx2item: Dict[int, int] = {i: it for it, i in item2idx.items()}

    return user2idx, item2idx, idx2user, idx2item


# ==============================
# 4. BPR Dataset
# ==============================
class BPRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        user2idx: Dict[int, int],
        item2idx: Dict[int, int],
        negative_ratio: int = 4,
    ):
        self.user2idx = user2idx
        self.item2idx = item2idx
        self.negative_ratio = negative_ratio

        self.user_pos_items: Dict[int, set] = defaultdict(set)
        self.positive_pairs: List[Tuple[int, int]] = []

        for row in df.itertuples(index=False):
            u_idx = user2idx[int(row.user_id)]
            i_idx = item2idx[int(row.item_id)]
            self.user_pos_items[u_idx].add(i_idx)
            self.positive_pairs.append((u_idx, i_idx))

        self.num_items = len(item2idx)

        self.user_neg_candidates: Dict[int, List[int]] = {}
        all_items = set(range(self.num_items))
        for u_idx, pos_set in self.user_pos_items.items():
            negs = list(all_items - pos_set)
            self.user_neg_candidates[u_idx] = negs

        for u_idx, negs in self.user_neg_candidates.items():
            if len(negs) == 0:
                raise RuntimeError(
                    f"BPR negative sampling 불가: 유저 index {u_idx}가 아이템 우주 전체에 대해 이벤트를 보유."
                    f" (현재 아이템 크기={self.num_items})"
                )

    def __len__(self):
        return len(self.positive_pairs) * self.negative_ratio

    def __getitem__(self, idx):
        pos_index = idx // self.negative_ratio
        u, i_pos = self.positive_pairs[pos_index]
        negs = self.user_neg_candidates[u]
        j_neg = random.choice(negs)
        return (
            torch.LongTensor([u]),
            torch.LongTensor([i_pos]),
            torch.LongTensor([j_neg]),
        )


# ==============================
# 5. BPR 모델
# ==============================
class BPRModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def forward(self, u, i, j):
        u_e = self.user_embed(u)
        i_e = self.item_embed(i)
        j_e = self.item_embed(j)
        pos_score = (u_e * i_e).sum(dim=-1)
        neg_score = (u_e * j_e).sum(dim=-1)
        return pos_score, neg_score

    def bpr_loss(self, pos_score, neg_score, l2_lambda: float = 1e-4):
        diff = pos_score - neg_score
        loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-8))
        l2_reg = sum(torch.sum(p ** 2) for p in self.parameters())
        return loss + l2_lambda * l2_reg


# ==============================
# 6. 학습
# ==============================
def train_bpr():
    _seed_everything(SEED)

    df, all_item_ids, engine = load_interactions_and_items()
    print(f"USER_REFERENCES 로드 완료: {len(df)} rows")
    print(f"recipes(아이템) 크기: {len(all_item_ids)}")

    user2idx, item2idx, idx2user, idx2item = build_id_mappings(df, all_item_ids)
    num_users = len(user2idx)
    num_items = len(item2idx)
    print(f"유저 수: {num_users}, 레시피 수: {num_items}")

    dataset = BPRDataset(
        df[["user_id", "item_id", "weight"]],
        user2idx=user2idx,
        item2idx=item2idx,
        negative_ratio=NEGATIVE_RATIO,
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPRModel(num_users, num_items, embed_dim=EMBED_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for u, i_pos, j_neg in dataloader:
            u, i_pos, j_neg = u.to(device), i_pos.to(device), j_neg.to(device)
            optimizer.zero_grad()
            pos_score, neg_score = model(u, i_pos, j_neg)
            loss = model.bpr_loss(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"[Epoch {epoch}/{N_EPOCHS}] loss = {avg_loss:.4f}")

    print("학습 완료")

    user_emb = model.user_embed.weight.detach().cpu().numpy()
    item_emb = model.item_embed.weight.detach().cpu().numpy()

    save_embeddings_to_db(engine, user_emb, item_emb, idx2user, idx2item)


# ==============================
# 7. 임베딩 DB 저장
# ==============================
def save_embeddings_to_db(
    engine,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    idx2user: Dict[int, int],
    idx2item: Dict[int, int],
):
    def vec_to_str(v: np.ndarray) -> str:
        return ",".join(f"{x:.6f}" for x in v.tolist())

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM user_embeddings"))
        conn.execute(text("DELETE FROM recipe_embeddings"))

    user_rows = [
        {"user_id": int(user_id), "vector": vec_to_str(user_emb[idx])}
        for idx, user_id in idx2user.items()
    ]
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO user_embeddings (user_id, vector) VALUES (:user_id, :vector)"),
            user_rows
        )
    print(f"user_embeddings 저장 완료: {len(user_rows)} rows")

    item_rows = [
        {"rcp_sno": int(item_id), "vector": vec_to_str(item_emb[idx])}
        for idx, item_id in idx2item.items()
    ]

    CHUNK_SIZE = 1000
    with engine.begin() as conn:
        for i in range(0, len(item_rows), CHUNK_SIZE):
            chunk = item_rows[i:i + CHUNK_SIZE]
            conn.execute(
                text("INSERT INTO recipe_embeddings (rcp_sno, vector) VALUES (:rcp_sno, :vector)"),
                chunk
            )
            print(f"recipe_embeddings 저장 중: {min(i + CHUNK_SIZE, len(item_rows))}/{len(item_rows)}")

    print(f"recipe_embeddings 저장 완료: {len(item_rows)} rows")


if __name__ == "__main__":
    train_bpr()