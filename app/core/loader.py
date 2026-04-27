from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from app.core.schemas import InputDataset, UserProfile




def load_raw_json(file_path: str | Path) -> Dict[str, Any]:
    """
    Load raw JSON file safely.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def sanitize_dataset(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove hidden answer-key sections if present.
    """
    cleaned = dict(raw_data)

    if "hidden_patterns_reference" in cleaned:
        del cleaned["hidden_patterns_reference"]

    return cleaned




def sort_user_conversations(dataset: InputDataset) -> InputDataset:
    """
    Ensure all conversations are chronological.
    """

    for user in dataset.users:
        user.conversations = sorted(
            user.conversations,
            key=lambda x: x.timestamp
        )

    return dataset




def load_dataset(file_path: str | Path) -> InputDataset:
    """
    Full production pipeline:
    load -> sanitize -> validate -> sort
    """

    raw = load_raw_json(file_path)
    clean = sanitize_dataset(raw)

    dataset = InputDataset.model_validate(clean)
    dataset = sort_user_conversations(dataset)

    return dataset



def get_user(dataset: InputDataset, user_id: str) -> UserProfile:
    """
    Fetch user by ID.
    """
    for user in dataset.users:
        if user.user_id == user_id:
            return user

    raise ValueError(f"User not found: {user_id}")




def dataset_summary(dataset: InputDataset) -> dict:
    return {
        "users": len(dataset.users),
        "total_conversations": sum(len(u.conversations) for u in dataset.users),
        "user_ids": [u.user_id for u in dataset.users]
    }




if __name__ == "__main__":
    data = load_dataset("data/askfirst_synthetic_dataset.json")
    print(dataset_summary(data))
