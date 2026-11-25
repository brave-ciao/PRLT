from pathlib import Path
from .data import BaseDataset, Sequence, SequenceList
from prlt.lsotb import load_split


class LSOTBDataset(BaseDataset):
    """Exactly the 100 LSOTB-TIR100 sequences from the user's PRLT protocol."""
    def __init__(self, root=None):
        super().__init__()
        self.base_path = str(root or self.env_settings.lsotb_path)
        manifest = Path(__file__).resolve().parents[2] / "train/data_specs/lsotb_tir100.txt"
        self.sequence_list = [s.strip() for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
        if len(self.sequence_list) != 100 or len(set(self.sequence_list)) != 100:
            raise ValueError("LSOTB-TIR100 manifest must contain exactly 100 unique sequences")
        self.records = load_split(self.base_path, names=self.sequence_list)

    def get_sequence_list(self):
        return SequenceList([Sequence(r.name, [str(p) for p in r.frames], "lsotb", r.boxes.copy())
                             for r in self.records])

    def __len__(self):
        return len(self.sequence_list)
