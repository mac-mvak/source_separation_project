import json
import logging


from hw_ss.base.base_dataset import BaseDataset
from hw_ss.utils import ROOT_PATH


logger = logging.getLogger(__name__)





class MixedDataset(BaseDataset):
    def __init__(self, part, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "mixture"
            data_dir.mkdir(exist_ok=True, parents=True)
        index_path = None
        index_path = data_dir / f"{part}_index.json"
        with open(index_path) as f:
            index = json.load(f)
        self._data_dir = data_dir

        super().__init__(index, *args, **kwargs)
