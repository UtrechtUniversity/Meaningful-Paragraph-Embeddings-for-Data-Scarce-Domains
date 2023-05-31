import datasets
import numpy as np
from ml_utils import h5
import json

BUILDER_CONFIGS = []


class H5DatasetBuilder(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description="H5DatasetBuilder",
            features = datasets.Features({
                "input_ids": datasets.Sequence(feature=datasets.Value(dtype='int32')),
                "next_sentence_label": datasets.Value(dtype='int32')
            }), 
        )


    def _split_generators(self, dl_manager):

        path = self.config.data_dir + "/" if self.config.data_dir else "./"
        config = json.load(open(path + "config.json", "r"))
      
        filepath = config["filepath"]
        print(f"Reading from file {filepath}")
        splits = [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": path + filepath,
                "split": "train",
                "keys": config["keys"],
            },
        ),
        datasets.SplitGenerator(
            name=datasets.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "filepath": path + filepath,
                "split": "test",
                "keys": config["keys"],
            },
        ),

        ]
        return splits


    def _generate_examples(
        self, filepath, split, keys  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        with h5.HFILE(filepath, mode="r") as hfile:
            for key in keys:
                assert key in hfile.keys(), f"{key} not in HFILE {filepath}!"
            
            assert np.isscalar(hfile.lengths) or all(len(hfile[key]) == len(hfile[keys[0]])   for key in keys), f"({filepath}) not all keys sets have the same lengths: {hfile.lengths}"

            rang = hfile.lengths if np.isscalar(hfile.lengths) else hfile.lengths[keys[0]]
            for i in range(0, rang, 1000000):
                try:
                    df = hfile.getPartAsDf(i, i + 1000000, keys, json_loads=True)
                    dicts = df.to_dict(orient='records')
                    for j, x in enumerate(dicts):
                        yield i + j, x
                except Exception as ex:
                    print(f"Could not retrieve item at {i + j}:")
                    print(ex)

