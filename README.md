# KGist: Knowledge Graph Summarization for Anomaly Detection & Completion

Caleb Belth, Xinyi Zheng, Jilles Vreeken, and Danai Koutra. _What is Normal, What is Strange, and What is Missing in a Knowledge Graph: Unified Characterization via Inductive Summarization_. ACM The Web Conference (WWW), April 2020. 

[[pdf](https://arxiv.org/abs/2003.10412)]

If used, please cite:
```bibtex
@inproceedings{belth2020normal,
  title={What is Normal, What is Strange, and What is Missing in a Knowledge Graph: Unified Characterization via Inductive Summarization},
  author={Belth, Caleb and Zheng, Xinyi and Vreeken, Jilles and Koutra, Danai},
  booktitle={Proceedings of The Web Conference 2020},
  pages={1115--1126},
  year={2020}
}
```

**Presentation**: https://youtu.be/Ql7VEfliPXo

## Setup

1. `git clone git@github.com:GemsLab/KGist.git`
2. `cd data/`
3. `unzip nell.zip`
4. `unzip dbpedia.zip`
5. `cd ../src/`
6. `cd test/`
7. `python tester.py`

## Data

Nell and DBpedia are zipped in the `data/` directory. Yago is too big to distribute via Github.

`{KG_name}.txt` format: space separated, one triple per line.

```
s1 p1 o1
s2 p2 o2
...
```

`{KG_name}_labels.txt` format: space separated, one entity per line followed by a variable number of labels, also space separated.

```
e1 l1 l2 ...
e2 l1 l2 l3 ...
...
```

## Example usage (from `src/` dir)

#### Command Line

`python main.py --graph nell`

#### Interface
```python
from graph import Graph
from searcher import Searcher
from model import Model

# load graph
graph = Graph('nell', idify=True)
# create a Searcher object to search for a model (set of rules)
searcher = Searcher(graph)
# build initial model
model = searcher.build_model()
model.print_stats()
# perform rule merging refinement
model = model.merge_rules()
model.print_stats()
# perform rule nesting refinement
model = model.nest_rules()
model.print_stats()
```

To compute anomaly scores for triples as in Section 4.3:

```python
from anomaly_detector import AnomalyDetector

# construct an anomaly detector with the KGist model
anomaly_detector = AnomalyDetector(model)
# an edge/triple to score
edge = ('concept:company:limited_brands', 'concept:companyceo', 'concept:ceo:leslie_wexner')
anomaly_detector.score_edge(edge)
>>> 26.5164
```

Larger numbers mean more anomalous. Note that in our experiments in Section 5.2, we used KGist+m, which would be the model without running `model.nest_rules()`.

### Arguments

`--graph {KG_name}` Expects `{KG_name}.txt` and `{KG_name}_labels.txt` to be in `data/` directory in format as described above for NELL and DBpedia.

`--rule_merging / -Rm True/False (Optional; Default = False)` Use rule merging refinement (Section 4.2.2)

`--rule_nesting / -Rn True/False (Optional; Default = False)` Use rule nesting refinement (Section 4.2.2)

`--idify / -i True/False (Optional; Default = True)` Convert entities and predicates to integer ids internally for faster processing

`--verbosity / -v [0, infinity) (Optional; Default = 1,000,000)` How frequently to log progress (use integers)

`--output_path / -o (Optional; Default = 'output/')` What directory to write the output to (log will still be printed to stdout)

### Output

- `output/{KG_name}_model.pickle` saves a Model object.
- `output/{KG_name}_model.rules` saves the rules, which are recursively defined, in parenthetical form.

### Coming Soon

- Documentation on loading models.
- More extensive examples.

### Comments or Questions

Contact [Caleb Belth](https://quickshift.xyz/) with comments or questions: `cbelth@umich.edu`
