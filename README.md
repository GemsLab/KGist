# KGist

Caleb Belth, Xinyi Zheng, Jilles Vreeken, and Danai Koutra. _What is Normal, What is Strange, and What is Missing in a Knowledge Graph: Unified Characterization via Inductive Summarization_. ACM The Web Conference (WWW), April 2020. [[pdf](https://arxiv.org/abs/2003.10412)]

Code and Reference Coming Soon

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

`{KG_name}.txt` format space separated, one triple per line.

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

`python main.py --graph nell -n`

### Arguments

`--graph {KG_name}` Expects `{KG_name}.txt` and `{KG_name}_labels.txt` to be in `data/` directory in format as described above for NELL and DBpedia.

`--rule_merging / -Rm True/False (Optional; Default = False)` Use rule merging refinement (Section 4.2.2)

`--rule_nesting / -Rn True/False (Optional; Default = False)` Use rule nesting refinement (Section 4.2.2)

`--idify / -i True/False (Optional; Default = True)` Convert entities and predicates to integer ids internally for faster processing

`--verbosity / -v [0, infinity) (Optional; Default = 1,000,000)` How frequently to log process (use integers)

`--output_path / -o (Optional; Default = '../data/output/')` What directory to write the output to (log will still be printed to stdout)

### Coming Soon

- Documentation on saving and loading models.
- More extensive examples.

### Comments or Questions

Contact Caleb Belth with comments or questions: `cbelth@umich.edu`

`https://quickshift.xyz/`
