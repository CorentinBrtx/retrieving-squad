from argparse import ArgumentParser
from best_contexts import BestContextFinder

parser = ArgumentParser(description="Find the best context for a given query.")
parser.add_argument("-n", "--n_contexts", dest="n_contexts", help="Number of contexts to output.")
parser.add_argument(
    "-d",
    "--dataset",
    dest="dataset_path",
    help="path to the dataset to use. Must be a json file in SQuAD format. Default used is the SQuAD v1.1 training dataset.",
)
parser.add_argument(
    "-m",
    "--method",
    dest="method",
    help="Method to use to encode the sentences. Can be 'transformers' or 'tfidf'. Default used is 'transformers'. See README for more information.",
)
parser.add_argument(
    "-fc",
    "--force_compute",
    dest="recompute",
    type=bool,
    help="Only for transformers. Force computation of paragraphs embeddings, even if a cache version is available.",
)

parser.add_argument("query", help="query to use to find the best context")

args = parser.parse_args()

if args.dataset_path is None:
    args.dataset_path = "squad1/train-v1.1.json"
if args.method is None:
    args.method = "transformers"
if args.recompute is None:
    args.recompute = False

finder = BestContextFinder(
    dataset_path=args.dataset_path, method=args.method, recompute=args.recompute
)

contexts, _, similarities = finder.get_best_contexts(args.query)

if args.n_contexts is not None:
    for i in range(int(args.n_contexts)):
        print("Context n°", i + 1, ":")
        print("-------------------")
        print(contexts[i])
        print("Similarity score: ", round(similarities[i] * 100, 2), "%")
        print("-------------------")

else:
    print(contexts[0])
    print("Confidence: ", (similarities[0] - similarities[1] / 2 - similarities[2] / 4) * 100, "%")
