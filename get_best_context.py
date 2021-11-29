from argparse import ArgumentParser
from best_contexts import BestContextFinder

parser = ArgumentParser(description="Find the best context for a given query.")
parser.add_argument(
    "-d",
    "--dataset",
    dest="dataset_path",
    help="path to the dataset to use. Must be a json file in SQuAD format",
)
parser.add_argument("query", help="query to search for")

args = parser.parse_args()

if args.dataset_path is None:
    args.dataset_path = "squad1/train-v1.1.json"
finder = BestContextFinder(args.dataset_path)

contexts, _, similarities = finder.get_best_contexts(args.query)

print(contexts[0])
print("Confidence: ", similarities[0] - similarities[1] / 2 - similarities[2] / 4)
