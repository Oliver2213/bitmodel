#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub",
#     "tqdm",
#     "requests",
# ]
# ///

import argparse
import datetime
import json

from huggingface_hub import HfApi
from tqdm import tqdm
import requests


class ModelProcessor:
    def __init__(self, args):
        self.args = args
        self.api = HfApi()

    def fetch_and_filter_models(self):
        kwargs = {}
        if self.args.user:
            kwargs["author"] = self.args.user
        if self.args.pipeline_tag:
            kwargs["pipeline_tag"] = self.args.pipeline_tag
        if self.args.params:
            kwargs["num_parameters"] = self.args.params
        if self.args.skip_gated:
            kwargs["gated"] = False

        sort_map = {
            "lastModified": "last_modified",
            "downloads": "downloads",
            "likes": "likes",
            "trending": "trending_score",
        }
        if self.args.sort in sort_map:
            kwargs["sort"] = sort_map[self.args.sort]

        # Use expand to get full metadata in one shot (avoids per-model model_info calls)
        kwargs["expand"] = ["lastModified", "trendingScore", "downloads", "likes", "author"]

        if self.args.limit is not None:
            kwargs["limit"] = self.args.limit

        if self.args.filter:
            seen = set()
            all_models = []
            for term in self.args.filter:
                for m in self.api.list_models(search=term, **kwargs):
                    if m.id not in seen:
                        seen.add(m.id)
                        all_models.append(m)
        else:
            all_models = list(self.api.list_models(**kwargs))

        print(f"Total models found: {len(all_models)}")

        if self.args.verbose:
            for m in all_models:
                parts = [m.id]
                if m.trending_score is not None:
                    parts.append(f"trending={m.trending_score}")
                if m.downloads is not None:
                    parts.append(f"downloads={m.downloads:,}")
                if m.likes is not None:
                    parts.append(f"likes={m.likes}")
                print("  " + " | ".join(parts))

        print("Number of models after filtering:", len(all_models))
        return all_models

    def process_models(self, filtered_models):
        now = datetime.datetime.now(datetime.timezone.utc).date()
        repo_table = []
        for model in tqdm(filtered_models, desc="Processing models"):
            try:
                last_modified = model.last_modified
                if last_modified is None:
                    # Fallback to per-model info if expand didn't populate it
                    info = self.api.model_info(model.id)
                    last_modified = info.last_modified
                if last_modified is None:
                    print(f"Skipping {model.id}: no last_modified date")
                    continue
                if isinstance(last_modified, str):
                    last_modified = datetime.datetime.fromisoformat(last_modified)
                last_modified_date = last_modified.date()
                if (now - last_modified_date).days > self.args.age:
                    print(f"Removed outdated repo: {model.id} : last update: {last_modified_date}")
                    continue
                repo_data = {
                    "name": model.id.replace("/", "#")
                    .replace("_", "#")
                    .replace("-", "#"),
                    "original_name": model.id,
                    "branches": [],
                    "last_update": last_modified.strftime('%Y-%m-%dT%H:%M:%SZ'),
                }
                git_refs = self.api.list_repo_refs(model.id, repo_type="model")
                branches = [b for b in git_refs.branches if not b.name.startswith(".git")]

                non_empty_branch_found = False
                for branch in branches:
                    try:
                        files = self.api.list_repo_files(model.id, revision=branch.name)
                        if files:
                            non_empty_branch_found = True
                            repo_data["branches"].append(
                                {
                                    "name": branch.name,
                                    "files": [
                                        f
                                        for f in files
                                        if not f.endswith(".gitattributes")
                                        and not f.startswith(".git")
                                    ],
                                }
                            )
                    except Exception as e:
                        print(f"Error fetching files for model {model.id} on branch {branch.name}: {e}")
                if non_empty_branch_found:
                    repo_table.append(repo_data)
                else:
                    print(f"Empty repo detected and skipped: {model.id}")
            except Exception as e:
                print(f"Error fetching branches for model {model.id}: {e}")
        return repo_table

    def get_existing_torrents(self, url):
        response = requests.get(url)
        files = response.json()
        file_list = []
        for file in files:
            if file["type"] == "file":
                file_name = (
                    file["name"]
                    .replace(".torrent", "")
                    .replace("/", "#")
                    .replace("_", "#")
                    .replace("-", "#")
                )
                if not file_name.startswith(".gitattributes"):
                    file_list.append(file_name)
        return file_list

    def filter_and_output_repos(self, repo_table, existing_torrents):
        removed_repos = []
        duplicated_repos = []
        output_repos = []
        for repo in repo_table:
            if repo["name"] not in existing_torrents:
                output_repos.append(repo)
            else:
                if self.args.rd:
                    removed_repos.append(repo)
                    print(f"Removed duplicated repo: {repo['original_name']}")
                else:
                    duplicated_repos.append(repo)
        with open(self.args.filename + "-models.json", "w") as f:
            json.dump(
                [{**repo, "name": repo["original_name"]} for repo in output_repos],
                f,
                indent=4,
            )
        if removed_repos:
            with open(self.args.filename + "removed_repos-models.json", "w") as f:
                json.dump(
                    {
                        "removed_repos": [
                            {"name": repo["original_name"]} for repo in removed_repos
                        ],
                        "duplicated_repos": [
                            {"name": repo["original_name"]} for repo in duplicated_repos
                        ],
                    },
                    f,
                    indent=4,
                )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate model lists from Hugging Face Hub for torrent creation."
    )
    parser.add_argument("--user", type=str, default=None, help="Filter by author/org (e.g. mlx-community, TheBloke)")
    parser.add_argument("--filter", type=str, nargs="+", default=[], help="Search terms to match in model names")
    parser.add_argument("--age", type=int, default=30, help="Max age in days (models older than this are skipped)")
    parser.add_argument(
        "--sort", type=str, default="lastModified",
        choices=["lastModified", "downloads", "likes", "trending", "name"],
        help="Sort models by this field",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of models to process")
    parser.add_argument(
        "--pipeline-tag", type=str, default=None,
        help="Filter by pipeline/task (e.g. text-generation, image-classification)",
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="Filter by parameter count (e.g. 'min:1B,max:7B', 'max:3B')",
    )
    parser.add_argument(
        "--trending", action="store_true",
        help="Discovery mode: fetch trending models from HF (no --user or --filter needed). Combines with --pipeline-tag, --params, --limit, etc.",
    )
    parser.add_argument(
        "--skip-gated", action="store_true",
        help="Exclude gated models that require access requests",
    )
    parser.add_argument(
        "--rd", "--remove-duplicates", action="store_true",
        help="Remove models that already have torrents",
    )
    parser.add_argument("--filename", type=str, default=f"string_model_{datetime.date.today().strftime('%d%m%Y')}")
    parser.add_argument("--verbose", action="store_true", help="Show details for each matched model")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.trending:
        # Trending is a standalone discovery mode — force sort and a default limit
        args.sort = "trending"
        if args.limit is None:
            args.limit = 20
    elif not (args.user or args.filter):
        raise ValueError("At least one of --user, --filter, or --trending must be provided")
    processor = ModelProcessor(args)
    filtered_models = processor.fetch_and_filter_models()
    repo_table = processor.process_models(filtered_models)
    existing_torrents = processor.get_existing_torrents(
        "https://api.github.com/repos/Nondzu/LlamaTor/contents/torrents?ref=torrents"
    )
    processor.filter_and_output_repos(repo_table, existing_torrents)


if __name__ == "__main__":
    main()
