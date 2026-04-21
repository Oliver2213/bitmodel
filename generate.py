#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "huggingface-hub",
#     "tqdm",
#     "requests",
# ]
# ///

import datetime
import json

import click
from huggingface_hub import HfApi
from tqdm import tqdm
import requests


class ModelProcessor:
    def __init__(self, user, filter, age, sort, limit, pipeline_tag, params,
                 trending, skip_gated, remove_duplicates, filename, verbose):
        self.user = user
        self.filter = filter
        self.age = age
        self.sort = sort
        self.limit = limit
        self.pipeline_tag = pipeline_tag
        self.params = params
        self.trending = trending
        self.skip_gated = skip_gated
        self.remove_duplicates = remove_duplicates
        self.filename = filename
        self.verbose = verbose
        self.api = HfApi()

    def fetch_and_filter_models(self):
        kwargs = {}
        if self.user:
            kwargs["author"] = self.user
        if self.pipeline_tag:
            kwargs["pipeline_tag"] = self.pipeline_tag
        if self.params:
            kwargs["num_parameters"] = self.params
        if self.skip_gated:
            kwargs["gated"] = False

        sort_map = {
            "last-modified": "last_modified",
            "downloads": "downloads",
            "likes": "likes",
            "trending": "trending_score",
        }
        if self.sort in sort_map:
            kwargs["sort"] = sort_map[self.sort]

        # Use expand to get full metadata in one shot (avoids per-model model_info calls)
        kwargs["expand"] = ["lastModified", "trendingScore", "downloads", "likes", "author"]

        if self.limit is not None:
            kwargs["limit"] = self.limit

        if self.filter:
            seen = set()
            all_models = []
            for term in self.filter:
                for m in self.api.list_models(search=term, **kwargs):
                    if m.id not in seen:
                        seen.add(m.id)
                        all_models.append(m)
        else:
            all_models = list(self.api.list_models(**kwargs))

        click.echo(f"Total models found: {len(all_models)}")

        if self.verbose:
            for m in all_models:
                parts = [m.id]
                if m.trending_score is not None:
                    parts.append(f"trending={m.trending_score}")
                if m.downloads is not None:
                    parts.append(f"downloads={m.downloads:,}")
                if m.likes is not None:
                    parts.append(f"likes={m.likes}")
                click.echo("  " + " | ".join(parts))

        click.echo(f"Number of models after filtering: {len(all_models)}")
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
                    click.echo(f"Skipping {model.id}: no last_modified date")
                    continue
                if isinstance(last_modified, str):
                    last_modified = datetime.datetime.fromisoformat(last_modified)
                last_modified_date = last_modified.date()
                if (now - last_modified_date).days > self.age:
                    click.echo(f"Removed outdated repo: {model.id} : last update: {last_modified_date}")
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
                        click.echo(f"Error fetching files for model {model.id} on branch {branch.name}: {e}")
                if non_empty_branch_found:
                    repo_table.append(repo_data)
                else:
                    click.echo(f"Empty repo detected and skipped: {model.id}")
            except Exception as e:
                click.echo(f"Error fetching branches for model {model.id}: {e}")
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
                if self.remove_duplicates:
                    removed_repos.append(repo)
                    click.echo(f"Removed duplicated repo: {repo['original_name']}")
                else:
                    duplicated_repos.append(repo)
        with open(self.filename + "-models.json", "w") as f:
            json.dump(
                [{**repo, "name": repo["original_name"]} for repo in output_repos],
                f,
                indent=4,
            )
        if removed_repos:
            with open(self.filename + "removed_repos-models.json", "w") as f:
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


@click.command()
@click.option("--user", default=None, help="Filter by author/org (e.g. mlx-community, TheBloke).")
@click.option("--filter", multiple=True, help="Search terms to match in model names. Can be repeated.")
@click.option("--age", default=30, show_default=True, help="Max age in days (models older than this are skipped).")
@click.option("--sort", default="last-modified",
              type=click.Choice(["last-modified", "downloads", "likes", "trending", "name"], case_sensitive=False),
              show_default=True, help="Sort models by this field.")
@click.option("--limit", default=None, type=int, help="Max number of models to process.")
@click.option("--pipeline-tag", default=None, help="Filter by pipeline/task (e.g. text-generation).")
@click.option("--params", default=None, help="Filter by parameter count (e.g. 'min:1B,max:7B', 'max:3B').")
@click.option("--trending", is_flag=True, help="Discovery mode: fetch trending models (no --user/--filter needed).")
@click.option("--skip-gated", is_flag=True, help="Exclude gated models that require access requests.")
@click.option("--remove-duplicates", is_flag=True, help="Remove models that already have torrents.")
@click.option("--filename", default=f"string_model_{datetime.date.today().strftime('%d%m%Y')}",
              show_default=True, help="Output filename prefix.")
@click.option("--verbose", is_flag=True, help="Show details for each matched model.")
def main(user, filter, age, sort, limit, pipeline_tag, params, trending,
         skip_gated, remove_duplicates, filename, verbose):
    """Generate model lists from Hugging Face Hub for torrent creation."""
    if trending:
        sort = "trending"
        if limit is None:
            limit = 20
    elif not (user or filter):
        raise click.UsageError("At least one of --user, --filter, or --trending must be provided.")

    processor = ModelProcessor(
        user=user, filter=filter, age=age, sort=sort, limit=limit,
        pipeline_tag=pipeline_tag, params=params, trending=trending,
        skip_gated=skip_gated, remove_duplicates=remove_duplicates,
        filename=filename, verbose=verbose,
    )
    filtered_models = processor.fetch_and_filter_models()
    repo_table = processor.process_models(filtered_models)
    existing_torrents = processor.get_existing_torrents(
        "https://api.github.com/repos/Nondzu/LlamaTor/contents/torrents?ref=torrents"
    )
    processor.filter_and_output_repos(repo_table, existing_torrents)


if __name__ == "__main__":
    main()
